import torch
import torch.optim as optim #Herramientas de optimización de pesos
from torch.utils.data import DataLoader, Dataset #Para manejar bien datasets
import numpy as np
import pandas as pd
from pathlib import Path 
from .modelo_unet import UNet #Nuestro modelo ya definido

class MRIDataset(Dataset):  #Creamos la clase de Dataset para trabajar más eficientemente
    def __init__(self, csv_path, images_dir, logger=None): #Contiene la ruta al archivo csv y la carpeta donde están las imágenes y máscaras
        self.data = pd.read_csv(csv_path) #Lee el csv y lo convierte en un dataframe
        self.images_dir = Path(images_dir) #Guarda la ruta de la carpeta de imágenes
        self.logger = logger  # ← Guarda logger
        self._diagnostic_done = False

    def __len__(self): #Devuelve cuantos elementos hay en el dataset (importante para ver cuanto hay que iterar)
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] #Saca la fila idx de la tabla
        ruta_procesada = Path(row['ruta_procesada']).name
        ruta_mascara = Path(row['ruta_mascara']).name
        img = np.load(self.images_dir / ruta_procesada)  #row['ruta_procesada'] te da la ruta de la imágen y carga esa imágen  # (Alto, Ancho, 3)
        mask = np.load(self.images_dir / ruta_mascara)   #Lo mismo pero para la máscara # (Alto, Ancho)
        
        # 🔍 DIAGNÓSTICO (una sola vez)
        if not self._diagnostic_done and self.logger:
            self.logger.log(f"🔍 DIAGNÓSTICO DE MÁSCARA:")
            self.logger.log(f"   dtype: {mask.dtype}")
            self.logger.log(f"   min: {mask.min()}, max: {mask.max()}")
            self.logger.log(f"   valores únicos: {np.unique(mask)}")
            self._diagnostic_done = True
        
        # ✅ CONVIERTE a 0/1 si es necesario
        if mask.max() > 1:
            mask = (mask > 0).astype(np.float32)
            if self.logger and not hasattr(self, '_conversion_logged'):
                self.logger.log(f"✅ Máscara convertida de {mask.max()} a 0/1")

        # Convertir a tensor: (Capa, Alto, Ancho)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) #Convierte las imágenes en tensor, el permute es para que pase de (Alto,Ancho,Canal) a (Canal,Alto,Ancho)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) #Unsqueeze añade una dimensión al principio para indicar que la máscara tiene un solo canal
        
        return img, mask  #Se queda con la imagen y su máscara

def entrenar_unet(train_csv, val_csv, images_dir, pesos_clase_path, epochs=1, lr=5e-5, batch_size=8, logger=None):
    """Entrena U-Net con early stopping basado en F1-Score"""
    
    if logger:
        logger.log(f"🚀 Iniciando entrenamiento U-Net")
        logger.log(f"📁 Train CSV: {train_csv}")
        logger.log(f"📁 Val CSV: {val_csv}")
        logger.log(f"📁 Images dir: {images_dir}")
        logger.log(f"⚙️ Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    
    # Cargar datasets
    if logger:
        logger.log("📂 Cargando datasets...")
    
    train_dataset = MRIDataset(train_csv, images_dir, logger=logger)
    val_dataset = MRIDataset(val_csv, images_dir, logger=logger)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if logger:
        logger.log(f"✅ Dataset cargado: {len(train_dataset)} train, {len(val_dataset)} val")
        logger.log(f"📦 Batches por época: {len(train_loader)}")
    
    # Modelo con logger
    model = UNet(entrada=3, salida=1, logger=logger)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if logger:
        logger.log(f"💻 Dispositivo: {device}")
        if device.type == 'cuda':
            logger.log(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar pesos de clase
    pesos_clase = np.load(pesos_clase_path)
    pos_weight = torch.tensor([pesos_clase[1]], dtype=torch.float32).to(device)
    #pos_weight = torch.tensor([175.0], dtype=torch.float32).to(device)
    
    if logger:
        logger.log(f"⚖️ Pesos de clase - Sano: {pesos_clase[0]:.3f}, Tumor: {pesos_clase[1]:.3f}")
        logger.log(f"⚖️ Pos_weight usado: {pos_weight.item():.1f}")
    
    def dice_loss(inputs, target):
        inputs = torch.sigmoid(inputs)
        intersection = (target * inputs).sum()
        union = target.sum() + inputs.sum()
        return 1 - (2 * intersection + 1.0) / (union + 1.0)

    def bce_dice_loss(inputs, target):
        bce = torch.nn.BCEWithLogitsLoss()(inputs, target)  # ← SIN pos_weight
        dice = dice_loss(inputs, target)
        return bce + dice

    criterion = bce_dice_loss
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # ========== EARLY STOPPING BASADO EN F1 ==========
    best_f1 = 0.0
    patience_counter = 0
    patience = 7
    best_model_state = None
    
    for epoch in range(epochs):
        if logger:
            logger.log(f"\n{'='*50}")
            logger.log(f"📈 Época {epoch+1}/{epochs}")
            logger.log(f"{'='*50}")
        
        # ========== ENTRENAMIENTO ==========
        model.train()
        train_loss = 0
        batch_count = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            
            # Log cada 50 batches
            if logger and batch_count % 50 == 0:
                logger.log(f"   🔄 Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== VALIDACIÓN CON MÉTRICAS REALES ==========
        model.eval()
        val_loss = 0
        
        # Almacenar todas las predicciones para calcular métricas
        todas_predicciones = []
        todas_reales = []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # Guardar para métricas
                probs = torch.sigmoid(outputs)
                preds_flat = probs.cpu().numpy().flatten()
                masks_flat = masks.cpu().numpy().flatten()
                todas_predicciones.extend(preds_flat)
                todas_reales.extend(masks_flat)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Convertir a arrays
        y_true = np.array(todas_reales)
        y_prob = np.array(todas_predicciones)
        
        # Probar diferentes umbrales para encontrar el mejor F1
        umbrales = np.arange(0.3, 0.8, 0.05)
        mejor_f1_epoch = 0
        mejor_umbral_epoch = 0.5
        mejor_sens_epoch = 0
        mejor_prec_epoch = 0
        
        for umbral in umbrales:
            y_pred = (y_prob > umbral).astype(int)
            y_true_bin = y_true.astype(int)
            
            tp = np.logical_and(y_pred == 1, y_true_bin == 1).sum()
            fp = np.logical_and(y_pred == 1, y_true_bin == 0).sum()
            fn = np.logical_and(y_pred == 0, y_true_bin == 1).sum()
            
            sensibilidad = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad + 1e-8)
            
            if f1 > mejor_f1_epoch:
                mejor_f1_epoch = f1
                mejor_umbral_epoch = umbral
                mejor_sens_epoch = sensibilidad
                mejor_prec_epoch = precision
        
        if logger:
            logger.log(f"\n📊 Resultados Época {epoch+1}:")
            logger.log(f"   Train Loss: {avg_train_loss:.4f}")
            logger.log(f"   Val Loss:   {avg_val_loss:.4f}")
            logger.log(f"   📈 Mejor F1-Score: {mejor_f1_epoch:.4f} (umbral={mejor_umbral_epoch:.2f})")
            logger.log(f"      Sensibilidad: {mejor_sens_epoch:.4f}")
            logger.log(f"      Precisión:    {mejor_prec_epoch:.4f}")
        
        # ========== EARLY STOPPING BASADO EN F1 ==========
        if mejor_f1_epoch > best_f1:
            best_f1 = mejor_f1_epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            if logger:
                logger.log(f"💾 Mejor modelo guardado! (F1-Score: {best_f1:.4f})")
        else:
            patience_counter += 1
            
            if logger:
                logger.log(f"⚠️ Sin mejora de F1 por {patience_counter}/{patience} épocas")
            
            if patience_counter >= patience:
                if logger:
                    logger.log(f"\n{'='*50}")
                    logger.log(f"🛑 EARLY STOPPING activado en época {epoch+1}")
                    logger.log(f"   Mejor F1-Score logrado: {best_f1:.4f}")
                    logger.log(f"   Restaurando mejor modelo...")
                    logger.log(f"{'='*50}")
                
                # Cargar el mejor modelo
                model.load_state_dict(best_model_state)
                break
    
    if logger:
        logger.log(f"\n✅ Entrenamiento completado!")
        logger.log(f"🏆 Mejor F1-Score: {best_f1:.4f}")
    
    return model


def encontrar_mejor_umbral(model, val_csv, images_dir,device='cpu',batch_size=8, logger=None):
    """
    Encuentra el mejor umbral usando datos de validación
    """
    val_dataset = MRIDataset(val_csv, images_dir, logger=logger)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #Aquí el orden no afecta así que lo ponemos en False
    model.eval()
    
    # Almacenar todas las predicciones y realidades
    todas_predicciones = []
    todas_reales = []
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            outputs = torch.sigmoid(logits)
            
            # Aplanar para tener todos los píxeles
            preds_flat = outputs.cpu().numpy().flatten() #Aplanar la matriz para tener un solo vector
            masks_flat = masks.cpu().numpy().flatten()
            
            todas_predicciones.extend(preds_flat) #Esto es una lista con todos los píxeles
            todas_reales.extend(masks_flat)
    
    # Convertir a arrays
    y_true = np.array(todas_reales) #Con esto lo transformamos en array para poder trabajar con ellos
    y_prob = np.array(todas_predicciones)


    if logger:
        logger.log(f"\n📊 Estadísticas del dataset de validación:")
        total_pixeles = len(y_true)
        pixeles_tumor = y_true.sum()
        pixeles_sano = total_pixeles - pixeles_tumor
        logger.log(f"   Total píxeles: {total_pixeles:,}")
        logger.log(f"   Píxeles con tumor: {pixeles_tumor:,} ({pixeles_tumor/total_pixeles*100:.2f}%)")
        logger.log(f"   Píxeles sanos: {pixeles_sano:,} ({pixeles_sano/total_pixeles*100:.2f}%)")
        logger.log(f"   Rango predicciones: [{y_prob.min():.4f}, {y_prob.max():.4f}]")


    
    # Probar diferentes umbrales
    umbrales = np.arange(0.1, 0.95, 0.05)
    resultados = []
    
    for umbral in umbrales:
        y_pred = (y_prob > umbral).astype(int)
        y_true = y_true.astype(int)

        tp = np.logical_and(y_pred == 1, y_true == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true == 1).sum()

        # Métricas
        sensibilidad = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        especificidad = (total_pixeles - tp - fp - fn) / (total_pixeles - tp - fn + 1e-8)
        f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad + 1e-8)
        
        
        resultados.append({
            'umbral': umbral,
            'sensibilidad': sensibilidad,
            'precision': precision,
            'especificidad': especificidad,
            'f1': f1
        })
    df_resultados = pd.DataFrame(resultados)

    # Usar F1-Score para encontrar el mejor umbral
    mejor_idx = df_resultados['f1'].idxmax()
    mejor_umbral = df_resultados.loc[mejor_idx, 'umbral']
    mejor_f1 = df_resultados.loc[mejor_idx, 'f1']
    mejor_sens = df_resultados.loc[mejor_idx, 'sensibilidad']
    mejor_prec = df_resultados.loc[mejor_idx, 'precision']
    mejor_espec = df_resultados.loc[mejor_idx, 'especificidad']
    
    if logger:
        logger.log(f"\n🎯 MEJOR UMBRAL (por F1-Score): {mejor_umbral:.3f}")
        logger.log(f"   F1-Score: {mejor_f1:.4f}")
        logger.log(f"   Sensibilidad: {mejor_sens:.4f}")
        logger.log(f"   Precisión: {mejor_prec:.4f}")
        logger.log(f"   Especificidad: {mejor_espec:.4f}")
        
        # Mostrar top 5
        logger.log(f"\n📊 TOP 5 umbrales por F1-Score:")
        top5 = df_resultados.nlargest(5, 'f1')[['umbral', 'f1', 'sensibilidad', 'precision']]
        for _, row in top5.iterrows():
            logger.log(f"   Umbral {row['umbral']:.2f}: F1={row['f1']:.4f}, Sens={row['sensibilidad']:.4f}, Prec={row['precision']:.4f}")
    
    return mejor_umbral

def diagnosticar_modelo(model, val_csv, images_dir, device, logger=None):
    """Verifica que el modelo produce salidas razonables"""
    model.eval()
    val_dataset = MRIDataset(val_csv, images_dir)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            if logger:
                logger.log(f"🔍 DIAGNÓSTICO DEL MODELO:")
                logger.log(f"   Logits - min: {logits.min():.4f}, max: {logits.max():.4f}")
                logger.log(f"   Probs  - min: {probs.min():.4f}, max: {probs.max():.4f}")
                logger.log(f"   Probs  - mean: {probs.mean():.4f}")
                logger.log(f"   Probs  - std:  {probs.std():.4f}")
                
                # Mostrar histograma simple
                bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                hist = np.histogram(probs.cpu().numpy(), bins=bins)[0]
                logger.log(f"   Histograma de predicciones:")
                for i in range(len(bins)-1):
                    logger.log(f"      {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:,} píxeles")
            break