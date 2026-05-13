import torch
import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, jaccard_score, precision_recall_curve

def segmentar_imagen(model, imagen_npy_path, device='cpu', umbral=0.5): #Recibe modelo, ruta de imágen, cpu o cuda y un umbral
    """
    Segmenta una imagen y devuelve máscara de probabilidad y binaria
    """
    img = np.load(imagen_npy_path)  # (Alto, Ancho, 3)
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) #(1,3,Alto,Ancho) #Lote de 1 imágen con tres canales
    
    model.eval() #Iniciamos modelo evaluación
    with torch.no_grad(): #No vamos a entrenar y por tanto no queremos gradientes
        logits = model(img_tensor).squeeze() #Aplica el modelo a la imágen, con squeeze le quitamos las dimensiones de 1 y nos quedamos con una matriz
        mascara_prob = torch.sigmoid(logits).cpu().numpy() #La llevamos a la cpu si es que estaba en la gpu y por ultimo la pasamos de tensor a matriz
    
    mascara_binaria = (mascara_prob > umbral).astype(np.uint8) #Generamos la máscara binaria
    return mascara_prob, mascara_binaria

def segmentar_lote(model, lista_imagenes, output_dir, device='cpu', umbral=0.5):
    """
    Segmenta un lote de imágenes y guarda las máscaras
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    resultados = []
    for img_path in lista_imagenes:
        img_path = Path(img_path)
        prob, binaria = segmentar_imagen(model, img_path, device, umbral)
        
        nombre_mascara = output_dir / f"{img_path.stem}_mask.npy"
        np.save(nombre_mascara, binaria)
        
        resultados.append({
            'imagen': str(img_path),
            'mascara': str(nombre_mascara),
            'tiene_tumor': int(binaria.sum() > 0)
        })
    
    return resultados