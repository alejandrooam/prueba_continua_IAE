import numpy as np
import pandas as pd
from skimage.measure import regionprops, label #Regionprops es para area, perímetro y tal. Label es para etiquetar tumores por si hay varios
from skimage.feature import graycomatrix, graycoprops #Para las texturas del tumor
from pathlib import Path

def extraer_caracteristicas(imagen, mascara_binaria):
    """
    imagen: array (H, W, 3) - canales: [FLAIR, pre, post]
    mascara_binaria: array (H, W) - 0 fondo, 1 tumor
    
    Retorna diccionario con 8 características:
    - area (píxeles)
    - perimetro (píxeles)
    - circularidad
    - intensidad_media_post
    - intensidad_minima_post
    - percentil_95_flair
    - textura_contraste
    - textura_homogeneidad
    """
    
    # Si no hay tumor, devolver todo ceros
    if mascara_binaria.sum() == 0:
        return {
            'area': 0,
            'perimetro': 0,
            'circularidad': 0,
            'intensidad_media_post': 0,
            'intensidad_minima_post': 0,
            'percentil_95_flair': 0,
            'textura_contraste': 0,
        }
    

    # 1. MORFOMÉTRICAS (area, perimetro, circularidad)
  
    labeled = label(mascara_binaria) #Con esto separamos distintos tumores. Entendemos que hay un tumor grande y alomejor unos pequeños que se han desprendido de este
    props = regionprops(labeled)[0] #Le calcula muchas cosas al tumor más grande (el primero)
    
    area = props.area #Un tumor pequeño tiene mejor pronóstico que uno grande (claramente)
    perimetro = props.perimeter
    circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0 #Super importante. Cuanto más redondo mejor. Si es redondo es benigno, si es estrellado es muy invasivo
    #Circularidad baja: el tumor invade el tejido sano. Circularidad alta: el tumor empuja el tejido sano
    

    # 2. INTENSIDAD POST-CONTRASTE (canal 2, índice 2)

    canal_post = imagen[:, :, 2]  # tercer canal = post-contraste
    valores_post = canal_post[mascara_binaria > 0] #Elegimos los pixeles de la imágen donde la máscara nos dice que hay tumor
    

    #El post es meter una sustancia que solo entra por donde se rompe la barrera hematoencefálica
    intensidad_media_post = float(valores_post.mean()) #Me dice si el tumor es de bajo grado (bajo) o maligno (alto)
    intensidad_minima_post = float(valores_post.min()) #Si la minima es baja, hay necrosis en el tumor porque el contraste no puede llegar por vasos sanguíneos y eso es que el tumor crece tan rápido que ni siquiera llegan nutrientes al centro (Muy malo)
    

    # 3. PERCENTIL 95 FLAIR (canal 0)

    canal_flair = imagen[:, :, 0]  # primer canal = FLAIR
    valores_flair = canal_flair[mascara_binaria > 0]
    percentil_95_flair = float(np.percentile(valores_flair, 95)) 
    #El FLAIR mide el edema. El edema es la reacción del cerebro entorno al tumor.
    #Elegimos la zona máxima de edema (inflamación)
    #Si hay poco es porque el tumor está bien delimitado (bueno)
    #Si hay mucho es porque hay infiltración tumoral. El tumor está mandando células como si fuesen esporas y cada una tiene edema a su alrededor (Malísimo)
    

    # 4. TEXTURA (sobre canal post, que es el más informativo)

    #La textura es cómo se organizan los píxeles, homogéneos o heterogéneos
    #Si la textura es baja el tumor es homogéneo, todas las celulas están vivas y no hay nada raro en el tumor
    #Si la textura es alta es porque hay sitios con mucho post y sitios con poco. Es un tumor heterogéneo.
    #Esto se debe a que hay calcificaciones o necrosis (no llega el post), celulas vivas (post normal) y microhemorragias (muchisimo post)


    # Recortar bounding box para eficiencia
    minr, minc, maxr, maxc = props.bbox #Me da los valores de la caja que envuelve al tumor
    rdi = canal_post[minr:maxr, minc:maxc] #Es la región de interés de la imágen con el post
    rdi_mask = mascara_binaria[minr:maxr, minc:maxc] 
    
    # Aplicar máscara y normalizar a 0-255 para GLCM (Gray Level Co-occurrence Matrix (Matriz de Co-ocurrencia de Niveles de Gris))
    rdi = rdi * rdi_mask #Nos quedamos con la imágen donde hay tumor
    rdi = (rdi / rdi.max() * 255).astype(np.uint8) if rdi.max() > 0 else rdi.astype(np.uint8) 
    
    # Calcular GLCM solo si hay suficientes píxeles
    if rdi.sum() > 0 and rdi.shape[0] > 1 and rdi.shape[1] > 1: #Si tiene al menos un pixel, mas de una fila y mas de una columna
        try:
            glcm = graycomatrix(rdi, distances=[1], angles=[0], levels=256, symmetric=True) #Es una matriz que cuenta con qué frecuencia aparecen pares de píxeles con ciertos valores.
            #distances=1 es porque compara los pixeles con los que están a distancia 1. angles=0 porque mira solo a los que están a 0 grados (derecha). levels=256 porque hay 256 tonos de gris. Simetric es porque no importa si vas de izq a dcha o de dcha a izq
            textura_contraste = float(graycoprops(glcm, 'contrast')[0, 0]) #Esto es porque le podemos meter muchas distancias y angulos, asiq nos quedamos con el primero
        except:
            textura_contraste = 0
    else:
        textura_contraste = 0
    
    return {
        'area': area,
        'perimetro': perimetro,
        'circularidad': circularidad,
        'intensidad_media_post': intensidad_media_post,
        'intensidad_minima_post': intensidad_minima_post,
        'percentil_95_flair': percentil_95_flair,
        'textura_contraste': textura_contraste,
    }


def generar_dataset_features(df_segmentacion, directorio_imagenes, directorio_mascaras, df_previa=None):
    """
    Genera dataset tabular con las 7 características
    
    Parámetros:
    - df_segmentacion: DataFrame con columnas 'paciente', 'num_corte', 'ruta_procesada'
    - directorio_imagenes: Path donde están las imágenes .npy
    - directorio_mascaras: Path donde están las máscaras predichas
    - df_previa: DataFrame con columna 'paciente' y otras columnas (opcional)
    """
    directorio_imagenes = Path(directorio_imagenes)
    directorio_mascaras = Path(directorio_mascaras)
    
    registros = []
    
    for _, row in df_segmentacion.iterrows():
        # Cargar imagen original
        img_path = directorio_imagenes / Path(row['ruta_procesada']).name
        if not img_path.exists():
            print(f"Imagen no encontrada: {img_path}")
            continue
        
        # Cargar máscara predicha
        mask_path = directorio_mascaras / f"{row['paciente']}_{row['num_corte']}_mask.npy"
        if not mask_path.exists():
            print(f"Máscara no encontrada: {mask_path}")
            continue
        
        img = np.load(img_path)
        mascara = np.load(mask_path)
        
        # Extraer características
        features = extraer_caracteristicas(img, mascara)
        
        # Añadir metadatos
        features['paciente'] = row['paciente']
        features['num_corte'] = row['num_corte']
        
        # Añadir información previa si existe
        if df_previa is not None:
            previa_paciente = df_previa[df_previa['id_paciente'] == row['paciente']]
            if len(previa_paciente) > 0:
                for col in df_previa.columns:
                    if col != 'paciente':
                        features[f'{col}'] = previa_paciente.iloc[0][col]
            else:
                # Si no hay previa para este paciente, poner NaN
                for col in df_previa.columns:
                    if col != 'paciente':
                        features[f'{col}'] = np.nan
        
        registros.append(features)
    
    return pd.DataFrame(registros)