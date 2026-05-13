import cv2
import numpy as np
import albumentations as A

# Definimos transformaciones que se aplicarán de forma idéntica a imagen y máscara
transformacion_aumento = A.Compose([
    A.RandomResizedCrop(size=(128,128), p=1.0),  # ← NUEVO
    A.HorizontalFlip(p=0.5), # Volteo horizontal (50% de probabilidad)
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5), # Rotación de 90 grados (50% de probabilidad)
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.25),
    A.Emboss(p=0.25),
    A.Blur(p=0.01, blur_limit = 3),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
    ], p=0.8),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25), # Pequeños giros y zoom
    A.Normalize(p=1.0),
])

def procesar_imagen_completo(fila_maestro, entrenando=False):
    if fila_maestro is None:
        raise ValueError("fila_maestro es None")
    
    if 'ruta_imagen' not in fila_maestro:
        raise KeyError(f"fila_maestro no tiene 'ruta_imagen': {fila_maestro}")
    # Cargamos la imagen original (3 canales: Pre, FLAIR, Post)
    img = cv2.imread(fila_maestro['ruta_imagen'], cv2.IMREAD_UNCHANGED).astype(np.float32)
    # Cargamos la máscara en escala de grises
    if fila_maestro['ruta_mascara'] is not None:
        mask = cv2.imread(fila_maestro['ruta_mascara'], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        # Crear una máscara vacía del mismo tamaño que la imagen
        mask = np.zeros(img.shape[:2], dtype=np.float32)
    
    # RECORTE AUTOMÁTICO
    # Buscamos todos los píxeles que no sean negros (valor > 0) en cualquier canal
    puntos_cerebro = np.argwhere(img.max(axis=2) > 0) #Busca en la matriz ancho x alto x 3 canales los pixeles que no son 0 en los tres canales
    
    if puntos_cerebro.size > 0:
        # Encontramos los límites: mínimo y máximo en ejes Y (filas) y X (columnas)
        y_min, x_min = puntos_cerebro.min(axis=0) #Buscamos la esquina superior izquierda que no tenga 0 
        y_max, x_max = puntos_cerebro.max(axis=0) #Buscamos la esquina inferior derecha  que no tenga 0
        
        # Recortamos tanto la imagen como la máscara usando esos límites
        img = img[y_min:y_max+1, x_min:x_max+1] #Recortamos el cuadrado de la imagen (le sumamos 1 porque empieza por 0)
        mask = mask[y_min:y_max+1, x_min:x_max+1] #Igual en la máscara

    # AUMENTO DE DATOS (DATA AUGMENTATION)
    if entrenando:
        # Convertimos la máscara a binaria (0 o 1) antes de transformar
        mask = (mask > 0).astype(np.float32)
        # Aplicamos las transformaciones aleatorias a ambos archivos a la vez
        resultado = transformacion_aumento(image=img, mask=mask)
        img, mask = resultado['image'], resultado['mask']

    # REDIMENSIONADO FINAL
    # Como el recorte cambia el tamaño, redimensionamos a un estándar (ej. 256x256) para el modelo
    img = cv2.resize(img, (128, 128)) #La interpolation por defecto INTER_LINEAR que nos mantiene una imágen suave
    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST) #INTER_NEAREST nos evita que la mascara se difumine

    # NORMALIZACIÓN Z-SCORE POR CANAL 
    #Normalizamos para que luego los podamos tratar a todos por igual y que la media y varianza no nos alteren
    #for i in range(3): # Iteramos por los 3 canales (0, 1 y 2)
    #    canal = img[:, :, i]
    #    pixeles_validos = canal[canal > 0] # Seleccionamos solo píxeles del cerebro
    #    
    #    if pixeles_validos.size > 0:
    #        media = pixeles_validos.mean() # Calculamos el promedio del canal
    #        std = pixeles_validos.std()   # Calculamos la desviación estándar
    #        # Aplicamos la fórmula: (valor - media) / desviación
    #        img[:, :, i] = (canal - media) / (std + 1e-8)
    #        # Nos aseguramos que el fondo recortado siga siendo 0 absoluto
    #        img[canal == 0, i] = 0


    return img, mask