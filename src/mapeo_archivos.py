import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

ruta_base = r"Trabajo\DATOS"
ruta_csv = os.path.join(ruta_base, "data.csv")


#CARGAMOS LOS DATOS Y VEMOS QUE ESTÁ BIEN

df_clinico = pd.read_csv(ruta_csv) #Leemos el archivo en data.frame
print(f"   Pacientes en data.csv: {len(df_clinico)}") #Vemos que hay 110 pacientes
print(f"   Columnas disponibles: {df_clinico.columns.tolist()}") #Vemos qué columnas hay



def detectar_secuencias_disponibles(ruta_imagen, umbral_similitud=0.95):
#Se utiliza para detectar si una foto carece de PRE o de POST
#Recibe la ruta de la foto y un umbral
#Todas las fotos tienen FLAIR
#Nos devuelve una tupla de booleanos  
    try:
        img = np.array(Image.open(ruta_imagen)).astype(np.float32)
        #Convierte la imagen en una matriz para tratarla
        
        #Las imágenes constan de tres canales
        #El canal central (índice 1) siempre es FLAIR
        canal_flair = img[:,:,1]
        
        #Si la imagen es constante (todo cero), algo va mal
        if np.all(canal_flair == 0):
            return True, True 
        #Si todo es 0 la imagen está en negro y puede ser que esté corrupta, pero
        #también puede ser que simplemente sea una imagen negra, por lo que suponemos
        #que todo está bien para no perder al paciente
        
        # Analizar canal Pre (índice 0)
        canal_pre = img[:,:,0]
        
        #Calcular correlación entre Pre y FLAIR
        #Si están altamente correlacionados, es que Pre no está disponible (relleno con FLAIR)
        if np.all(canal_pre == 0):
            pre_disponible = False
        else:
            #Normalizar para correlación
            pre_flat = canal_pre.flatten()
            flair_flat = canal_flair.flatten()
            
            #Evitar división por cero
            if np.std(pre_flat) > 0 and np.std(flair_flat) > 0:
                correlacion_pre = np.corrcoef(pre_flat, flair_flat)[0,1]
                pre_disponible = correlacion_pre < umbral_similitud
            else:
                pre_disponible = True
        
        #Analizar canal Post (índice 2)
        canal_post = img[:,:,2]
        
        if np.all(canal_post == 0):
            post_disponible = False
        else:
            post_flat = canal_post.flatten()
            
            if np.std(post_flat) > 0 and np.std(flair_flat) > 0:
                correlacion_post = np.corrcoef(post_flat, flair_flat)[0,1]
                post_disponible = correlacion_post < umbral_similitud
            else:
                post_disponible = True
        
        return pre_disponible, post_disponible
        
    except Exception as e:
        print(f"   Error detectando secuencias: {e}")
        return True, True  # Por defecto, asumir que tiene todo



def crear_indice_archivos(ruta_base, df_clinico, detectar=True):

    registros = []
    pacientes_procesados = 0
    estadisticas = {'pre_faltan': 0, 'post_faltan': 0, 'ambos_faltan': 0}
    ruta_base = os.path.abspath(ruta_base)
    
    for carpeta in os.listdir(ruta_base): #listdir nos da la lista de direcciones de DATOS
        ruta_carpeta = os.path.join(ruta_base, carpeta) #Une la dirección a DATOS con la de la carpeta
        
        if not os.path.isdir(ruta_carpeta) or not carpeta.startswith("TCGA_"): #Comprueba que sea una dirección y que sea de un paciente
            continue
        
        # Extraer ID real (primeras 3 partes)
        partes = carpeta.split('_') 
        id_paciente = '_'.join(partes[:3])
        fecha = partes[3]
        
        # Verificar si el paciente existe en data.csv
        if id_paciente not in df_clinico['Patient'].values:
            print(f"   Paciente {id_paciente} no encontrado en data.csv")
            continue
        
        pacientes_procesados += 1
        info_paciente = df_clinico[df_clinico['Patient'] == id_paciente]
        
        # Procesar archivos
        archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.tif')] #todas las imágenes
        imagenes = [f for f in archivos if not f.endswith('_mask.tif')] #imágenes (no máscaras)
        
        # Para el primer corte, detectar secuencias disponibles
        tiene_pre = True
        tiene_post = True
        
        if detectar and len(imagenes) > 0:
            primer_corte = os.path.join(ruta_carpeta, imagenes[0])
            tiene_pre, tiene_post = detectar_secuencias_disponibles(primer_corte)
            
            # Actualizar estadísticas
            if not tiene_pre:
                estadisticas['pre_faltan'] += 1
            if not tiene_post:
                estadisticas['post_faltan'] += 1
            if not tiene_pre and not tiene_post:
                estadisticas['ambos_faltan'] += 1
        
        for img in imagenes:
            mask = img.replace('.tif', '_mask.tif')
            ruta_img = os.path.join(ruta_carpeta, img)
            ruta_mask = os.path.join(ruta_carpeta, mask)
            
            
            # Extraer número de corte
            try:
                num_corte = int(img.replace('.tif', '').split('_')[-1])
            except:
                num_corte = -1
            
            # Leer máscara para estadísticas
            try:
                mask_array = np.array(Image.open(ruta_mask))
                tiene_tumor = np.any(mask_array > 0)
                tamaño_tumor = np.sum(mask_array > 0) if tiene_tumor else 0
            except:
                tiene_tumor = None
                tamaño_tumor = None
            
            registros.append({
                'id_paciente': id_paciente,
                'carpeta_original': carpeta,
                'fecha_muestra': fecha,
                'num_corte': num_corte,
                'ruta_imagen': ruta_img,
                'ruta_mascara': ruta_mask,
                'institucion': id_paciente.split('_')[1],
                'tiene_pre': tiene_pre,
                'tiene_post': tiene_post,
                'canales_disponibles': 1 + int(tiene_pre) + int(tiene_post),
                'mascara_tiene_tumor': tiene_tumor,
                'tamaño_tumor_pixeles': tamaño_tumor
            })
    
    df = pd.DataFrame(registros)
    
    print(f"\n  Pacientes procesados: {pacientes_procesados}") #Vemos que haya 110
    
    if detectar:
        print(f"\n   SECUENCIAS DETECTADAS:")
        print(f"       Pacientes sin Pre-contrast: {estadisticas['pre_faltan']}")
        print(f"       Pacientes sin Post-contrast: {estadisticas['post_faltan']}") #Debe haber 9
        print(f"       Pacientes sin ambos: {estadisticas['ambos_faltan']}") #Debe haber 6
        
    return df

# Crear índice
df_archivos = crear_indice_archivos(ruta_base, df_clinico, detectar=True)



df_catalogo = pd.merge(
    df_archivos,
    df_clinico,
    left_on='id_paciente',
    right_on='Patient',
    how='left'
)

print(f"    Registros en catálogo: {len(df_catalogo)}")
print(f"    Pacientes únicos: {df_catalogo['id_paciente'].nunique()}")


cortes_con_tumor = df_catalogo['mascara_tiene_tumor'].sum()
print(f"    Cortes CON tumor: {cortes_con_tumor} ({cortes_con_tumor/len(df_catalogo)*100:.1f}%)")
print(f"    Cortes SIN tumor: {len(df_catalogo)-cortes_con_tumor} ({(len(df_catalogo)-cortes_con_tumor)/len(df_catalogo)*100:.1f}%)")


ruta_catalogo = os.path.join(ruta_base, "catalogo_maestro_final.csv")
df_catalogo.to_csv(ruta_catalogo, index=False)
print(f"   Guardado en: {ruta_catalogo}")