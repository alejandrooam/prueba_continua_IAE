# CREAMOS LA COMUNICACION ENTRE LA APLICACION Y EL DAGSTER

import requests
import json
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Dict, Any, Optional
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import joblib
from datetime import datetime
import cv2
import torch
import sys
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

src_path = Path(__file__).parent.parent  # Sube de Aplicacion a src
sys.path.insert(0, str(src_path))

# Ahora importar desde src
from transformar_datos import procesar_imagen_completo
from procesamiento_datos.modelo.segmentar import segmentar_imagen
from procesamiento_datos.modelo.modelo_unet import DobleConv, UNet


class DagsterClient:
    """Cliente para comunicarse con la API de Dagster"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
    
    def inicializar_sesion(self, session_id: str) -> str:
        """Inicializa una nueva sesión y crea el directorio temporal"""
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        return str(temp_dir)
    
    def guardar_datos_clinicos(self, datos_clinicos: Dict[str, Any], session_id: str) -> str:
        """
        Guarda los datos clínicos (aunque tengan valores nulos) y aplica analisis_datos()
        Devuelve la ruta del directorio temporal
        """
        # Crear directorio temporal para esta sesión si no existe
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar datos clínicos en CSV (con valores None permitidos)
        df_clinicos = pd.DataFrame([datos_clinicos])
        ruta_clinicos = temp_dir / "datos_clinicos.csv"
        df_clinicos.to_csv(ruta_clinicos, index=False, na_rep='NULL')
        
        return str(temp_dir)
    
    def procesar_imagen(self, imagen_tif_bytes: bytes, datos_clinicos: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Ejecuta el activo 'procesar_imagen' de Dagster
        Devuelve máscara y características de la imagen
        """
        # 1. Guardar la imagen temporalmente
        temp_dir = Path(tempfile.gettempdir()) / "mrai_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_img_path = temp_dir / f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tif"
        
        with open(temp_img_path, 'wb') as f:
            f.write(imagen_tif_bytes)
        
        # 2. Procesar imagen con tu función (recorte, normalización, etc.)
        fila_maestro = {
            'ruta_imagen': str(temp_img_path),
            'ruta_mascara': None
        }
        img_procesada, _ = procesar_imagen_completo(fila_maestro, entrenando=False)
        # img_procesada es (H, W, 3) con canales: [FLAIR, pre, post]
        
        # 3. Guardar imagen procesada temporalmente para el modelo
        temp_npy_path = temp_dir / f"img_procesada_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
        np.save(temp_npy_path, img_procesada)
        
        # 4. Cargar modelo de segmentación
        DATOS_PROCESADOS = src_path.parent / "datos_procesados"
        model = UNet(entrada=3, salida=1)
        modelo_unet_entrenado = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
        model.load_state_dict(torch.load(modelo_unet_entrenado, map_location='cpu'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Busco si tengo gpu para trabajar ahí
        model.to(device) #Le digo al modelo dónde trabajamos

        umbral = np.load(DATOS_PROCESADOS / "umbral.npy")
        
        # 5. Segmentar la imagen (obtener máscara)
        mascara_prob, mascara_binaria = segmentar_imagen(
            model=model,
            imagen_npy_path=str(temp_npy_path),
            device=device,
            umbral=umbral
        )
        
        # 6. Extraer características de la máscara
        caracteristicas = self.extraer_caracteristicas_completas(img_procesada, mascara_binaria)

        mascara_binaria = cv2.resize(mascara_binaria.astype(np.uint8), (256,256),interpolation=cv2.INTER_NEAREST)
        
        # 7. Crear DataFrame con las características
        df_caracteristicas = pd.DataFrame([caracteristicas])
        
        # 8. Limpiar archivos temporales
        temp_img_path.unlink()
        temp_npy_path.unlink()
        
        return mascara_binaria, df_caracteristicas
    
    def extraer_caracteristicas_completas(self, imagen: np.ndarray, mascara_binaria: np.ndarray) -> Dict[str, Any]:
        """
        imagen: array (H, W, 3) - canales: [FLAIR, pre, post]
        mascara_binaria: array (H, W) - 0 fondo, 1 tumor
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
        
        # 1. MORFOMÉTRICAS
        labeled = label(mascara_binaria)
        props = regionprops(labeled)[0]  # Tumor más grande
        
        area = props.area
        perimetro = props.perimeter
        circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        # 2. INTENSIDAD POST-CONTRASTE (canal 2)
        canal_post = imagen[:, :, 2]
        valores_post = canal_post[mascara_binaria > 0]
        intensidad_media_post = float(valores_post.mean())
        intensidad_minima_post = float(valores_post.min())
        
        # 3. PERCENTIL 95 FLAIR (canal 0)
        canal_flair = imagen[:, :, 0]
        valores_flair = canal_flair[mascara_binaria > 0]
        percentil_95_flair = float(np.percentile(valores_flair, 95))
        
        # 4. TEXTURA (sobre canal post)
        minr, minc, maxr, maxc = props.bbox
        rdi = canal_post[minr:maxr, minc:maxc]
        rdi_mask = mascara_binaria[minr:maxr, minc:maxc]
        rdi = rdi * rdi_mask
        
        if rdi.sum() > 0 and rdi.max() > 0:
            rdi = (rdi / rdi.max() * 255).astype(np.uint8)
        
        if rdi.sum() > 0 and rdi.shape[0] > 1 and rdi.shape[1] > 1:
            try:
                glcm = graycomatrix(rdi, distances=[1], angles=[0], levels=256, symmetric=True)
                textura_contraste = float(graycoprops(glcm, 'contrast')[0, 0])
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

    
    def calcular_urgencia(self, caracteristicas_df: pd.DataFrame, datos_clinicos: Dict[str, Any]) -> float:
        """
        Combina características del tumor con datos clínicos para predecir urgencia
        Usa el modelo guardado en datos_procesados/modelo_urgencia/modelo_urgencia.pkl
        """

        if 'area' in caracteristicas_df.columns:
            area = caracteristicas_df['area'].iloc[0]
        if area == 0 or pd.isna(area):
            return 0.0
        
        # 1. Cargar el modelo de urgencia
        DATOS_PROCESADOS = src_path.parent / "datos_procesados"
        modelo_path =  DATOS_PROCESADOS / "modelo_urgencia" / "modelo_urgencia.pkl"
        
        if not modelo_path.exists():
            raise FileNotFoundError(f"Modelo de urgencia no encontrado en {modelo_path}")
        
        data = joblib.load(modelo_path)
        modelo = data['modelo']  # El modelo está dentro de la clave 'modelo'
        
        # Crear DataFrame con las 9 variables en el orden correcto
        # Usar un diccionario con todas las variables
        features_dict = {}
        
        # Variables de características del tumor
        features_dict['area'] = caracteristicas_df['area'].iloc[0]
        features_dict['perimetro'] = caracteristicas_df['perimetro'].iloc[0]
        features_dict['circularidad'] = caracteristicas_df['circularidad'].iloc[0]
        features_dict['intensidad_media_post'] = caracteristicas_df['intensidad_media_post'].iloc[0]
        features_dict['intensidad_minima_post'] = caracteristicas_df['intensidad_minima_post'].iloc[0]
        features_dict['percentil_95_flair'] = caracteristicas_df['percentil_95_flair'].iloc[0]
        features_dict['textura_contraste'] = caracteristicas_df['textura_contraste'].iloc[0]
        
        # Variables de datos clínicos
        features_dict['age_at_initial_pathologic'] = datos_clinicos.get('age_at_initial_pathologic', 55) or 55
        
        # Convertir grado histológico a valor numérico
        grado = datos_clinicos.get('neoplasm_histologic_grade')
        if grado == "Grado IV" or grado == "IV":
            features_dict['neoplasm_histologic_grade'] = 4
        elif grado == "Grado III" or grado == "III":
            features_dict['neoplasm_histologic_grade'] = 3
        elif grado == "Grado II" or grado == "II":
            features_dict['neoplasm_histologic_grade'] = 2
        elif grado == "Grado I" or grado == "I":
            features_dict['neoplasm_histologic_grade'] = 1
        else:
            features_dict['neoplasm_histologic_grade'] = 2  # Valor por defecto
        
        # Crear DataFrame con el orden correcto de columnas
        columnas_ordenadas = ['area', 'perimetro', 'circularidad', 'intensidad_media_post', 
                            'intensidad_minima_post', 'percentil_95_flair', 'textura_contraste', 
                            'age_at_initial_pathologic', 'neoplasm_histologic_grade']
        
        features_completas = pd.DataFrame([features_dict])[columnas_ordenadas]
        
        # 4. Hacer predicción
        urgencia = modelo.predict_proba(features_completas)[0, 1]  # Probabilidad de clase positiva
        
        return float(urgencia)
    
    def guardar_resultados_imagen(self, session_id: str, mascara: np.ndarray, 
                                caracteristicas_df: pd.DataFrame, 
                                urgencia: float) -> str:
        """
        Guarda los resultados del procesamiento de imagen
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar máscara
        np.save(temp_dir / "mascara.npy", mascara)
        
        # Guardar características
        caracteristicas_df.to_csv(temp_dir / "caracteristicas_tumor.csv", index=False)
        
        # Guardar urgencia
        with open(temp_dir / "nivel_urgencia.txt", "w", encoding="utf-8") as f:
            f.write(f"Nivel de urgencia: {urgencia:.3f}\n")
            f.write(f"Fecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Interpretación clínica:\n")
            if urgencia < 0.3:
                f.write("- BAJA URGENCIA: Seguimiento programado en consultas externas\n")
                f.write("- Recomendación: Revisión en 3-6 meses\n")
            elif urgencia < 0.7:
                f.write("- URGENCIA MODERADA: Priorizar atención en consulta\n")
                f.write("- Recomendación: Evaluación en menos de 2 semanas\n")
            else:
                f.write("- ALTA URGENCIA: Requiere intervención inmediata\n")
                f.write("- Recomendación: Derivación a urgencias neuroquirúrgicas\n")
        
        return str(temp_dir)
    
    def guardar_todos_los_datos(self, session_id: str, mascara: np.ndarray, 
                            caracteristicas_df: pd.DataFrame, 
                            urgencia: float,
                            datos_clinicos: Dict) -> str:
        """
        Guarda todos los datos generados en una carpeta temporal
        Crea un CSV completo combinando características y datos clínicos
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # 1. Guardar máscara
        np.save(temp_dir / "mascara.npy", mascara)
        
        # 2. Guardar características del tumor
        caracteristicas_df.to_csv(temp_dir / "caracteristicas_tumor.csv", index=False)
        
        # 3. Crear CSV completo (características + datos clínicos)
        df_completo = caracteristicas_df.copy()
        
        # Añadir datos clínicos
        if datos_clinicos:
            for key, value in datos_clinicos.items():
                if value is not None:
                    df_completo[key] = value
                else:
                    df_completo[key] = 'NULL'
        
        # Añadir nivel de urgencia
        df_completo['nivel_urgencia'] = urgencia
        
        # Guardar DataFrame completo
        df_completo.to_csv(temp_dir / "datos_completos.csv", index=False, na_rep='NULL')
        
        # 4. Guardar nivel de urgencia en texto legible
        with open(temp_dir / "nivel_urgencia.txt", "w", encoding="utf-8") as f:
            f.write("=== NIVEL DE URGENCIA ===\n")
            f.write(f"Puntuación: {urgencia:.3f} / 1.000\n\n")
            if urgencia < 0.3:
                f.write("Clasificación: 🟢 URGENCIA BAJA\n")
                f.write("Recomendación: Seguimiento programado en consultas externas\n")
                f.write("Plazo sugerido: 3-6 meses\n")
            elif urgencia < 0.7:
                f.write("Clasificación: 🟡 URGENCIA MODERADA\n")
                f.write("Recomendación: Priorizar atención en consulta especializada\n")
                f.write("Plazo sugerido: Menos de 2 semanas\n")
            else:
                f.write("Clasificación: 🔴 ALTA URGENCIA\n")
                f.write("Recomendación: Intervención neuroquirúrgica inmediata\n")
                f.write("Plazo sugerido: 24-48 horas\n")
        
        # 5. Guardar datos clínicos originales
        df_clinicos = pd.DataFrame([datos_clinicos])
        df_clinicos.to_csv(temp_dir / "datos_clinicos.csv", index=False, na_rep='NULL')
        
        # 6. Crear README
        with open(temp_dir / "README.txt", "w", encoding="utf-8") as f:
            f.write("=== INFORME COMPLETO MRAI ===\n\n")
            f.write(f"ID de sesión: {session_id}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("ARCHIVOS CONTENIDOS:\n")
            f.write("1. mascara.npy - Máscara de segmentación del tumor\n")
            f.write("2. caracteristicas_tumor.csv - Características radiomicas extraídas\n")
            f.write("3. datos_completos.csv - Características + datos clínicos + urgencia\n")
            f.write("4. nivel_urgencia.txt - Nivel de urgencia clínica (0-1) interpretado\n")
            f.write("5. datos_clinicos.csv - Datos demográficos y clínicos del paciente\n")
        
        return str(temp_dir)
    
    def limpiar_datos_sesion(self, session_id: str):
        """Elimina todos los datos temporales de una sesión"""
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# Instancia global del cliente
@st.cache_resource
def get_dagster_client():
    return DagsterClient(base_url="http://localhost:3000")