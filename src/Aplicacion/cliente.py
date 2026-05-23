# APLICACION/CLIENTE
# CREAMOS LA COMUNICACION ENTRE LA APLICACION Y EL DAGSTER

# =============================================================================
# LIBRERIAS NECESARIAS
# =============================================================================

import requests          # Comunicacion HTTP con la API de Dagster
import json              # Manejo de datos en formato JSON
import pandas as pd      # Manipulacion de dataframes
import numpy as np       # Computacion numerica y manejo de arrays
from PIL import Image    # Procesamiento basico de imagenes
import io                # Operaciones de entrada/salida en memoria
import base64            # Codificacion/decodificacion Base64
from typing import Tuple, Dict, Any, Optional  # Tipado de datos
import streamlit as st   # Framework para interfaz web
from pathlib import Path # Manejo de rutas de archivos
import tempfile          # Creacion de archivos y directorios temporales
import shutil            # Operaciones avanzadas de archivos
import joblib            # Carga de modelos guardados (formato .pkl)
from datetime import datetime  # Generacion de timestamps
import cv2               # OpenCV: procesamiento avanzado de imagenes
import torch             # PyTorch: deep learning
import sys               # Manipulacion del sistema
from skimage.measure import label, regionprops  # Propiedades morfologicas
from skimage.feature import graycomatrix, graycoprops  # Matriz de co-ocurrencia

# =============================================================================
# CONFIGURACION DE RUTAS
# =============================================================================

# Obtiene el directorio padre del directorio actual (sube de Aplicacion a src)
src_path = Path(__file__).parent.parent
# Inserta src_path al inicio de sys.path para que Python busque modulos alli
sys.path.insert(0, str(src_path))

RAIZ_PROYECTO = Path(__file__).parent.parent.parent
DATOS_PROCESADOS = RAIZ_PROYECTO / "datos_procesados"

# =============================================================================
# IMPORTACION DE MODULOS PROPIOS DEL PROYECTO
# =============================================================================

from transformar_datos import procesar_imagen_completo  # Preprocesamiento de resonancias
from src.procesamiento_datos.modelo.segmentar import segmentar_imagen  # Inferencia del modelo
from src.procesamiento_datos.modelo.modelo_fpn import DobleConv, FPN  # Arquitectura FPN


# =============================================================================
# CLASE PRINCIPAL: DAGSTER CLIENT
# =============================================================================

class DagsterClient:
    """
    Cliente para comunicarse con la API de Dagster.
    Actua como intermediario entre la aplicacion web (Streamlit) y el orquestador Dagster,
    gestionando los datos temporales de cada sesion de paciente.
    """
    
    def __init__(self):
        """
        Inicializa el cliente con la URL base del orquestador Dagster.
        
        Args:
            base_url: URL donde corre el servidor de Dagster (puerto 3000 por defecto)
        """
        self.model = None
        self.device = None
        self.umbral = None
        self.modelo_urgencia = None

        # Cargar modelos al iniciar
        self._cargar_modelos()
    

    def _cargar_modelos(self):
        """Carga los modelos entrenados desde datos_procesados/"""
        try:
            # Ruta a la raíz del proyecto
            PROJECT_ROOT = Path(__file__).parent.parent.parent
            DATOS_PROCESADOS = PROJECT_ROOT / "datos_procesados"
            
            # Cargar modelo FPN
            modelo_fpn_path = DATOS_PROCESADOS / "modelo_fpn_mejor.pth"
            if modelo_fpn_path.exists():
                self.model = FPN()
                self.model.load_state_dict(torch.load(modelo_fpn_path, map_location='cpu'))
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.model.eval()
            else:
                st.error(f"Modelo FPN no encontrado en {modelo_fpn_path}")
            
            # Cargar umbral
            umbral_path = DATOS_PROCESADOS / "umbral.npy"
            if umbral_path.exists():
                self.umbral = np.load(umbral_path)
            else:
                self.umbral = 0.5
            
            # Cargar modelo de urgencia
            modelo_urgencia_path = DATOS_PROCESADOS / "modelo_urgencia" / "modelo_urgencia.pkl"
            if modelo_urgencia_path.exists():
                data = joblib.load(modelo_urgencia_path)
                self.modelo_urgencia = data['modelo']
            else:
                st.warning(f"Modelo urgencia no encontrado en {modelo_urgencia_path}")
                
        except Exception as e:
            st.error(f"Error al cargar modelos: {str(e)}")


    def inicializar_sesion(self, session_id: str) -> str:
        """
        Inicializa una nueva sesion y crea el directorio temporal.
        
        Args:
            session_id: Identificador unico de la sesion
        
        Returns:
            Ruta del directorio temporal creado
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        return str(temp_dir)
    
    def guardar_datos_clinicos(self, datos_clinicos: Dict[str, Any], session_id: str) -> str:
        """
        Guarda los datos clinicos del paciente en un CSV temporal.
        
        Args:
            datos_clinicos: Diccionario con los datos del paciente
            session_id: Identificador de la sesion
        
        Returns:
            Ruta del directorio temporal donde se guardo el CSV
        """
        # Crear directorio temporal para esta sesion si no existe
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar datos clinicos en CSV (con valores None permitidos)
        df_clinicos = pd.DataFrame([datos_clinicos])
        ruta_clinicos = temp_dir / "datos_clinicos.csv"
        df_clinicos.to_csv(ruta_clinicos, index=False, na_rep='NULL')
        
        return str(temp_dir)
    
    def procesar_imagen(self, imagen_tif_bytes: bytes, datos_clinicos: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de procesamiento de imagen:
        1. Preprocesado (recorte y normalizacion)
        2. Segmentacion del tumor con FPN
        3. Extraccion de biomarcadores
        """
        # PASO 1: Guardar la imagen temporalmente
        temp_dir = Path(tempfile.gettempdir()) / "mrai_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_img_path = temp_dir / f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tif"
        
        with open(temp_img_path, 'wb') as f:
            f.write(imagen_tif_bytes)
        
        # PASO 2: Procesar imagen (recorte, normalizacion, etc.)
        fila_maestro = {
            'ruta_imagen': str(temp_img_path),
            'ruta_mascara': None
        }
        img_procesada, _ = procesar_imagen_completo(fila_maestro, entrenando=False)
        
        # PASO 3: Guardar imagen procesada temporalmente para el modelo
        temp_npy_path = temp_dir / f"img_procesada_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
        np.save(temp_npy_path, img_procesada)
        
        # PASO 4: USAR EL MODELO YA CARGADO (no crear uno nuevo)
        # model = FPN()  ← ELIMINA ESTO
        # modelo_fpn_entrenado = DATOS_PROCESADOS / "modelo_fpn_mejor.pth"  ← ELIMINA ESTO
        # model.load_state_dict(...)  ← ELIMINA ESTO
        
        # PASO 5: Segmentar usando self.model (ya cargado en __init__)
        mascara_prob, mascara_binaria = segmentar_imagen(
            model=self.model,           # ← CAMBIA model=model por model=self.model
            imagen_npy_path=str(temp_npy_path),
            device=self.device,         # ← usa self.device
            umbral=self.umbral          # ← usa self.umbral
        )
        
        # PASO 6: Extraer caracteristicas radiomicas de la mascara
        caracteristicas = self.extraer_caracteristicas_completas(img_procesada, mascara_binaria)
        
        # PASO 7: Crear DataFrame con las caracteristicas
        df_caracteristicas = pd.DataFrame([caracteristicas])
        
        # PASO 8: Limpiar archivos temporales
        temp_img_path.unlink()
        temp_npy_path.unlink()
        
        return mascara_binaria, df_caracteristicas
    
    def extraer_caracteristicas_completas(self, imagen: np.ndarray, mascara_binaria: np.ndarray) -> Dict[str, Any]:
        """
        Extrae caracteristicas radiomicas del tumor:
        - Morfologicas: area, perimetro, circularidad
        - Intensidad: media y minima post-contraste, percentil 95 FLAIR
        - Textura: contraste GLCM (heterogeneidad)
        
        Args:
            imagen: Array (H, W, 3) con canales [FLAIR, pre, post]
            mascara_binaria: Array (H, W) con valores 0 (fondo) y 1 (tumor)
        
        Returns:
            Diccionario con todas las caracteristicas extraidas
        """
        
        # CASO ESPECIAL: Si no hay tumor, devolver todo ceros
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
        
        # SECCION 1: MORFOMETRICAS (forma y tamaño)
        labeled = label(mascara_binaria)
        props = regionprops(labeled)[0]  # Tomar la region mas grande (tumor principal)
        
        area = props.area  # Numero de pixeles que ocupa el tumor
        perimetro = props.perimeter  # Longitud del borde del tumor
        # Circularidad: 1 = circulo perfecto, <1 = irregular (tumores agresivos)
        circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        # SECCION 2: INTENSIDAD POST-CONTRASTE (canal 2)
        canal_post = imagen[:, :, 2]
        valores_post = canal_post[mascara_binaria > 0]
        intensidad_media_post = float(valores_post.mean())  # Vascularizacion
        intensidad_minima_post = float(valores_post.min())  # Necrosis
        
        # SECCION 3: PERCENTIL 95 FLAIR (canal 0) - edema peritumoral
        canal_flair = imagen[:, :, 0]
        valores_flair = canal_flair[mascara_binaria > 0]
        percentil_95_flair = float(np.percentile(valores_flair, 95))
        
        # SECCION 4: TEXTURA (heterogeneidad) sobre canal post-contraste
        minr, minc, maxr, maxc = props.bbox  # Obtener bounding box del tumor
        rdi = canal_post[minr:maxr, minc:maxc]  # Recortar region de interes
        rdi_mask = mascara_binaria[minr:maxr, minc:maxc]
        rdi = rdi * rdi_mask  # Aplicar mascara
        
        # Normalizar a 8 bits (0-255) si hay valores positivos
        if rdi.sum() > 0 and rdi.max() > 0:
            rdi = (rdi / rdi.max() * 255).astype(np.uint8)
        
        # Calcular matriz de co-ocurrencia (GLCM) y extraer contraste
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
        Predice el nivel de urgencia clinica combinando:
        - Caracteristicas radiomicas del tumor (7 variables)
        - Datos clinicos del paciente (edad y grado tumoral)
        """
        # Verificar si hay tumor (area > 0)
        if 'area' in caracteristicas_df.columns:
            area = caracteristicas_df['area'].iloc[0]
        if area == 0 or pd.isna(area):
            return 0.0  # Sin tumor = sin urgencia
        
        # PASO 1: USAR EL MODELO YA CARGADO (no recargar)
        # ELIMINA estas 3 líneas:
        # modelo_path = DATOS_PROCESADOS / "modelo_urgencia" / "modelo_urgencia.pkl"
        # data = joblib.load(modelo_path)
        # modelo = data['modelo']
        
        # En su lugar, usa self.modelo_urgencia directamente:
        if self.modelo_urgencia is None:
            return 0.5  # valor por defecto si no hay modelo
        
        # PASO 2: Preparar diccionario con todas las caracteristicas
        features_dict = {}
        
        features_dict['area'] = caracteristicas_df['area'].iloc[0]
        features_dict['perimetro'] = caracteristicas_df['perimetro'].iloc[0]
        features_dict['circularidad'] = caracteristicas_df['circularidad'].iloc[0]
        features_dict['intensidad_media_post'] = caracteristicas_df['intensidad_media_post'].iloc[0]
        features_dict['intensidad_minima_post'] = caracteristicas_df['intensidad_minima_post'].iloc[0]
        features_dict['percentil_95_flair'] = caracteristicas_df['percentil_95_flair'].iloc[0]
        features_dict['textura_contraste'] = caracteristicas_df['textura_contraste'].iloc[0]
        features_dict['age_at_initial_pathologic'] = datos_clinicos.get('age_at_initial_pathologic', 55) or 55
        
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
            features_dict['neoplasm_histologic_grade'] = 2
        
        columnas_ordenadas = [
            'area', 'perimetro', 'circularidad', 
            'intensidad_media_post', 'intensidad_minima_post', 
            'percentil_95_flair', 'textura_contraste', 
            'age_at_initial_pathologic', 'neoplasm_histologic_grade'
        ]
        
        features_completas = pd.DataFrame([features_dict])[columnas_ordenadas]
        
        # PASO 3: Predecir usando self.modelo_urgencia (no modelo)
        urgencia = self.modelo_urgencia.predict_proba(features_completas)[0, 1]
        
        return float(urgencia)
    
    def guardar_resultados_imagen(self, session_id: str, mascara: np.ndarray, 
                                caracteristicas_df: pd.DataFrame, 
                                urgencia: float) -> str:
        """
        Guarda los resultados del procesamiento de imagen en archivos.
        
        Args:
            session_id: Identificador de la sesion
            mascara: Mascara binaria del tumor
            caracteristicas_df: DataFrame con caracteristicas tumorales
            urgencia: Probabilidad de urgencia (0-1)
        
        Returns:
            Ruta del directorio con los archivos guardados
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar mascara en formato numpy
        np.save(temp_dir / "mascara.npy", mascara)
        
        # Guardar caracteristicas en CSV
        caracteristicas_df.to_csv(temp_dir / "caracteristicas_tumor.csv", index=False)
        
        # Guardar nivel de urgencia en archivo de texto
        with open(temp_dir / "nivel_urgencia.txt", "w", encoding="utf-8") as f:
            f.write(f"Nivel de urgencia: {urgencia:.3f}\n")
            f.write(f"Fecha del analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Interpretacion clinica:\n")
            if urgencia < 0.3:
                f.write("- BAJA URGENCIA: Seguimiento programado en consultas externas\n")
                f.write("- Recomendacion: Revision en 3-6 meses\n")
            elif urgencia < 0.7:
                f.write("- URGENCIA MODERADA: Priorizar atencion en consulta\n")
                f.write("- Recomendacion: Evaluacion en menos de 2 semanas\n")
            else:
                f.write("- ALTA URGENCIA: Requiere intervencion inmediata\n")
                f.write("- Recomendacion: Derivacion a urgencias neuroquirurgicas\n")
        
        return str(temp_dir)
    
    def guardar_todos_los_datos(self, session_id: str, mascara: np.ndarray, 
                            caracteristicas_df: pd.DataFrame, 
                            urgencia: float,
                            datos_clinicos: Dict) -> str:
        """
        Guarda todos los datos generados (version completa con datos clinicos).
        Crea un CSV unificado combinando caracteristicas y datos clinicos.
        
        Args:
            session_id: Identificador de la sesion
            mascara: Mascara binaria del tumor
            caracteristicas_df: DataFrame con caracteristicas tumorales
            urgencia: Probabilidad de urgencia
            datos_clinicos: Diccionario con datos clinicos del paciente
        
        Returns:
            Ruta del directorio con todos los archivos
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # 1. Guardar mascara
        np.save(temp_dir / "mascara.npy", mascara)
        
        # 2. Guardar caracteristicas del tumor
        caracteristicas_df.to_csv(temp_dir / "caracteristicas_tumor.csv", index=False)
        
        # 3. Crear CSV completo (caracteristicas + datos clinicos + urgencia)
        df_completo = caracteristicas_df.copy()
        
        # Añadir datos clinicos como columnas adicionales
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
            f.write(f"Puntuacion: {urgencia:.3f} / 1.000\n\n")
            if urgencia < 0.3:
                f.write("Clasificacion: [BAJA] URGENCIA BAJA\n")
                f.write("Recomendacion: Seguimiento programado en consultas externas\n")
                f.write("Plazo sugerido: 3-6 meses\n")
            elif urgencia < 0.7:
                f.write("Clasificacion: [MODERADA] URGENCIA MODERADA\n")
                f.write("Recomendacion: Priorizar atencion en consulta especializada\n")
                f.write("Plazo sugerido: Menos de 2 semanas\n")
            else:
                f.write("Clasificacion: [ALTA] ALTA URGENCIA\n")
                f.write("Recomendacion: Intervencion neuroquirurgica inmediata\n")
                f.write("Plazo sugerido: 24-48 horas\n")
        
        # 5. Guardar datos clinicos originales
        df_clinicos = pd.DataFrame([datos_clinicos])
        df_clinicos.to_csv(temp_dir / "datos_clinicos.csv", index=False, na_rep='NULL')
        
        # 6. Crear archivo README con descripcion de los archivos
        with open(temp_dir / "README.txt", "w", encoding="utf-8") as f:
            f.write("=== INFORME COMPLETO MRAI ===\n\n")
            f.write(f"ID de sesion: {session_id}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("ARCHIVOS CONTENIDOS:\n")
            f.write("1. mascara.npy - Mascara de segmentacion del tumor\n")
            f.write("2. caracteristicas_tumor.csv - Caracteristicas radiomicas extraidas\n")
            f.write("3. datos_completos.csv - Caracteristicas + datos clinicos + urgencia\n")
            f.write("4. nivel_urgencia.txt - Nivel de urgencia clinica interpretado\n")
            f.write("5. datos_clinicos.csv - Datos demograficos y clinicos del paciente\n")
        
        return str(temp_dir)
    
    def limpiar_datos_sesion(self, session_id: str):
        """
        Elimina todos los datos temporales de una sesion.
        Importante para garantizar la privacidad del paciente.
        
        Args:
            session_id: Identificador de la sesion a eliminar
        """
        temp_dir = Path(tempfile.gettempdir()) / f"mrai_session_{session_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)  # Eliminacion recursiva


# =============================================================================
# INSTANCIA GLOBAL DEL CLIENTE (SINGLETON)
# =============================================================================

@st.cache_resource
def get_dagster_client():
    """
    Decorador de Streamlit que mantiene el cliente en memoria cache.
    Evita recrear el cliente en cada interaccion del usuario.
    
    Returns:
        Instancia unica de DagsterClient (patron singleton)
    """
    return DagsterClient()