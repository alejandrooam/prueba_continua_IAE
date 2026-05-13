import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Ficha Técnica - MRAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Ficha Técnica de Calidad")
st.markdown("### Sistema MRAI - Análisis de Tumores Cerebrales")
st.markdown("---")


# ========================
# RUTAS DE LOS ARCHIVOS
# ========================

# Determinar la ruta base
BASE_PATH = Path(__file__).parent.parent.parent.parent  # Sube hasta Trabajo
DATOS_PROCESADOS = BASE_PATH / "datos_procesados"
EVAL_CALIDAD = DATOS_PROCESADOS / "evaluacion_calidad"
RESULTADOS_R = DATOS_PROCESADOS / "resultados_r"

# ========================
# MÉTRICAS DEL MODELO
# ========================

st.header("🎯 Métricas de Calidad del Modelo de Segmentación")

# Leer resumen_metricas.txt
resumen_metricas_path = EVAL_CALIDAD / "resumen_metricas.txt"

if resumen_metricas_path.exists():
    with open(resumen_metricas_path, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    # Intentar parsear el contenido como tabla
    try:
        # Buscar líneas con formato "Métrica: Valor"
        metricas = {}
        for linea in contenido.split('\n'):
            if ':' in linea:
                key, val = linea.split(':', 1)
                if not key.strip().endswith('_total'):
                    metricas[key.strip()] = val.strip()
        
        if metricas:
            cols = st.columns(4)
            for idx, (metrica, valor) in enumerate(metricas.items()):
                with cols[idx % 4]:
                    # Intentar convertir a float y formatear
                    try:
                        valor_float = float(valor)
                        valor_formateado = f"{valor_float:.4f}"
                    except ValueError:
                        valor_formateado = valor
                    st.metric(metrica, valor_formateado)
        else:
            st.text(contenido)
    except:
        st.text(contenido)
else:
    st.warning("No se encuentra el archivo resumen_metricas.txt")

st.markdown("---")

# ========================
# MÉTRICAS POR IMAGEN
# ========================

st.header("📈 Métricas por Imagen")

metricas_csv_path = EVAL_CALIDAD / "metricas_por_imagen.csv"

if metricas_csv_path.exists():
    df_metricas = pd.read_csv(metricas_csv_path)
    st.dataframe(df_metricas, use_container_width=True)
    
    # Gráfico de métricas principales
    if 'dice' in df_metricas.columns:
        fig = px.box(df_metricas, y='dice', title='Distribución del Coeficiente Dice')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'iou' in df_metricas.columns:
        fig = px.histogram(df_metricas, x='iou', nbins=20, title='Histograma de IoU')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No se encuentra el archivo metricas_por_imagen.csv")

st.markdown("---")

# ========================
# GRÁFICAS DE R
# ========================

st.header("📊 Análisis Estadístico (R)")

col_r1, col_r2 = st.columns(2)

with col_r1:
    img1_path = RESULTADOS_R / "01_DISTRIBUCION_CIRCULARIDAD.png"
    if img1_path.exists():
        st.image(str(img1_path), caption="Distribución de la Circularidad del Tumor", use_container_width=True)
    else:
        st.info("Gráfica no disponible: 01_DISTRIBUCION_CIRCULARIDAD.png")
    
    img3_path = RESULTADOS_R / "04_TAMAÑO_VS_FORMA.png" if not (RESULTADOS_R / "03_CONTRASTE_POR_FORMA.png").exists() else RESULTADOS_R / "03_CONTRASTE_POR_FORMA.png"
    if img3_path.exists():
        st.image(str(img3_path), caption="Relación Tamaño vs Forma", use_container_width=True)
    else:
        st.info("Gráfica no disponible")

with col_r2:
    img2_path = RESULTADOS_R / "02_HETEROGENEIDAD_VS_FORMA.png"
    if img2_path.exists():
        st.image(str(img2_path), caption="Heterogeneidad vs Forma", use_container_width=True)
    else:
        st.info("Gráfica no disponible: 02_HETEROGENEIDAD_VS_FORMA.png")

st.markdown("---")

# ========================
# RESULTADOS DEL ANÁLISIS R
# ========================

st.header("📄 Informe de Análisis Estadístico")

resultados_txt_path = RESULTADOS_R / "resultados_analisis.txt"

if resultados_txt_path.exists():
    with open(resultados_txt_path, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    with st.expander("Ver informe completo", expanded=True):
        st.text(contenido)
else:
    st.warning("No se encuentra el archivo resultados_analisis.txt")

st.markdown("---")

# ========================
# RESUMEN GENERAL
# ========================

st.header("📋 Resumen General")

col_sum1, col_sum2, col_sum3 = st.columns(3)

with col_sum1:
    if df_metricas is not None and 'dice' in df_metricas.columns:
        st.metric("Dice promedio", f"{df_metricas['dice'].mean():.3f}")
    else:
        st.metric("Dice promedio", "N/A")

with col_sum2:
    if df_metricas is not None and 'iou' in df_metricas.columns:
        st.metric("IoU promedio", f"{df_metricas['iou'].mean():.3f}")
    else:
        st.metric("IoU promedio", "N/A")

with col_sum3:
    st.metric("Fecha análisis", datetime.now().strftime("%d/%m/%Y"))

st.markdown("---")

# Limitaciones y recomendaciones
st.header("⚠️ Limitaciones del Sistema")
col_lim1, col_lim2 = st.columns(2)

with col_lim1:
    st.subheader("Limitaciones conocidas")
    st.markdown("""
    - ❌ No sustituye el juicio clínico especializado
    - ❌ No validado para tumores pediátricos (<18 años)
    """)

with col_lim2:
    st.subheader("Recomendaciones de uso")
    st.markdown("""
    - ✅ Utilizar como apoyo diagnóstico, no como único criterio
    - ✅ Confirmar hallazgos con especialista en neuroimagen
    - ✅ Verificar calidad de la imagen antes del análisis
    - ✅ Mantener actualizado el sistema con nuevas versiones
    """)

st.markdown("---")

# Contacto y soporte
st.header("📞 Soporte y Contacto")

col_cont1, col_cont2, col_cont3 = st.columns(3)

with col_cont1:
    st.markdown("**Soporte técnico**")
    st.markdown("📧 tecnologia@mrai.com")
    st.markdown("📞 +34 900 123 456")
    st.markdown("🕒 24/7 para urgencias")

with col_cont2:
    st.markdown("**Validación clínica**")
    st.markdown("📧 clinica@mrai.com")
    st.markdown("📞 +34 900 123 457")
    st.markdown("🕒 L-V 9:00-18:00")

with col_cont3:
    st.markdown("**Reportes y sugerencias**")
    st.markdown("📧 feedback@mrai.com")
    st.markdown("🌐 www.mrai.com/support")

st.markdown("---")
st.caption(f"📅 Ficha técnica actualizada: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.caption("Documento confidencial - Propiedad de MRAI Medical Systems")