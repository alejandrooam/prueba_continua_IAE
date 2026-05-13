# src/orquestador/activos.py
import os
from dagster import asset, Output, AssetExecutionContext, MetadataValue
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from pathlib import Path
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from procesamiento_datos.procesador_dask import DaskBrainProcessor
from mapeo_archivos import crear_indice_archivos
from transformar_datos import procesar_imagen_completo
from procesamiento_datos.modelo.modelo_unet import DobleConv, UNet
from procesamiento_datos.modelo.entrenar_modelo import MRIDataset, entrenar_unet, encontrar_mejor_umbral
from procesamiento_datos.modelo.segmentar import segmentar_imagen, segmentar_lote
from procesamiento_datos.modelo.caracteristicas import extraer_caracteristicas, generar_dataset_features



@asset
def catalogo_maestro():
    """
    Crea el catálogo con todos los archivos
    """
    ruta_base = r"Trabajo\DATOS"
    ruta_csv = os.path.join(ruta_base, "data.csv")   
    df_clinico = pd.read_csv(ruta_csv)
    df_archivos = crear_indice_archivos(ruta_base, df_clinico, detectar=True)

    df_catalogo = pd.merge(
        df_archivos,
        df_clinico,
        left_on='id_paciente',
        right_on='Patient',
        how='left'
    )
    
    # Normalizar rutas
    df_catalogo['ruta_imagen'] = df_catalogo['ruta_imagen'].str.replace('\\', '/')
    
    if 'ruta_mascara' in df_catalogo.columns:
        df_catalogo['ruta_mascara'] = df_catalogo['ruta_mascara'].str.replace('\\', '/')
    
    # Guardar copia
    df_catalogo.to_csv("Trabajo/DATOS/catalogo_maestro.csv", index=False)

    n_pacientes = df_catalogo['id_paciente'].nunique()
    n_imagenes = len(df_catalogo)
    n_tumores = df_catalogo['mascara_tiene_tumor'].sum()
    
    return Output(
        df_catalogo,
        metadata={
            "total_pacientes": n_pacientes,
            "total_imagenes": n_imagenes,
            "con_tumor": int(n_tumores),
            "sin_tumor": int(n_imagenes - n_tumores),
            "porcentaje_tumor": round(float(n_tumores/n_imagenes*100), 2)
        }
    )


@asset
def imagenes_procesadas(catalogo_maestro):
    """
    Procesa imágenes en paralelo con Dask
    """
    import pandas as pd
    import os
    import traceback
    
    # CREAR CARPETA
    os.makedirs("Trabajo/datos_procesados", exist_ok=True)
    
    try:
        processor = DaskBrainProcessor(n_workers=4)
    except Exception as e:
        raise RuntimeError(f"No se pudo iniciar Dask: {type(e).__name__}: {e}")
    
    try:
        resultados = processor.procesar_todas_imagenes(catalogo_maestro)
    except Exception as e:
        raise RuntimeError(f"Fallo en procesamiento Dask:\n{type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        processor.shutdown()
        
    #CALCULAR MÉTRICAS
    n_procesadas = len(resultados)
    n_tumores_proc = sum(1 for r in resultados if r.get('tiene_tumor', False))
    n_sanos_proc = n_procesadas - n_tumores_proc
    
    return Output(
        resultados,
        metadata={
            "imagenes_procesadas": n_procesadas,
            "con_tumor": n_tumores_proc,
            "sin_tumor": n_sanos_proc,
            "porcentaje_tumor": round(n_tumores_proc/n_procesadas*100, 2) if n_procesadas > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    )


@asset
def dividir_dataset_balanceado(imagenes_procesadas):
    """
    Dividimos en entrenamiento/validación/test con ESTRATIFICACIÓN
    y calcula PESOS DE CLASE para entrenamiento balanceado

    - No se pierde información (se usan todos los datos)
    - No se crean datos nuevos (evita overfitting)
    - Los pesos hacen que los tumores tengan más importancia
    """
    
    # Convertir a DataFrame
    df = pd.DataFrame(imagenes_procesadas)
    
    
    # DIVISIÓN ESTRATIFICADA
    # Primera división: 70% entrenamiento, 30% temporal
    train, temp = train_test_split(
        df,
        test_size=0.3,
        stratify=df['tiene_tumor'],
        random_state=42,  # La semilla permite reproducir el modelo
        shuffle=True
    )
    
    # Segunda división: del 30% temporal, mitad val (15%) y mitad test (15%)
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp['tiene_tumor'],
        random_state=42,
        shuffle=True
    )
    
    # CALCULAR PESOS DE CLASE para el entrenamiento
    
    total_pixeles_fondo = 0
    total_pixeles_tumor = 0
    for _, fila in df.iterrows():
        path_mask = fila.ruta_mascara
        mask = np.load(path_mask)
        pixeles_fondo = np.sum(mask == 0)
        pixeles_tumor = np.sum(mask == 1)
        
        total_pixeles_fondo += pixeles_fondo
        total_pixeles_tumor += pixeles_tumor
    
    total_pixeles = total_pixeles_fondo + total_pixeles_tumor

    peso_fondo = total_pixeles / (2 * total_pixeles_fondo)
    peso_tumor = total_pixeles / (2 * total_pixeles_tumor)
    pesos_clase = np.array([peso_fondo,peso_tumor])
    
    
    
    
    
    pesos_imagen = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),  # 0 = sano, 1 = tumor
        y=train['tiene_tumor']
    )
    
    # Calcular factor de peso (cuánto más importante es el tumor)
    factor_peso = pesos_imagen[1] / pesos_imagen[0]
    
    # GUARDAR ARCHIVOS
    train.to_csv("Trabajo/datos_procesados/train.csv", index=False)
    val.to_csv("Trabajo/datos_procesados/val.csv", index=False)
    test.to_csv("Trabajo/datos_procesados/test.csv", index=False)
    
    # Guardar pesos para usarlos durante el entrenamiento
    np.save("Trabajo/datos_procesados/pesos_imagen.npy", pesos_imagen)
    np.save("Trabajo/datos_procesados/pesos_clase.npy", pesos_clase)
    
    # Guardar metadata adicional
    metadata = {
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'train_tumores': int(sum(train['tiene_tumor'])),
        'train_sanos': int(sum(train['tiene_tumor'] == 0)),
        'peso_sano': float(pesos_clase[0]),
        'peso_tumor': float(pesos_clase[1]),
        'factor_peso_tumor': float(factor_peso),
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar metadata en JSON
    import json
    with open("Trabajo/datos_procesados/metadata_split.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    
    return Output(
        metadata,
        metadata=metadata
    )

BASE_DIR = Path("Trabajo")  
DATOS_PROCESADOS = BASE_DIR / "datos_procesados"

@asset
def modelo_unet_entrenado(context: AssetExecutionContext):
    """Entrena U-Net con balanceo de clases"""
    
    # Crear un logger que use el contexto de Dagster
    class DagsterLogger:
        def __init__(self, context):
            self.context = context
        
        def log(self, message):
            self.context.log.info(message)
    
    logger = DagsterLogger(context)
    
    train_csv = DATOS_PROCESADOS / "train.csv"
    val_csv = DATOS_PROCESADOS / "val.csv"
    pesos_clase = DATOS_PROCESADOS / "pesos_clase.npy"
    
    logger.log("="*60)
    logger.log("🎯 INICIANDO ENTRENAMIENTO DE U-NET")
    logger.log("="*60)
    logger.log(f"📁 Directorio base: {BASE_DIR}")
    logger.log(f"📁 Datos procesados: {DATOS_PROCESADOS}")
    
    model = entrenar_unet(
        train_csv=train_csv,
        val_csv=val_csv,
        images_dir=DATOS_PROCESADOS,
        pesos_clase_path=pesos_clase,
        epochs=10, 
        lr=1e-3,
        batch_size=26,
        logger=logger  # Pasar el logger
    )
    
    model_path = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
    torch.save(model.state_dict(), model_path)
    logger.log(f"💾 Modelo guardado en: {model_path}")

    from procesamiento_datos.modelo.entrenar_modelo import diagnosticar_modelo 
    diagnosticar_modelo(model, val_csv, DATOS_PROCESADOS, device='cpu', logger=logger)
    
    logger.log("🔍 Encontrando mejor umbral...")
    umbral = encontrar_mejor_umbral(
        model, val_csv, DATOS_PROCESADOS, 
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=8,
        logger=logger
    )
    
    np.save(DATOS_PROCESADOS / "umbral.npy", umbral)
    logger.log(f"🎯 Mejor umbral encontrado: {umbral:.3f}")
    logger.log("✅ PROCESO COMPLETADO EXITOSAMENTE")
    
    return str(model_path)

@asset
def mejor_umbral(context: AssetExecutionContext):
    # Crear un logger que use el contexto de Dagster
    class DagsterLogger:
        def __init__(self, context):
            self.context = context
        
        def log(self, message):
            self.context.log.info(message)
    
    logger = DagsterLogger(context)

    val_csv = DATOS_PROCESADOS / "val.csv"
    modelo_unet_entrenado = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
        # Cargar modelo
    model = UNet(entrada=3, salida=1)
    model.load_state_dict(torch.load(modelo_unet_entrenado, map_location='cpu'))

    logger.log("🔍 Encontrando mejor umbral...")
    umbral = encontrar_mejor_umbral(
        model, val_csv, DATOS_PROCESADOS, 
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=8,
        logger=logger
    )
    
    np.save(DATOS_PROCESADOS / "umbral.npy", umbral)
    logger.log(f"🎯 Mejor umbral encontrado: {umbral:.3f}")
    logger.log("✅ PROCESO COMPLETADO EXITOSAMENTE")

@asset
def segmentaciones_test(): #Los datos test son para calificar mi modelo final, si es bueno o no
    """Segmenta todas las imágenes de test"""
    test_df = pd.read_csv(DATOS_PROCESADOS / "test.csv")
    
    # Cargar modelo
    model = UNet(entrada=3, salida=1)
    modelo_unet_entrenado = DATOS_PROCESADOS / "modelo_unet_mejor.pth"
    model.load_state_dict(torch.load(modelo_unet_entrenado, map_location='cpu'))
    #UNA VEZ CARGADO modelo_unet_entrenado MEJOR HACER QUE ESTO NO DEPENDA DEL ANTERIOR PQ TARDA MUCHO EN CARGAR Y MEJOR PONER LA RUTA DIRECTAMENTE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Busco si tengo gpu para trabajar ahí
    model.to(device) #Le digo al modelo dónde trabajamos
    
    # Preparar lista de imágenes
    imagenes_test = [DATOS_PROCESADOS / Path(row['ruta_procesada']).name for _, row in test_df.iterrows()]
    
    output_dir = DATOS_PROCESADOS / "mascaras_predichas"
    umbral = np.load(DATOS_PROCESADOS / "umbral.npy")
    resultados = segmentar_lote(
        model=model,
        lista_imagenes=imagenes_test,
        output_dir=output_dir,
        device=device,
        umbral= umbral
    )
    
    # Guardar registro de segmentaciones
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(DATOS_PROCESADOS / "resultados_segmentacion.csv", index=False)
    
    return str(output_dir)


@asset
def caracteristicas_tumorales(segmentaciones_test: str):
    """Extrae las 7 características cuantitativas + previas"""
    test_df = pd.read_csv(DATOS_PROCESADOS / "test.csv")
    
    # Cargar genómica si existe (ajusta el nombre del archivo)
    DATOS = BASE_DIR / "DATOS"
    catalogo_path = DATOS / "catalogo_maestro.csv"
    catalogo = pd.read_csv(catalogo_path)
    
    df_previa = catalogo.drop(catalogo.columns[1:6], axis=1)
    
    # Extraer características
    caract_df = generar_dataset_features(
        df_segmentacion=test_df.iloc[:,:3],
        directorio_imagenes=DATOS_PROCESADOS,
        directorio_mascaras=segmentaciones_test,
        df_previa=df_previa
    )
    
    # Guardar
    output_path = DATOS_PROCESADOS / "caracteristicas_tumorales.csv"
    caract_df.to_csv(output_path, index=False)
    
    return str(output_path)

from Analisis.Analisis_test import limpiar_datos, analisis_descriptivo

@asset
def analisis_datos():
    """Analiza los datos test con los que hemos entrenado el modelo"""
    datos = DATOS_PROCESADOS / "caracteristicas_tumorales.csv"
    datos_limpios = limpiar_datos(datos)
    analisis_descriptivo(datos_limpios)

@asset
def calidad_modelo_segmentacion(context: AssetExecutionContext):
    from Analisis.Metricas_modelo import evaluar_calidad_segmentacion
    
    # Las máscaras originales están en DATOS_PROCESADOS con nombre *_m.npy
    directorio_originales = DATOS_PROCESADOS
    
    # Las máscaras predichas están en mascaras_predichas
    directorio_predichas = DATOS_PROCESADOS / "mascaras_predichas"
    
    context.log.info(f"📂 Buscando originales en: {directorio_originales}")
    context.log.info(f"📂 Buscando predichas en: {directorio_predichas}")
    
    # Listar algunos archivos para diagnóstico
    import glob
    originales_m = list(directorio_originales.glob("*_m.npy"))
    context.log.info(f"📊 Archivos *_m.npy encontrados: {len(originales_m)}")
    if len(originales_m) > 0:
        context.log.info(f"   Ejemplo: {originales_m[0].name}")
    
    df_metricas, metricas_globales = evaluar_calidad_segmentacion(
        directorio_mascaras_originales=directorio_originales,
        directorio_mascaras_predichas=directorio_predichas,
        guardar_resultados=True
    )
    
    context.log.info(f"📊 F1-Score medio: {metricas_globales.get('f1_score_media', 0):.4f}")
    context.log.info(f"📊 Sensibilidad media: {metricas_globales.get('sensibilidad_media', 0):.4f}")
    context.log.info(f"📊 Precisión media: {metricas_globales.get('precision_media', 0):.4f}")
    
    return metricas_globales

@asset
def entrenar_modelo_urgencia(context: AssetExecutionContext):
    from Analisis.Analisis_imagen import entrenar_modelo_urgencia
    
    # ============================================================
    # USA EL ARCHIVO CORRECTO (el que me acabas de mostrar)
    # ============================================================
    df = pd.read_csv(DATOS_PROCESADOS / "tumores_limpio.csv")
    
    # ============================================================
    # DIAGNÓSTICO COMPLETO
    # ============================================================
    context.log.info("="*60)
    context.log.info("🔍 DIAGNÓSTICO DE DATOS")
    context.log.info("="*60)
    
    context.log.info(f"Shape: {df.shape}")
    
    # Verificar death01
    if 'death01' in df.columns:
        # Contar valores (incluyendo NaNs)
        n_vivos = (df['death01'] == 0).sum()
        n_fallecidos = (df['death01'] == 1).sum()
        n_nulos = df['death01'].isna().sum()
        
        context.log.info(f"\n📊 death01 - Vivos: {n_vivos}")
        context.log.info(f"   death01 - Fallecidos: {n_fallecidos}")
        context.log.info(f"   death01 - Nulos: {n_nulos}")
        
        # Mostrar algunos fallecidos
        if n_fallecidos > 0:
            context.log.info(f"\n📋 Ejemplo de filas con death01=1:")
            fallecidos = df[df['death01'] == 1]
            for idx, row in fallecidos.head(3).iterrows():
                context.log.info(f"   {row['id_paciente']}: area={row['area']:.0f}, circularidad={row['circularidad']:.3f}, edad={row['age_at_initial_pathologic']}")
    
    # Filtrar solo donde tenemos death01 conocido (no nulo)
    df_conocidos = df[df['death01'].notna()].copy()
    
    context.log.info(f"\n🩻 Pacientes con death01 conocido: {len(df_conocidos)}")
    
    if len(df_conocidos) == 0:
        context.log.error("❌ No hay pacientes con death01 conocido")
        return None
    
    n_fallecidos_conocidos = (df_conocidos['death01'] == 1).sum()
    context.log.info(f"   Vivos: {(df_conocidos['death01'] == 0).sum()}")
    context.log.info(f"   Fallecidos: {n_fallecidos_conocidos}")
    
    if n_fallecidos_conocidos == 0:
        context.log.error("❌ No hay fallecidos en los datos conocidos")
        return None
    
    # ============================================================
    # ENTRENAR MODELO
    # ============================================================
    
    # Seleccionar variables predictoras
    variables_predictoras = [
        'area', 'perimetro', 'circularidad',
        'intensidad_media_post', 'intensidad_minima_post', 
        'percentil_95_flair', 'textura_contraste',
        'age_at_initial_pathologic', 'neoplasm_histologic_grade'
    ]
    
    # Filtrar columnas que existen
    variables_existentes = [v for v in variables_predictoras if v in df_conocidos.columns]
    
    context.log.info(f"\n📊 Variables predictoras: {variables_existentes}")
    
    # Preparar datos
    X = df_conocidos[variables_existentes]
    y = df_conocidos['death01']
    
    # Imputar valores faltantes en X
    from sklearn.impute import SimpleImputer
    imputador = SimpleImputer(strategy='median')
    X_imputado = imputador.fit_transform(X)
    
    # Escalar
    from sklearn.preprocessing import StandardScaler
    escalador = StandardScaler()
    X_scaled = escalador.fit_transform(X_imputado)
    
    # Entrenar modelo
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    # Dividir
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar con balanceo de clases
    modelo = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    modelo.fit(X_train, y_train)
    
    # Evaluar
    y_pred_proba = modelo.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    context.log.info("\n" + "="*60)
    context.log.info("📊 RESULTADOS DEL MODELO")
    context.log.info("="*60)
    context.log.info(f"   AUC ROC: {auc:.4f}")
    
    # Coeficientes
    coefs = pd.DataFrame({
        'Variable': variables_existentes,
        'Coeficiente': modelo.coef_[0],
        'Odds_Ratio': np.exp(modelo.coef_[0])
    })
    coefs = coefs.reindex(coefs['Coeficiente'].abs().sort_values(ascending=False).index)
    
    context.log.info("\n📈 VARIABLES MÁS IMPORTANTES:")
    for _, row in coefs.head(5).iterrows():
        direccion = "🔴 AUMENTA el riesgo" if row['Coeficiente'] > 0 else "🟢 DISMINUYE el riesgo"
        context.log.info(f"   {row['Variable']}: {row['Coeficiente']:.4f} → {direccion}")
    
    # Guardar modelo
    output_dir = DATOS_PROCESADOS / "modelo_urgencia"
    output_dir.mkdir(exist_ok=True)
    
    import pickle
    with open(output_dir / "modelo_urgencia.pkl", 'wb') as f:
        pickle.dump({
            'modelo': modelo,
            'imputador': imputador,
            'escalador': escalador,
            'variables': variables_existentes,
            'auc': auc,
            'coeficientes': coefs
        }, f)
    
    context.log.info(f"\n💾 Modelo guardado en: {output_dir / 'modelo_urgencia.pkl'}")
    
    return {
        'auc': float(auc),
        'n_vivos': int((y == 0).sum()),
        'n_fallecidos': int((y == 1).sum()),
        'variables': variables_existentes,
        'coeficientes': coefs.to_dict()
    }
