# src/evaluacion/metricas_segmentacion.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def cargar_mascaras(ruta_mascara_original: Path, ruta_mascara_predicha: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga una máscara original y su correspondiente máscara predicha"""
    
    original = np.load(ruta_mascara_original)
    predicha = np.load(ruta_mascara_predicha)
    
    # Asegurar que son binarias (0 y 1)
    original = (original > 0).astype(np.uint8)
    predicha = (predicha > 0.5).astype(np.uint8)
    
    # Asegurar mismo tamaño
    if original.shape != predicha.shape:
        from scipy.ndimage import zoom
        zoom_factor = (original.shape[0] / predicha.shape[0], 
                       original.shape[1] / predicha.shape[1])
        predicha = zoom(predicha, zoom_factor, order=0)
        predicha = (predicha > 0.5).astype(np.uint8)
    
    return original, predicha


def calcular_metricas_por_imagen(original: np.ndarray, predicha: np.ndarray, 
                                   imagen_id: str) -> Dict:
    """Calcula todas las métricas de segmentación para una imagen"""
    
    y_true = original.flatten()
    y_pred = predicha.flatten()
    
    # Matriz de confusión
    VP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    VN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Métricas básicas
    sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0
    especificidad = VN / (VN + FP) if (VN + FP) > 0 else 0
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    exactitud = (VP + VN) / (VP + VN + FP + FN) if (VP + VN + FP + FN) > 0 else 0
    f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
    iou = VP / (VP + FP + FN) if (VP + FP + FN) > 0 else 0
    npv = VN / (VN + FN) if (VN + FN) > 0 else 0
    
    # Matthews Correlation Coefficient
    numerador = (VP * VN) - (FP * FN)
    denominador = np.sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN))
    mcc = numerador / denominador if denominador > 0 else 0
    
    balanced_acc = (sensibilidad + especificidad) / 2
    
    # Distancias (solo si hay tumor)
    if original.sum() > 0 and predicha.sum() > 0:
        coords_orig = np.column_stack(np.where(original > 0))
        coords_pred = np.column_stack(np.where(predicha > 0))
        
        hausdorff_orig_a_pred = directed_hausdorff(coords_orig, coords_pred)[0]
        hausdorff_pred_a_orig = directed_hausdorff(coords_pred, coords_orig)[0]
        hausdorff = max(hausdorff_orig_a_pred, hausdorff_pred_a_orig)
        
        tree_pred = KDTree(coords_pred)
        tree_orig = KDTree(coords_orig)
        
        dists_orig_to_pred = tree_pred.query(coords_orig)[0]
        dists_pred_to_orig = tree_orig.query(coords_pred)[0]
        avg_surface_distance = (np.mean(dists_orig_to_pred) + np.mean(dists_pred_to_orig)) / 2
    else:
        hausdorff = np.nan
        avg_surface_distance = np.nan
    
    fpr = FP / (FP + VN) if (FP + VN) > 0 else 0
    fnr = FN / (FN + VP) if (FN + VP) > 0 else 0
    
    resultado = {
        'imagen_id': imagen_id,
        'VP': int(VP), 'FP': int(FP), 'VN': int(VN), 'FN': int(FN),
        'sensibilidad': round(sensibilidad, 4),
        'especificidad': round(especificidad, 4),
        'precision': round(precision, 4),
        'exactitud': round(exactitud, 4),
        'f1_score': round(f1, 4),
        'dice': round(f1, 4),
        'iou': round(iou, 4),
        'mcc': round(mcc, 4),
        'npv': round(npv, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'area_real': int(original.sum()),
        'area_predicha': int(predicha.sum()),
        'hausdorff_distance': round(hausdorff, 2) if not np.isnan(hausdorff) else None,
        'avg_surface_distance': round(avg_surface_distance, 2) if not np.isnan(avg_surface_distance) else None,
    }
    
    return resultado


def calcular_metricas_globales(df_metricas: pd.DataFrame) -> Dict:
    """Calcula métricas globales a partir del DataFrame"""
    
    if df_metricas.empty:
        return {}
    
    metricas_globales = {}
    
    metricas_interes = ['sensibilidad', 'especificidad', 'precision', 'exactitud', 
                        'f1_score', 'iou', 'mcc', 'balanced_accuracy']
    
    for metrica in metricas_interes:
        if metrica in df_metricas.columns:
            metricas_globales[f'{metrica}_media'] = round(df_metricas[metrica].mean(), 4)
            metricas_globales[f'{metrica}_std'] = round(df_metricas[metrica].std(), 4)
    
    # Matriz de confusión global
    metricas_globales['VP_total'] = int(df_metricas['VP'].sum())
    metricas_globales['FP_total'] = int(df_metricas['FP'].sum())
    metricas_globales['VN_total'] = int(df_metricas['VN'].sum())
    metricas_globales['FN_total'] = int(df_metricas['FN'].sum())
    
    VP = metricas_globales['VP_total']
    FP = metricas_globales['FP_total']
    VN = metricas_globales['VN_total']
    FN = metricas_globales['FN_total']
    
    metricas_globales['global_sensibilidad'] = VP / (VP + FN) if (VP + FN) > 0 else 0
    metricas_globales['global_especificidad'] = VN / (VN + FP) if (VN + FP) > 0 else 0
    metricas_globales['global_precision'] = VP / (VP + FP) if (VP + FP) > 0 else 0
    metricas_globales['global_exactitud'] = (VP + VN) / (VP + VN + FP + FN) if (VP + VN + FP + FN) > 0 else 0
    
    global_f1 = 2 * (metricas_globales['global_precision'] * metricas_globales['global_sensibilidad']) / (metricas_globales['global_precision'] + metricas_globales['global_sensibilidad']) if (metricas_globales['global_precision'] + metricas_globales['global_sensibilidad']) > 0 else 0
    metricas_globales['global_f1'] = round(global_f1, 4)
    
    return metricas_globales


def evaluar_segmentacion(directorio_mascaras_originales: Path, 
                          directorio_mascaras_predichas: Path,
                          archivo_test_csv: Path = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Evalúa todas las máscaras de segmentación
    """
    
    print("="*70)
    print("🔬 EVALUACIÓN DE SEGMENTACIÓN TUMORAL")
    print("="*70)
    
    # Verificar directorios
    if not Path(directorio_mascaras_originales).exists():
        raise FileNotFoundError(f"Directorio original no existe: {directorio_mascaras_originales}")
    
    if not Path(directorio_mascaras_predichas).exists():
        raise FileNotFoundError(f"Directorio predicho no existe: {directorio_mascaras_predichas}")
    
    # ============================================================
    # BUSCAR MÁSCARAS ORIGINALES: archivos que terminan en _m.npy
    # ============================================================
    originales = sorted(list(Path(directorio_mascaras_originales).glob("*_m.npy")))
    
    print(f"\n📂 Buscando archivos '*_m.npy' en: {directorio_mascaras_originales}")
    print(f"📂 Originales encontradas: {len(originales)}")
    for o in originales[:5]:
        print(f"   - {o.name}")
    
    # Si no encuentra con _m, buscar con mask
    if len(originales) == 0:
        originales = sorted(list(Path(directorio_mascaras_originales).glob("*mask*.npy")))
        print(f"\n📂 Buscando '*mask*.npy': {len(originales)} encontradas")
    
    # Máscaras predichas
    predichas = sorted(list(Path(directorio_mascaras_predichas).glob("*_mask.npy")))
    if len(predichas) == 0:
        predichas = sorted(list(Path(directorio_mascaras_predichas).glob("*.npy")))
    
    print(f"\n📂 Predichas encontradas: {len(predichas)}")
    for p in predichas[:5]:
        print(f"   - {p.name}")
    
    if len(originales) == 0:
        raise ValueError(f"No se encontraron máscaras originales en {directorio_mascaras_originales}")
    
    if len(predichas) == 0:
        raise ValueError(f"No se encontraron máscaras predichas en {directorio_mascaras_predichas}")
    
    # Crear diccionario de correspondencias
    mapeo_predichas = {}
    for p in predichas:
        nombre_base = p.stem.replace('_mask', '')
        mapeo_predichas[nombre_base] = p
    
    # Calcular métricas
    resultados = []
    
    for ruta_original in originales:
        # Obtener ID base (quitar _m)
        nombre_base = ruta_original.stem.replace('_m', '')
        
        # Buscar máscara predicha correspondiente
        if nombre_base in mapeo_predichas:
            ruta_predicha = mapeo_predichas[nombre_base]
            print(f"✅ Comparando: {ruta_original.name} vs {ruta_predicha.name}")
            
            try:
                original, predicha = cargar_mascaras(ruta_original, ruta_predicha)
                metricas = calcular_metricas_por_imagen(original, predicha, nombre_base)
                resultados.append(metricas)
            except Exception as e:
                print(f"❌ Error con {nombre_base}: {e}")
        else:
            print(f"⚠️ No se encontró máscara predicha para: {nombre_base}")
    
    if not resultados:
        raise ValueError("No se pudo calcular ninguna métrica")
    
    df_metricas = pd.DataFrame(resultados)
    metricas_globales = calcular_metricas_globales(df_metricas)
    
    print(f"\n✅ Evaluación completada: {len(df_metricas)} imágenes analizadas")
    
    return df_metricas, metricas_globales


def evaluar_calidad_segmentacion(directorio_mascaras_originales: Path,
                                   directorio_mascaras_predichas: Path,
                                   archivo_test_csv: Path = None,
                                   guardar_resultados: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Función principal para evaluar calidad"""
    
    print("\n🔬 INICIANDO EVALUACIÓN DE CALIDAD")
    
    df_metricas, metricas_globales = evaluar_segmentacion(
        directorio_mascaras_originales=directorio_mascaras_originales,
        directorio_mascaras_predichas=directorio_mascaras_predichas,
        archivo_test_csv=archivo_test_csv
    )
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE MÉTRICAS")
    print("="*70)
    print(f"  Imágenes evaluadas: {len(df_metricas)}")
    print(f"  F1-Score medio: {metricas_globales.get('f1_score_media', 0):.4f}")
    print(f"  Sensibilidad media: {metricas_globales.get('sensibilidad_media', 0):.4f}")
    print(f"  Precisión media: {metricas_globales.get('precision_media', 0):.4f}")
    print(f"  IoU medio: {metricas_globales.get('iou_media', 0):.4f}")
    
    # Guardar resultados
    if guardar_resultados:
        output_dir = Path(directorio_mascaras_predichas).parent / "evaluacion_calidad"
        output_dir.mkdir(exist_ok=True)
        
        df_metricas.to_csv(output_dir / "metricas_por_imagen.csv", index=False)
        
        # Guardar resumen
        with open(output_dir / "resumen_metricas.txt", "w", encoding='utf-8') as f:
            f.write("RESUMEN DE MÉTRICAS DE SEGMENTACIÓN\n")
            f.write("="*50 + "\n\n")
            for key, value in metricas_globales.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n💾 Resultados guardados en: {output_dir}")
    
    return df_metricas, metricas_globales