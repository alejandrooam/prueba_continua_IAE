# src/Analisis/Analisis_imagen.py

# =============================================================================
# MODELO DE CLASIFICACION DE RIESGO (URGENCIA TUMORAL)
# =============================================================================
# Este modulo implementa un modelo de regresion logistica para predecir
# la probabilidad de fallecimiento (urgencia clinica) de pacientes con tumores
# cerebrales a partir de caracteristicas radiomicas y datos clinicos.
# =============================================================================

# LIBRERIAS NECESARIAS
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CLASE PRINCIPAL: MODELO DE URGENCIA TUMORAL
# =============================================================================

class ModeloUrgenciaTumoral:
    """
    Modelo para predecir probabilidad de muerte (urgencia del tumor)
    Maneja automaticamente datos faltantes y clases desbalanceadas
    """
    
    def __init__(self):
        """
        Inicializa el modelo con sus componentes de preprocesamiento
        """
        # Imputador: reemplaza valores faltantes con la mediana
        self.imputador = SimpleImputer(strategy='median')
        # Escalador: estandariza las variables a media 0 y desviacion 1
        self.escalador = StandardScaler()
        # Modelo de regresion logistica (se entrenara despues)
        self.modelo = None
        # Lista de variables usadas durante el entrenamiento
        self.variables_entrenamiento = None
        # Umbral de decision optimo (calculado durante entrenamiento)
        self.umbral = 0.5
        # Riesgo base: probabilidad media de muerte en la cohorte
        self.media_riesgo_base = None
        
    def preprocesar(self, X):
        """
        Preprocesa los datos: imputa NAs y escala las variables
        
        Args:
            X: DataFrame o array con las variables predictoras
            
        Returns:
            X_escalado: Array numpy con los datos preprocesados
        """
        # Paso 1: Imputar valores faltantes con la mediana
        X_imputado = self.imputador.transform(X)
        # Paso 2: Estandarizar (media 0, desviacion 1)
        X_escalado = self.escalador.transform(X_imputado)
        return X_escalado
    
    def entrenar(self, df, variables=None, target='death01', test_size=0.2):
        """
        Entrena el modelo de regresion logistica
        
        Args:
            df: DataFrame con los datos de entrenamiento
            variables: Lista de columnas a usar como predictoras
            target: Nombre de la columna objetivo (death01)
            test_size: Proporcion de datos para validacion
            
        Returns:
            metricas: Diccionario con AUC, coeficientes y estadisticas
        """
        
        # Seleccionar variables predictoras
        if variables is None:
            variables = [
                'area', 'perimetro', 'circularidad',
                'intensidad_media_post', 'intensidad_minima_post', 
                'percentil_95_flair', 'textura_contraste',
                'age_at_initial_pathologic', 'neoplasm_histologic_grade'
            ]
            # Filtrar solo las que existen en el DataFrame
            variables = [v for v in variables if v in df.columns]
        
        # Guardar lista de variables para uso futuro
        self.variables_entrenamiento = variables
        
        # Preparar datos: separar predictores (X) y objetivo (y)
        X = df[variables].copy()
        y = df[target].copy()
        
        # Eliminar filas sin valor en la variable objetivo
        mascara_valida = ~y.isna()
        X = X[mascara_valida]
        y = y[mascara_valida]
        
        # Asegurar que y es numerico entero (0 o 1)
        y = y.astype(int)
        
        # Contar casos positivos (fallecidos) para validacion
        n_fallecidos = (y == 1).sum()
        
        # Validar que existan casos positivos
        if n_fallecidos == 0:
            raise ValueError("No hay casos positivos (death01=1) en los datos")
        
        # Dividir datos en entrenamiento y validacion
        try:
            # Si hay al menos 2 fallecidos, usar estratificacion
            if n_fallecidos >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                # Si no, hacer division simple
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        except ValueError:
            # Fallback: division simple si falla la estratificacion
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Verificar que validacion tenga ambas clases
        if len(np.unique(y_val)) < 2:
            # Si no, usar entrenamiento como validacion
            X_val = X_train.copy()
            y_val = y_train.copy()
        
        # Entrenar preprocesador con datos de entrenamiento
        self.imputador.fit(X_train)
        self.escalador.fit(self.imputador.transform(X_train))
        
        # Preprocesar ambos conjuntos
        X_train_proc = self.preprocesar(X_train)
        X_val_proc = self.preprocesar(X_val)
        
        # Configurar modelo de regresion logistica con balanceo de clases
        self.modelo = LogisticRegression(
            C=1.0,
            class_weight='balanced',  # Balancea clases desbalanceadas
            random_state=42,
            max_iter=1000
        )
        
        # Entrenar el modelo
        self.modelo.fit(X_train_proc, y_train)
        
        # Predecir probabilidades en validacion
        y_pred_proba = self.modelo.predict_proba(X_val_proc)[:, 1]
        
        # Calcular AUC (Area Under the ROC Curve)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Encontrar el mejor umbral de decision usando F1-score
        if len(np.unique(y_val)) > 1:
            umbrales = np.arange(0.1, 0.9, 0.05)
            mejores_f1 = 0
            mejor_umbral = 0.5
            for umb in umbrales:
                y_pred = (y_pred_proba >= umb).astype(int)
                if len(np.unique(y_pred)) > 1:
                    f1 = f1_score(y_val, y_pred)
                    if f1 > mejores_f1:
                        mejores_f1 = f1
                        mejor_umbral = umb
            self.umbral = mejor_umbral
        
        # Crear DataFrame con coeficientes y odds ratios
        coefs = pd.DataFrame({
            'Variable': variables,
            'Coeficiente': self.modelo.coef_[0],
            'Odds_Ratio': np.exp(self.modelo.coef_[0])
        })
        # Ordenar por magnitud del coeficiente (importancia)
        coefs = coefs.reindex(coefs['Coeficiente'].abs().sort_values(ascending=False).index)
        
        # Calcular riesgo base (probabilidad media de muerte)
        self.media_riesgo_base = y.mean()
        
        # Empaquetar metricas para retorno
        metricas = {
            'auc': auc,
            'mejor_umbral': self.umbral,
            'coeficientes': coefs,
            'variables': variables,
            'riesgo_base': self.media_riesgo_base,
            'n_fallecidos': n_fallecidos,
            'n_total': len(y)
        }
        
        return metricas
    
    def predecir_urgencia(self, df, return_proba=True):
        """
        Predice la probabilidad de muerte (urgencia) para nuevos pacientes
        
        Args:
            df: DataFrame con los datos de los pacientes
            return_proba: Si True devuelve probabilidad, si False devuelve clase binaria
            
        Returns:
            Array con probabilidades o clases predichas
        """
        # Verificar que el modelo fue entrenado
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado aun")
        
        # Identificar variables faltantes en los datos de entrada
        variables_faltantes = [v for v in self.variables_entrenamiento if v not in df.columns]
        if variables_faltantes:
            # Crear columnas faltantes con NaN (se imputaran)
            for v in variables_faltantes:
                df[v] = np.nan
        
        # Seleccionar variables en el orden correcto
        X = df[self.variables_entrenamiento].copy()
        
        # Preprocesar (imputar y escalar)
        X_proc = self.preprocesar(X)
        
        # Predecir probabilidad de muerte
        prob_muerte = self.modelo.predict_proba(X_proc)[:, 1]
        
        # Devolver probabilidad o clase segun parametro
        if return_proba:
            return prob_muerte
        else:
            return (prob_muerte >= self.umbral).astype(int)
    
    def guardar(self, ruta):
        """
        Guarda el modelo completo en disco usando pickle
        
        Args:
            ruta: Ruta donde guardar el archivo .pkl
        """
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def cargar(cls, ruta):
        """
        Carga un modelo previamente guardado desde disco
        
        Args:
            ruta: Ruta del archivo .pkl
            
        Returns:
            modelo: Instancia de ModeloUrgenciaTumoral restaurada
        """
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        return modelo


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def entrenar_modelo_urgencia(df, output_dir=None):
    """
    Funcion principal para entrenar el modelo de urgencia
    
    Args:
        df: DataFrame con los datos de entrenamiento
        output_dir: Directorio donde guardar resultados (opcional)
        
    Returns:
        modelo: Modelo entrenado
        metricas: Diccionario con metricas de rendimiento
    """
    
    # Crear instancia del modelo
    modelo = ModeloUrgenciaTumoral()
    
    # Entrenar y obtener metricas
    metricas = modelo.entrenar(df)
    
    # Guardar resultados si se especifica directorio
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo completo
        modelo.guardar(output_dir / "modelo_urgencia.pkl")
        
        # Guardar coeficientes en CSV
        metricas['coeficientes'].to_csv(output_dir / "coeficientes_modelo.csv", index=False)
        
        # Guardar resumen en archivo de texto
        with open(output_dir / "resumen_modelo.txt", "w", encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODELO DE URGENCIA TUMORAL\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"AUC ROC: {metricas['auc']:.4f}\n")
            f.write(f"Mejor umbral: {metricas['mejor_umbral']:.3f}\n")
            f.write(f"Riesgo base: {metricas['riesgo_base']:.2%}\n\n")
            f.write("VARIABLES MAS IMPORTANTES:\n")
            f.write(metricas['coeficientes'].head(10).to_string())
    
    return modelo, metricas


def diagnosticar_datos(df, target='death01'):
    """
    Diagnostico rapido de la calidad de los datos
    
    Args:
        df: DataFrame a diagnosticar
        target: Nombre de la variable objetivo
    """
    
    # Mostrar dimensiones del dataset
    print("\n" + "=" * 60)
    print("DIAGNOSTICO DE DATOS")
    print("=" * 60)
    print(f"\nDimensiones: {df.shape}")
    
    # Analizar variable objetivo
    print(f"\nVariable objetivo '{target}':")
    if target in df.columns:
        print(f"   Valores unicos: {sorted(df[target].unique())}")
        print(f"   Vivos (0): {(df[target] == 0).sum()}")
        print(f"   Fallecidos (1): {(df[target] == 1).sum()}")
        print(f"   NAs: {df[target].isna().sum()}")
        
        # Advertencia si hay pocos fallecidos
        if (df[target] == 1).sum() < 5:
            print(f"   Advertencia: Solo {(df[target]==1).sum()} fallecidos. Modelo poco robusto.")
    else:
        print(f"   Error: No se encuentra la columna '{target}'")
    
    # Mostrar primeras filas como muestra
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    return