# src/modelo_urgencia/modelo_urgencia.py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class ModeloUrgenciaTumoral:
    """
    Modelo sencillo para predecir probabilidad de muerte (urgencia del tumor)
    Maneja automáticamente datos faltantes y clases desbalanceadas
    """
    
    def __init__(self):
        self.imputador = SimpleImputer(strategy='median')
        self.escalador = StandardScaler()
        self.modelo = None
        self.variables_entrenamiento = None
        self.umbral = 0.5
        self.media_riesgo_base = None
        
    def preprocesar(self, X):
        """Preprocesa los datos: imputa NAs y escala"""
        X_imputado = self.imputador.transform(X)
        X_escalado = self.escalador.transform(X_imputado)
        return X_escalado
    
    def entrenar(self, df, variables=None, target='death01', test_size=0.2):
        """
        Entrena el modelo
        """
        
        print("="*60)
        print("🏥 ENTRENANDO MODELO DE URGENCIA TUMORAL")
        print("="*60)
        
        # Seleccionar variables
        if variables is None:
            variables = [
                'area', 'perimetro', 'circularidad',
                'intensidad_media_post', 'intensidad_minima_post', 
                'percentil_95_flair', 'textura_contraste',
                'age_at_initial_pathologic', 'neoplasm_histologic_grade'
            ]
            variables = [v for v in variables if v in df.columns]
        
        self.variables_entrenamiento = variables
        print(f"\n📊 Variables predictoras: {len(variables)}")
        for v in variables:
            print(f"   - {v}")
        
        # Preparar datos
        X = df[variables].copy()
        y = df[target].copy()
        
        # Eliminar filas sin target
        mascara_valida = ~y.isna()
        X = X[mascara_valida]
        y = y[mascara_valida]
        
        # Asegurar que y es numérico
        y = y.astype(int)
        
        print(f"\n📋 Total pacientes: {len(X)}")
        print(f"   Vivos (clase 0): {(y==0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")
        print(f"   Fallecidos (clase 1): {(y==1).sum()} ({100*(y==1).sum()/len(y):.1f}%)")
        
        # Verificar si hay suficientes fallecidos
        n_fallecidos = (y==1).sum()
        
        if n_fallecidos == 0:
            print("\n⚠️ ERROR: No hay pacientes fallecidos en los datos!")
            print("   No se puede entrenar el modelo.")
            raise ValueError("No hay casos positivos (death01=1) en los datos")
        
        if n_fallecidos < 5:
            print(f"\n⚠️ ADVERTENCIA: Solo hay {n_fallecidos} pacientes fallecidos.")
            print("   El modelo tendrá poca capacidad predictiva.")
        
        # Dividir entrenamiento/validación
        # Para clases muy desbalanceadas, usar stratify pero con cuidado
        try:
            if n_fallecidos >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        except ValueError:
            # Si falla la estratificación, hacer split normal
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"\n📚 Entrenamiento: {len(X_train)} pacientes")
        print(f"   Vivos: {(y_train==0).sum()}, Fallecidos: {(y_train==1).sum()}")
        print(f"📖 Validación: {len(X_val)} pacientes")
        print(f"   Vivos: {(y_val==0).sum()}, Fallecidos: {(y_val==1).sum()}")
        
        # Verificar que el conjunto de validación tenga ambas clases
        if len(np.unique(y_val)) < 2:
            print(f"\n⚠️ El conjunto de validación tiene solo una clase: {np.unique(y_val)[0]}")
            print("   Se usará el conjunto de entrenamiento para evaluar.")
            X_val = X_train.copy()
            y_val = y_train.copy()
        
        # Contar NAs
        print("\n🔍 Datos faltantes en entrenamiento:")
        for v in variables:
            nulos = X_train[v].isna().sum()
            if nulos > 0:
                print(f"   {v}: {nulos} ({100*nulos/len(X_train):.1f}%)")
        
        # Entrenar preprocesador
        print("\n⚙️ Entrenando preprocesador...")
        self.imputador.fit(X_train)
        self.escalador.fit(self.imputador.transform(X_train))
        
        # Preprocesar
        X_train_proc = self.preprocesar(X_train)
        X_val_proc = self.preprocesar(X_val)
        
        # Entrenar modelo
        print("🤖 Entrenando modelo de regresión logística...")
        
        # Pesos para balancear clases
        proporcion_fallecidos = (y_train == 1).sum() / len(y_train)
        print(f"   Proporción fallecidos en entrenamiento: {proporcion_fallecidos:.2%}")
        
        self.modelo = LogisticRegression(
            C=1.0,
            class_weight='balanced',  # Siempre balancear para este caso
            random_state=42,
            max_iter=1000
        )
        self.modelo.fit(X_train_proc, y_train)
        
        # Evaluar
        y_pred_proba = self.modelo.predict_proba(X_val_proc)[:, 1]
        
        # Calcular AUC (solo si hay dos clases en validación)
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, y_pred_proba)
        else:
            auc = 0.5
            print(f"   ⚠️ No se pudo calcular AUC (validación con una sola clase)")
        
        # Encontrar mejor umbral
        if len(np.unique(y_val)) > 1:
            umbrales = np.arange(0.1, 0.9, 0.05)
            mejores_f1 = 0
            mejor_umbral = 0.5
            for umb in umbrales:
                y_pred = (y_pred_proba >= umb).astype(int)
                if len(np.unique(y_pred)) > 1:  # Evitar f1_score con una clase
                    f1 = f1_score(y_val, y_pred)
                    if f1 > mejores_f1:
                        mejores_f1 = f1
                        mejor_umbral = umb
            self.umbral = mejor_umbral
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL MODELO")
        print("="*60)
        print(f"   AUC ROC: {auc:.4f}")
        print(f"   Mejor umbral: {self.umbral:.3f}")
        
        # Mostrar coeficientes
        coefs = pd.DataFrame({
            'Variable': variables,
            'Coeficiente': self.modelo.coef_[0],
            'Odds_Ratio': np.exp(self.modelo.coef_[0])
        })
        coefs = coefs.reindex(coefs['Coeficiente'].abs().sort_values(ascending=False).index)
        
        print("\n📈 VARIABLES QUE MÁS INFLUYEN EN LA URGENCIA:")
        for _, row in coefs.head(5).iterrows():
            direccion = "🔴 AUMENTA el riesgo" if row['Coeficiente'] > 0 else "🟢 DISMINUYE el riesgo"
            print(f"   {row['Variable']}: {row['Coeficiente']:.4f} → {direccion}")
        
        # Calcular riesgo base (probabilidad media de muerte)
        self.media_riesgo_base = y.mean()
        print(f"\n📊 Riesgo base de muerte en la cohorte: {self.media_riesgo_base:.2%}")
        
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
        """
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        # Verificar variables faltantes
        variables_faltantes = [v for v in self.variables_entrenamiento if v not in df.columns]
        if variables_faltantes:
            print(f"⚠️ Variables faltantes (se imputarán): {variables_faltantes}")
            for v in variables_faltantes:
                df[v] = np.nan
        
        # Seleccionar variables en el orden correcto
        X = df[self.variables_entrenamiento].copy()
        
        # Preprocesar
        X_proc = self.preprocesar(X)
        
        # Predecir probabilidad
        prob_muerte = self.modelo.predict_proba(X_proc)[:, 1]
        
        if return_proba:
            return prob_muerte
        else:
            return (prob_muerte >= self.umbral).astype(int)
    
    def guardar(self, ruta):
        """Guarda el modelo en disco"""
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Modelo guardado en: {ruta}")
    
    @classmethod
    def cargar(cls, ruta):
        """Carga el modelo desde disco"""
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        print(f"📂 Modelo cargado desde: {ruta}")
        return modelo


def entrenar_modelo_urgencia(df, output_dir=None):
    """Función principal para entrenar el modelo"""
    
    modelo = ModeloUrgenciaTumoral()
    metricas = modelo.entrenar(df)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        modelo.guardar(output_dir / "modelo_urgencia.pkl")
        metricas['coeficientes'].to_csv(output_dir / "coeficientes_modelo.csv", index=False)
        
        # Guardar también el resumen
        with open(output_dir / "resumen_modelo.txt", "w", encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MODELO DE URGENCIA TUMORAL\n")
            f.write("="*60 + "\n\n")
            f.write(f"AUC ROC: {metricas['auc']:.4f}\n")
            f.write(f"Mejor umbral: {metricas['mejor_umbral']:.3f}\n")
            f.write(f"Riesgo base: {metricas['riesgo_base']:.2%}\n\n")
            f.write("VARIABLES MÁS IMPORTANTES:\n")
            f.write(metricas['coeficientes'].head(10).to_string())
    
    return modelo, metricas


def diagnosticar_datos(df, target='death01'):
    """Diagnóstico rápido de los datos"""
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICO DE DATOS")
    print("="*60)
    
    print(f"\n📊 Dimensiones: {df.shape}")
    
    print(f"\n🎯 Variable objetivo '{target}':")
    if target in df.columns:
        print(f"   Valores únicos: {sorted(df[target].unique())}")
        print(f"   Vivos (0): {(df[target]==0).sum()}")
        print(f"   Fallecidos (1): {(df[target]==1).sum()}")
        print(f"   NAs: {df[target].isna().sum()}")
        
        # Verificar si hay suficientes
        if (df[target]==1).sum() < 5:
            print(f"   ⚠️ Solo {(df[target]==1).sum()} fallecidos. Muy pocos para un modelo robusto.")
    else:
        print(f"   ❌ No se encuentra la columna '{target}'")
    
    # Mostrar primeras filas
    print("\n📋 Primeras 5 filas:")
    print(df.head())
    
    return


# ============================================================
# EJECUCIÓN DIRECTA PARA PRUEBA
# ============================================================

if __name__ == "__main__":
    # Crear datos de ejemplo con fallecidos
    data = """area,circularidad,textura_contraste,age_at_initial_pathologic,neoplasm_histologic_grade,death01
326.0,0.1395,997.42,20.0,1.0,0
9.0,1.6646,14808.11,54.0,2.0,0
5860.0,0.5317,3293.98,57.0,1.0,0
372.0,0.4802,2452.33,37.0,1.0,0
4254.0,0.1748,940.13,31.0,1.0,1
192.0,0.2207,3467.64,69.0,1.0,0
2054.0,0.2624,1477.20,58.0,2.0,1
"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(data))
    
    # Diagnóstico
    diagnosticar_datos(df)
    
    # Entrenar
    modelo, metricas = entrenar_modelo_urgencia(df, output_dir="modelo_prueba")
    
    # Predecir
    nuevo = pd.DataFrame({
        'area': [3000],
        'circularidad': [0.30],
        'textura_contraste': [5000],
        'age_at_initial_pathologic': [65],
        'neoplasm_histologic_grade': [3]
    })
    
    urgencia = modelo.predecir_urgencia(nuevo)[0]
    print(f"\n🔮 URGENCIA DEL TUMOR: {urgencia:.3f}")
    print(f"   Riesgo de muerte: {urgencia*100:.1f}%")