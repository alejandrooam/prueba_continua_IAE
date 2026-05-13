# Brain Tumor UNet Segmentation App

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Dagster](https://img.shields.io/badge/Dagster-1.5+-green.svg)](https://dagster.io)
[![UNet](https://img.shields.io/badge/Architecture-UNet-orange.svg)](https://arxiv.org/abs/1505.04597)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-cyan.svg)](https://docs.astral.sh/uv)

> **Trabajo académico** - Aplicación web para segmentación de tumores cerebrales en resonancias magnéticas (MRI) con evaluación de urgencia médica.

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Características](#características)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración de Kaggle](#configuración-de-kaggle)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Autor](#autor)
- [Licencia](#licencia)


## Descripción General

Esta aplicación permite la segmentación automática de tumores cerebrales a partir de imágenes de resonancia magnética (MRI) en formato .tif. Utilizando una arquitectura UNet, el sistema genera:

- Máscara de segmentación del tumor
- Características tumorales (tamaño, forma, ubicación)
- Nivel de urgencia basado en análisis de imagen + datos clínicos

La aplicación incluye un formulario médico donde se pueden ingresar datos clínicos del paciente (edad, grado histológico, etc.) que se combinan con el análisis de la imagen para predecir un nivel de urgencia más preciso.

## Características

- Segmentación UNet de tumores en MRI cerebrales
- Interfaz web interactiva con Streamlit
- Formulario clínico para mejorar predicción de urgencia
- Orquestación con Dagster para flujos de trabajo reproducibles
- Procesamiento paralelo con Dask para grandes volúmenes de datos
- Análisis estadístico integrado con R
- Ficha técnica automática con métricas del modelo

## Requisitos Previos

- Python 3.10 o superior
- Git para clonar el repositorio
- uv (gestor de paquetes)
- Cuenta de Kaggle (para descargar los datos)
- R (para análisis estadístico, opcional)
- GPU recomendada (funciona con CPU pero más lento)

## Instalación

1. Clonar el repositorio:

git clone https://github.com/alejandrooam/brain-tumor-segmentation-unet-app.git
cd brain-tumor-segmentation-unet-app

2. Instalar uv:

pip install uv

3. Crear entorno virtual e instalar dependencias:

uv venv

Activar entorno (Windows):
.venv\Scripts\activate

Instalar dependencias:
uv sync

## Configuración de Kaggle

Para descargar los datos necesitas una API key de Kaggle:

1. Ve a Kaggle Settings (https://www.kaggle.com/settings)
2. En la sección "API", haz clic en "Create New Token"
3. Se descargará kaggle.json
4. Coloca el archivo en la raíz del proyecto

Importante: kaggle.json ya está en .gitignore, no se subirá a GitHub.

Enlace del dataset: [INSERTAR AQUÍ EL ENLACE]

## Estructura del Proyecto

brain-tumor-segmentation-unet-app/
│
├── src/
│   ├── datos.py                   # Importación de datos desde Kaggle
│   ├── mapeo_archivos.py          # Generación del catálogo de trabajo
│   └── transformar_datos.py       # Adaptación y preprocesamiento de imágenes
│
├── Analisis/
│   ├── Analisis_imagen.py         # Mini-modelo para predicción de urgencia
│   ├── Analisis_test.py           # Análisis del conjunto de test
│   ├── Metricas_modelo.py         # Métricas de calidad de segmentación
│   └── Script_R_1.R               # Análisis estadístico en R
│
├── Aplicacion/
│   ├── streamlit.py               # Página principal de la app
│   ├── cliente.py                 # Comunicación con Dagster
│   └── pages/
│       └── Ficha_tecnica.py       # Ficha técnica del modelo
│
├── orquestador/
│   └── activos.py                 # Activos de Dagster para el flujo de trabajo
│
├── procesamiento_datos/
│   ├── procesador_dask.py         # Transformación paralela con Dask
│   └── modelo/
│       ├── modelo_unet.py         # Arquitectura UNet
│       ├── entrenar_modelo.py     # Entrenamiento del modelo
│       ├── segmentar.py           # Lógica de segmentación
│       └── caracteristicas.py     # Extracción de características tumorales
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md

## Uso de la Aplicación

Iniciar la aplicación Streamlit:

streamlit run Aplicacion/streamlit.py

La aplicación se abrirá en el navegador (normalmente http://localhost:8501).

Flujo de uso:

1. Subir imagen MRI (formato .tif)
2. Rellenar formulario clínico (edad, síntomas, historial médico)
3. Ejecutar análisis
4. Resultados: máscara de segmentación, características del tumor, nivel de urgencia (Bajo/Medio/Alto), recomendación médica

## Entrenamiento del Modelo

1. Descargar y preparar datos:

python src/datos.py
python src/mapeo_archivos.py
python src/transformar_datos.py

2. Procesar datos con Dask:

python procesamiento_datos/procesador_dask.py

3. Entrenar el modelo UNet:

python procesamiento_datos/modelo/entrenar_modelo.py

4. Evaluar el modelo:

python Analisis/Metricas_modelo.py

5. Entrenar modelo de urgencia:

python Analisis/Analisis_imagen.py

## Métricas de Evaluación

- Dice Score - Similitud entre máscara predicha y real
- IoU (Intersection over Union) - Precisión de la segmentación
- Precisión - Tasa de verdaderos positivos
- Recall - Sensibilidad del modelo

## Tecnologías Utilizadas

- Python 3.10+ - Lenguaje principal
- UNet - Arquitectura de segmentación
- Streamlit - Interfaz web
- Dagster - Orquestación de pipelines
- Dask - Procesamiento paralelo
- R - Análisis estadístico
- uv - Gestor de paquetes
- NumPy/Pandas - Manipulación de datos
- OpenCV/PIL - Procesamiento de imágenes

## Requisitos de Hardware

- Mínimo (CPU): 8GB RAM, CPU de 4 núcleos
- Recomendado (GPU): NVIDIA GPU con 4GB+ VRAM, 16GB RAM

## Autor

Alejandro - https://github.com/alejandrooam

## Licencia

Este proyecto es un trabajo académico de uso educativo. Para uso comercial, contactar al autor.
