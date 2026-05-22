import shutil
from pathlib import Path
import kagglehub

ruta_descargas = Path("C:/Users/Usuario/Desktop/Entorno/Trabajo/kaggle.json")

# Crear carpeta .kaggle si no existe
carpeta_kaggle = Path("C:/Users/Usuario/Desktop/Entorno/Trabajo/.kaggle")
carpeta_kaggle.mkdir(exist_ok=True)

datos_dir = Path("C:/Users/Usuario/Desktop/Entorno/Trabajo/DATOS")
datos_dir.mkdir(exist_ok=True, parents=True)


# Copiar el archivo
shutil.copy(ruta_descargas, carpeta_kaggle / "kaggle.json")


path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

for archivo in path.rglob("*"):
    # Copiar cada archivo directamente a DATOS
    destino = datos_dir / archivo.name
    shutil.copy2(archivo, destino)

print("Path al dataset:", path)