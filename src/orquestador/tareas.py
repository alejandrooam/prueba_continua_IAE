from dagster import job
from .activos import catalogo_maestro, imagenes_procesadas, dividir_dataset

@job
def pipeline_completo():
    """
    Pipeline que ejecuta todo el flujo
    """
    catalogo = catalogo_maestro()
    procesadas = imagenes_procesadas(catalogo)
    dividir_dataset(procesadas)

@job
def solo_procesamiento():
    """
    Solo procesa imágenes (útil para nuevos datos)
    """
    catalogo = catalogo_maestro()
    imagenes_procesadas(catalogo)