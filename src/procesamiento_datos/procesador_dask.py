from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import os
import traceback

class DaskBrainProcessor: #Definimos la clase para poder trabajar con dask bien
    
    def __init__(self, n_workers=1): #Por defecto vamos a tener un trabjador
        self.n_workers = n_workers #Guardamos como atributo el número de trabajadores
        self.cluster = LocalCluster( #Usamos la clase LocalCluster para iniciar el cluster
            n_workers=n_workers, #Esto es el número de trabajadores
            threads_per_worker=2, #El número de hilos por trabajador
            memory_limit='4GB' #La memoria límite por trabajador
        )
        self.client = Client(self.cluster) #Crea la interfaz para enviar tareas
    
    def procesar_todas_imagenes(self, catalogo_df): #Método que espera un dataframe
        """
        Procesa todas las imágenes en paralelo
        """
        os.makedirs("Trabajo/datos_procesados", exist_ok=True) #Si no la tenemos creamos una carpeta para los datos_procesados
        
        # Dividir en lotes
        chunk_size = max(1, len(catalogo_df) // self.n_workers) #Si hay mas trabajadores que imágenes les paso a cada uno una, si no pues divido las imágenes equitativamente
        chunks = []
        for i in range(0, len(catalogo_df), chunk_size): #Agrupamos los lotes de imágenes que les vamos a pasar a cada trabajador
            chunks.append(catalogo_df.iloc[i:i+chunk_size])
    
        
        def procesar_lote(lote_df):
            """Procesa un lote de imágenes"""
            
            resultados = []
            
            import sys
            import os
            from pathlib import Path
            
            # Añadir la carpeta src al path para que Dask workers la encuentren
            src_path = Path(__file__).parent.parent  # sube de procesamiento_datos a src
            sys.path.insert(0, str(src_path))

            # Añadimos la ruta del proyecto
            #proyecto_path = r"C:\Users\Usuario\Desktop\Entorno\Trabajo"
            #if proyecto_path not in sys.path: 
            #    sys.path.insert(0, proyecto_path) #Añadimos en posición 0 (máxima prioridad) la ruta donde estamos trabajando si no la teníamos ya ahí
            
            from transformar_datos import procesar_imagen_completo #Importamos la función de procesamiento
            
            for idx, (_, fila) in enumerate(lote_df.iterrows()):
                id_paciente = fila.get('id_paciente')
                num_corte = fila.get('num_corte')
                    
                    # Procesar imagen
                img, mask = procesar_imagen_completo(fila, entrenando=True) 
                    
                if img is not None:
                    nombre_archivo = f"Trabajo/datos_procesados/{id_paciente}_{num_corte}.npy"
                    np.save(nombre_archivo, img)
            
                    nombre_mascara = f"Trabajo/datos_procesados/{id_paciente}_{num_corte}_m.npy"
                    np.save(nombre_mascara,mask)    
                    
                    resultados.append({
                        'paciente': id_paciente,
                        'num_corte': num_corte,
                        'ruta_procesada': nombre_archivo,
                        'ruta_mascara' : nombre_mascara,
                        'tiene_tumor': fila.get('mascara_tiene_tumor', False)
                    })
            
            return resultados
        
        # Enviamos las tareas
        futures = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > 0:
                future = self.client.submit(procesar_lote, chunk) #Le mandamos una tarea y no esperamos a que termine sino que nos quedamos con la promesa y seguimos
                futures.append(future)
        
        # Recoger resultados
        resultados = []
        
        for future in futures:
            resultado = future.result() #Bloqueamos las promesas y extraemos resultados
            
            resultados.extend(resultado)
        
        return resultados
    
    def shutdown(self): #Cerramos para que no se coma la memoria del ordenador
        self.client.close()
        self.cluster.close()