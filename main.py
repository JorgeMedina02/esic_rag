import os
import pandas as pd
from src.acquisition import obtener_productos
from src.preprocessing import procesar_datos
from src.rag import consultar

# -------------------------------------------------------------------
# 1. ORQUESTADOR PRINCIPAL DEL PIPELINE
# -------------------------------------------------------------------

def main():
    """Controla el flujo completo: Adquisicion -> Preprocesamiento -> RAG."""
    print("=== PIPELINE DE ASISTENTE NUTRICIONAL ===")
    
    # FASE 1: ADQUISICION DE DATOS
    respuesta = input("Deseas volver a descargar los datos desde la web? (s/n): ")
    
    if respuesta.lower() == 's':
        print("\nIniciando extraccion de datos (Web Scraping)...")
        obtener_productos()
    else:
        print("\nOmitiendo extraccion. Usando datos guardados.")

    # FASE 2: PREPROCESAMIENTO Y LIMPIEZA
    print("\nIniciando limpieza y normalizacion de datos...")
    df_procesado = procesar_datos()
    
    if df_procesado is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ruta_clean = os.path.join(BASE_DIR, "data", "clean", "productos_limpios.json")
        df_procesado = pd.read_json(ruta_clean, orient="records")

    print(f"Datos procesados correctamente: {len(df_procesado)} productos listos.")

    # FASE 3: MOTOR RAG Y BUSQUEDA VECTORIAL
    print("\nLevantando el motor de Inteligencia Artificial...")
    consultar(df_procesado)

if __name__ == "__main__":
    main()