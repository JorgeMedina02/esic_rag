import os
import pandas as pd
from src.acquisition import main_acquisition_v3
from src.preprocessing import procesar_datos
from src.rag import consultar

def main():
    print("=== INICIANDO SISTEMA RAG DE SUPERMERCADO ===")
    
    # Define paths based on your project structure
    ruta_raw = "data/raw/productos_dia.json"
    ruta_limpios = "data/clean/productos_limpios.json"
    
    # 1 & 2: Acquisition and Preprocessing
    if os.path.exists(ruta_limpios):
        print(f"\n[INFO] Datos limpios encontrados en: {ruta_limpios}")
        print("Cargando datos directamente para el motor de búsqueda...")
        # Load the existing JSON into a DataFrame
        df_limpio = pd.read_json(ruta_limpios, orient="records")
    else:
        print("\n[INFO] No se encontraron datos limpios listos.")
        
        # Run Acquisition if raw data doesn't exist either
        if not os.path.exists(ruta_raw):
            print("1. Iniciando fase de adquisición...")
            main_acquisition_v3()
        else:
             print(f"[INFO] Datos brutos encontrados en: {ruta_raw}. Saltando adquisición.")

        print("\n2. Iniciando fase de preprocesamiento...")
        # procesar_datos() must return the DataFrame. If it doesn't, we load it manually to be safe.
        procesar_datos() 
        df_limpio = pd.read_json(ruta_limpios, orient="records")

    # 3: RAG Execution
    if df_limpio is not None and not df_limpio.empty:
        print("\n3. Ejecutando RAG...")
        consultar(df_limpio)
    else:
        print("\n[ERROR] El DataFrame está vacío o es None. Revisa el archivo clean/productos_limpios.json.")

if __name__ == "__main__":
    main()