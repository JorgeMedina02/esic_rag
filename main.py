import pandas as pd
from src.rag import consultar

def main():
    print("=== INICIANDO ASISTENTE NUTRICIONAL ===")
    
    ruta_limpios = "data/clean/productos_limpios.json"
    
    try:
        # Cargamos exclusivamente los datos que ya están listos
        df_limpio = pd.read_json(ruta_limpios, orient="records")
        
        # Lanzamos directamente el motor de búsqueda vectorial
        consultar(df_limpio)
        
    except FileNotFoundError:
        print(f"[ERROR] No se encuentra el archivo en: {ruta_limpios}")
        print("Asegúrate de no haber borrado la carpeta 'data/clean'.")

if __name__ == "__main__":
    main()