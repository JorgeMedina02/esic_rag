import pandas as pd
import json
import os
import re

def clean_numeric(value):
    """Extrae el número de strings como '6.7 gr' o '537 kcal'."""
    if pd.isna(value) or value == "": return 0.0
    try:
        # Busca el primer numero que encuentre en el texto
        nums = re.findall(r"\d+\.?\d*", str(value).replace(",", "."))
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0

def procesar_datos():
    input_file = "data/raw/productos_dia.json"
    output_dir = "data/clean"
    os.makedirs(output_dir, exist_ok=True)

    print("--- INICIANDO PREPROCESAMIENTO ---")
    print(f"Leyendo archivo: {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    conteo_inicial = len(df)
    
    print("Calculando transformaciones nutricionales y normalizando scores...")

    # Transformaciones nutricionales usando la funcion clean_numeric
    df['proteinas'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Proteinas', 0)) if isinstance(x, dict) else 0.0)
    df['carbohidratos'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Hidratos de carbono', 0)) if isinstance(x, dict) else 0.0)
    df['grasas'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Grasas', 0)) if isinstance(x, dict) else 0.0)
    df['calories'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Valor energetico', 0)) if isinstance(x, dict) else 0.0)
    
    df['precio'] = df['precio_total']
    
    # Aseguramos que titulo y categorias sean texto antes de concatenar
    df['texto_busqueda'] = df['titulo'].fillna("") + " " + df['categorias'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    
    # Limpieza 1: Duplicados y nulos por título
    print("Eliminando duplicados...")
    df = df.drop_duplicates(subset=['titulo']).dropna(subset=['titulo'])
    conteo_sin_duplicados = len(df)
    
    # Limpieza 2: Eliminar productos sin valor nutricional
    # Conservamos solo aquellos que tengan algun valor mayor a 0 en sus macros o calorias
    print("Eliminando productos sin informacion nutricional...")
    df = df[(df['proteinas'] > 0) | (df['carbohidratos'] > 0) | (df['grasas'] > 0) | (df['calories'] > 0)]
    conteo_final = len(df)

    # Normalización (0-1) para el ranking del RAG
    # Lo hacemos despues de limpiar para que los max/min sean sobre los datos reales finales
    if not df.empty and df['precio'].max() != df['precio'].min():
        df['norm_precio'] = 1 - (df['precio'] - df['precio'].min()) / (df['precio'].max() - df['precio'].min())
    else:
        df['norm_precio'] = 1.0

    if not df.empty and df['proteinas'].max() != df['proteinas'].min():
        df['norm_nutri'] = (df['proteinas'] - df['proteinas'].min()) / (df['proteinas'].max() - df['proteinas'].min())
    else:
        df['norm_nutri'] = 0.5
        
    df['score_nutricional'] = df['norm_nutri'] * 100
    
    print("\n--- AUDITORIA DE DATOS ---")
    print(f"1. Productos cargados desde RAW: {conteo_inicial}")
    print(f"2. Productos tras eliminar duplicados: {conteo_sin_duplicados}")
    print(f"3. Productos tras eliminar sin valor nutricional: {conteo_final}")
    print(f"4. Cumple requisito > 200?: {'SI' if conteo_final >= 200 else 'NO'}")
    print("--------------------------\n")
    
    output_path = f"{output_dir}/productos_limpios.json"
    df.to_json(output_path, orient="records", force_ascii=False)
    print(f"Proceso finalizado. Archivo guardado en: {output_path}")
    
    return df

if __name__ == "__main__":
    procesar_datos()