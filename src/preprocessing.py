import pandas as pd
import json
import os
import re

# -------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES
# -------------------------------------------------------------------

def clean_numeric(value):
    """Extrae el valor numerico puro de textos mixtos."""
    if pd.isna(value) or value == "": return 0.0
    try:
        nums = re.findall(r"\d+\.?\d*", str(value).replace(",", "."))
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0

# -------------------------------------------------------------------
# 2. FUNCION PRINCIPAL DE PREPROCESAMIENTO
# -------------------------------------------------------------------

def procesar_datos():
    """Limpia los datos y crea metricas normalizadas."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(BASE_DIR, "data", "raw", "productos_dia.json")
    output_dir = os.path.join(BASE_DIR, "data", "clean")
    
    os.makedirs(output_dir, exist_ok=True)

    print("--- INICIANDO PREPROCESAMIENTO ---")
    
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    conteo_inicial = len(df)
    
    # -------------------------------------------------------------------
    # 3. EXTRACCION Y LIMPIEZA DE COLUMNAS
    # -------------------------------------------------------------------
    
    df['proteinas'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Proteinas', 0)) if isinstance(x, dict) else 0.0)
    df['carbohidratos'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Hidratos de carbono', 0)) if isinstance(x, dict) else 0.0)
    df['grasas'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Grasas', 0)) if isinstance(x, dict) else 0.0)
    df['calories'] = df['valores_nutricionales_100_g'].apply(lambda x: clean_numeric(x.get('Valor energetico', 0)) if isinstance(x, dict) else 0.0)
    
    # Calculo de rentabilidad: Priorizamos el precio por cantidad (kg/litro)
    df['precio'] = df.apply(lambda row: row['precio_por_cantidad'] if pd.notna(row.get('precio_por_cantidad')) and row.get('precio_por_cantidad', 0) > 0 else row.get('precio_total', 0), axis=1)
    
    df['texto_busqueda'] = df['titulo'].fillna("") + " " + df['categorias'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    
    # -------------------------------------------------------------------
    # 4. FILTRADO DE CALIDAD DE DATOS
    # -------------------------------------------------------------------
    
    df = df.drop_duplicates(subset=['titulo']).dropna(subset=['titulo'])
    df = df[(df['proteinas'] > 0) | (df['carbohidratos'] > 0) | (df['grasas'] > 0) | (df['calories'] > 0)]
    conteo_final = len(df)

    # -------------------------------------------------------------------
    # 5. NORMALIZACION MATEMATICA (0 a 1)
    # -------------------------------------------------------------------
    
    if not df.empty and df['precio'].max() != df['precio'].min():
        df['norm_precio'] = 1 - (df['precio'] - df['precio'].min()) / (df['precio'].max() - df['precio'].min())
    else:
        df['norm_precio'] = 1.0

    if not df.empty and df['proteinas'].max() != df['proteinas'].min():
        df['norm_nutri'] = (df['proteinas'] - df['proteinas'].min()) / (df['proteinas'].max() - df['proteinas'].min())
    else:
        df['norm_nutri'] = 0.5
        
    df['score_nutricional'] = df['norm_nutri'] * 100
    
    # -------------------------------------------------------------------
    # 6. EXPORTACION
    # -------------------------------------------------------------------
    
    output_path = os.path.join(output_dir, "productos_limpios.json")
    df.to_json(output_path, orient="records", force_ascii=False)
    print(f"Proceso finalizado. {conteo_final} productos guardados en: {output_path}")
    
    return df

if __name__ == "__main__":
    procesar_datos()