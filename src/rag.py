import faiss
import numpy as np
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# 1. CONFIGURACION E INICIALIZACION DE IA
# -------------------------------------------------------------------

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# -------------------------------------------------------------------

def limpiar_numero(texto):
    """Extrae el valor numerico puro de textos mixtos."""
    try:
        nums = re.findall(r"\d+\.?\d*", str(texto).replace(",", "."))
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0

# -------------------------------------------------------------------
# 3. CREACION DEL MOTOR VECTORIAL (FAISS)
# -------------------------------------------------------------------

def crear_indice(df):
    """Transforma el texto de busqueda en vectores y crea el indice."""
    print("[RAG] Generando embeddings e indice FAISS...")
    
    textos_enriquecidos = []
    for _, r in df.iterrows():
        texto_base = str(r.get("texto_busqueda", ""))
        texto_extra = f" {r['proteinas']} proteinas {r['calories']} calorias {r['grasas']} grasas {r['carbohidratos']} carbohidratos"
        textos_enriquecidos.append(texto_base + texto_extra)

    embeddings = embedder.encode(textos_enriquecidos, show_progress_bar=False)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))
    
    return index

# -------------------------------------------------------------------
# 4. LOGICA DE BUSQUEDA, FILTROS Y RE-RANKING HIBRIDO
# -------------------------------------------------------------------

def buscar_y_responder(consulta, df, index, precio_max=None, categoria=None):
    """Busca, aplica filtros de usuario, hibrida el resultado y formatea la salida."""
    
    # Paso 1: Busqueda Vectorial Semantica
    vec_query = embedder.encode([consulta]).astype("float32")
    dist, indices = index.search(vec_query, 100)
    candidatos = df.iloc[indices[0]].copy()

    # Paso 2: Aplicacion de Filtros Estrictos
    if precio_max is not None:
        candidatos = candidatos[candidatos['precio_total'] <= precio_max]
    
    if categoria and categoria != "Todas":
        candidatos = candidatos[candidatos['categorias'].apply(lambda cats: categoria.lower() in [c.lower() for c in cats] if isinstance(cats, list) else categoria.lower() in str(cats).lower())]

    if candidatos.empty:
        return "No he encontrado productos que coincidan con esos filtros estrictos."

    # Paso 3: Re-ranking Matematico
    max_dist = dist[0].max() if dist[0].max() > 0 else 1
    candidatos["norm_dist"] = 1 - (dist[0] / max_dist)

    candidatos["rank_final"] = (
        candidatos["norm_dist"] * 0.6
        + candidatos["norm_nutri"] * 0.2
        + candidatos["norm_precio"] * 0.2
    )

    # Paso 4: Booster Lexico con Diccionario de Sinonimos Comerciales
    diccionario_sinonimos = {
        "papas": "patatas", "jugo": "zumo", "chucherias": "snacks", 
        "chuches": "snacks", "refresco": "cola", "bizcocho": "bolleria",
        "cerdo": "porcino", "res": "vacuno", "pollo": "ave", "panes": "pan"
    }
    
    palabras_vacias = {'mas', 'menos', 'los', 'las', 'con', 'sin', 'para', 'del', 'que', 'muy', 'top', 'mejores', 'sanos', 'sano', 'saludable', 'saludables', 'barato', 'baratos'}
    palabras_clave_crudas = [p.lower() for p in re.findall(r'\b[a-záéíóúñ]{3,}\b', consulta.lower()) if p not in palabras_vacias]
    
    palabras_clave = []
    for p in palabras_clave_crudas:
        singular = p[:-2] if p.endswith('es') else (p[:-1] if p.endswith('s') else p)
        termino_final = diccionario_sinonimos.get(singular, singular)
        palabras_clave.append(termino_final)

    def calcular_boost_lexico(row):
        texto_item = str(row['titulo']).lower() + " " + str(row.get('categorias', '')).lower()
        boost = 0
        for palabra in palabras_clave:
            if palabra in texto_item:
                boost += 2.0
            elif len(palabra) > 4 and palabra in texto_item:
                boost += 1.0
        return boost

    candidatos['boost_lexico'] = candidatos.apply(calcular_boost_lexico, axis=1)
    candidatos['rank_final'] = candidatos['rank_final'] + candidatos['boost_lexico']

    # Paso 5: Deteccion de Intenciones y Formateo
    match_top = re.search(r'\btop\s*(\d+)|(\d+)\s*mejores', consulta, re.IGNORECASE)
    top_n = min(max(int(match_top.group(1) or match_top.group(2)), 1), 15) if match_top else 5

    focuses = []
    if re.search(r'salud|sano|sana|fitness|fit', consulta, re.IGNORECASE): focuses.append('salud')
    if re.search(r'prote[íi]na', consulta, re.IGNORECASE): focuses.append('proteinas')
    if re.search(r'sal\b|sales\b|sodio', consulta, re.IGNORECASE): focuses.append('sal')
    if re.search(r'grasa|l[íi]pido', consulta, re.IGNORECASE): focuses.append('grasas')
    if re.search(r'az[úu]car|dulce', consulta, re.IGNORECASE): focuses.append('azucares')
    if re.search(r'fibra', consulta, re.IGNORECASE): focuses.append('fibra')
    if re.search(r'carbohidrato|hidrato', consulta, re.IGNORECASE): focuses.append('carbohidratos')
    if re.search(r'calor[íi]a', consulta, re.IGNORECASE): focuses.append('calories')
    if re.search(r'precio|barato|caro|euro', consulta, re.IGNORECASE): focuses.append('precio')

    mejores = candidatos.sort_values("rank_final", ascending=False).head(top_n)

    primary_focus = focuses[0] if focuses else None
    if primary_focus:
        valores_focus = []
        for _, r in mejores.iterrows():
            nutri = r['valores_nutricionales_100_g'] if isinstance(r['valores_nutricionales_100_g'], dict) else {}
            if primary_focus == 'proteinas': val = r['proteinas']
            elif primary_focus == 'salud': val = r['score_nutricional']
            elif primary_focus == 'precio': val = r['precio_total']
            elif primary_focus == 'grasas': val = r['grasas']
            elif primary_focus == 'calories': val = r['calories']
            elif primary_focus == 'carbohidratos': val = r['carbohidratos']
            elif primary_focus == 'sal': val = limpiar_numero(nutri.get('Sal', '0'))
            elif primary_focus == 'azucares': val = limpiar_numero(nutri.get('Azucares', '0'))
            elif primary_focus == 'fibra': val = limpiar_numero(nutri.get('Fibra alimentaria', '0'))
            else: val = 0.0
            valores_focus.append(val)
        
        mejores['val_orden'] = valores_focus
        ascendente = True if primary_focus in ['precio', 'grasas', 'calories', 'carbohidratos', 'azucares', 'sal'] else False
        mejores = mejores.sort_values('val_orden', ascending=ascendente)

    lineas_contexto = []
    for _, r in mejores.iterrows():
        nutri = r['valores_nutricionales_100_g'] if isinstance(r['valores_nutricionales_100_g'], dict) else {}
        
        if focuses and 'salud' not in focuses:
            detalles_lista = []
            for f in focuses:
                if f == 'proteinas': detalles_lista.append(f"Proteinas: {r['proteinas']}g")
                elif f == 'sal': detalles_lista.append(f"Sal: {nutri.get('Sal', '0g')}")
                elif f == 'grasas': detalles_lista.append(f"Grasas: {r['grasas']}g (Sat: {nutri.get('Saturadas', '0g')})")
                elif f == 'azucares': detalles_lista.append(f"Azucares: {nutri.get('Azucares', '0g')}")
                elif f == 'fibra': detalles_lista.append(f"Fibra: {nutri.get('Fibra alimentaria', '0g')}")
                elif f == 'carbohidratos': detalles_lista.append(f"Carbohidratos: {r['carbohidratos']}g")
                elif f == 'calories': detalles_lista.append(f"Calorias: {r['calories']} kcal")
                elif f == 'precio': detalles_lista.append(f"Precio: {r['precio_total']} euros")
            detalle = " | ".join(detalles_lista)
        else:
            detalle = (
                f"Proteinas: {r['proteinas']}g | Grasas: {r['grasas']}g | "
                f"Carbos: {r['carbohidratos']}g (Azucares: {nutri.get('Azucares', '0g')}) | "
                f"Fibra: {nutri.get('Fibra alimentaria', '0g')} | Sal: {nutri.get('Sal', '0g')}"
            )

        linea = f"- **{r['titulo']}** | Precio: {r['precio_total']} euros | Salud: {int(r['score_nutricional'])}/100\n  > {detalle}\n"
        lineas_contexto.append(linea)

    contexto = "\n".join(lineas_contexto)
    return f"**Asistente Nutricional:** Para '{consulta}', he encontrado estas opciones:\n\n{contexto}"

# -------------------------------------------------------------------
# 5. BUCLE DE EJECUCION POR CONSOLA
# -------------------------------------------------------------------

def consultar(df):
    id = crear_indice(df)
    while True:
        consulta = input("\nIntroduce tu consulta (o 'salir' para terminar): ")
        if consulta.lower() == "salir":
            break
        respuesta = buscar_y_responder(consulta, df, id)
        print(respuesta)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_prueba = os.path.join(BASE_DIR, "data", "clean", "productos_limpios.json")
    if os.path.exists(ruta_prueba):
        df_prueba = pd.read_json(ruta_prueba, orient="records")
        consultar(df_prueba)
    else:
        print(f"No se encontraron datos en {ruta_prueba}")