import faiss
import numpy as np
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer

# Inicializamos el modelo de lenguaje de forma global
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def limpiar_numero(texto):
    """Extrae el valor numérico puro de textos como '1.5 g' o '0.2 gr' para poder ordenar."""
    try:
        nums = re.findall(r"\d+\.?\d*", str(texto).replace(",", "."))
        return float(nums[0]) if nums else 0.0
    except:
        return 0.0

def crear_indice(df):
    """Transforma el texto de búsqueda en vectores y crea el índice FAISS."""
    print("[RAG] Generando embeddings e índice FAISS...")
    
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

def buscar_y_responder(consulta, df, index):
    """Busca, filtra, re-ordena dinámicamente y formatea la salida."""
    # 1. Búsqueda Vectorial Semántica
    vec_query = embedder.encode([consulta]).astype("float32")
    dist, indices = index.search(vec_query, 15)

    candidatos = df.iloc[indices[0]].copy()

    # 2. Re-ranking matemático base
    max_dist = dist[0].max() if dist[0].max() > 0 else 1
    candidatos["norm_dist"] = 1 - (dist[0] / max_dist)

    candidatos["rank_final"] = (
        candidatos["norm_dist"] * 0.6
        + candidatos["norm_nutri"] * 0.2
        + candidatos["norm_precio"] * 0.2
    )

    # 3. Detección dinámica de Top N
    match_top = re.search(r'\btop\s*(\d+)|(\d+)\s*mejores', consulta, re.IGNORECASE)
    top_n = min(max(int(match_top.group(1) or match_top.group(2)), 1), 15) if match_top else 5

    # 4. Detección MÚLTIPLE de Foco de Nutrientes
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

    # Filtramos a los N mejores
    mejores = candidatos.sort_values("rank_final", ascending=False).head(top_n)

    # 5. Re-ordenación específica según el foco PRINCIPAL (el primero que se detecte)
    primary_focus = focuses[0] if focuses else None

    if primary_focus:
        valores_focus = []
        for _, r in mejores.iterrows():
            nutri = r['valores_nutricionales_100_g'] if isinstance(r['valores_nutricionales_100_g'], dict) else {}
            if primary_focus == 'proteinas': val = r['proteinas']
            elif primary_focus == 'salud': val = r['score_nutricional']
            elif primary_focus == 'precio': val = r['precio']
            elif primary_focus == 'grasas': val = r['grasas']
            elif primary_focus == 'calories': val = r['calories']
            elif primary_focus == 'carbohidratos': val = r['carbohidratos']
            elif primary_focus == 'sal': val = limpiar_numero(nutri.get('Sal', '0'))
            elif primary_focus == 'azucares': val = limpiar_numero(nutri.get('Azucares', '0'))
            elif primary_focus == 'fibra': val = limpiar_numero(nutri.get('Fibra alimentaria', '0'))
            else: val = 0.0
            valores_focus.append(val)
        
        mejores['val_orden'] = valores_focus
        
        # Orden ascendente (menor a mayor) para cosas que queremos reducir. Descendente para lo que queremos maximizar.
        ascendente = True if primary_focus in ['precio', 'grasas', 'calories', 'carbohidratos', 'azucares', 'sal'] else False
        mejores = mejores.sort_values('val_orden', ascending=ascendente)

    # 6. Formateo limpio y dinámico de la respuesta
    lineas_contexto = []
    for _, r in mejores.iterrows():
        nutri = r['valores_nutricionales_100_g'] if isinstance(r['valores_nutricionales_100_g'], dict) else {}
        
        # Si hay focos específicos (y no es solo "salud"), mostramos solo esos macros
        if focuses and 'salud' not in focuses:
            detalles_lista = []
            for f in focuses:
                if f == 'proteinas': detalles_lista.append(f"Proteínas: {r['proteinas']}g")
                elif f == 'sal': detalles_lista.append(f"Sal: {nutri.get('Sal', '0g')}")
                elif f == 'grasas': detalles_lista.append(f"Grasas: {r['grasas']}g (Sat: {nutri.get('Saturadas', '0g')})")
                elif f == 'azucares': detalles_lista.append(f"Azúcares: {nutri.get('Azucares', '0g')}")
                elif f == 'fibra': detalles_lista.append(f"Fibra: {nutri.get('Fibra alimentaria', '0g')}")
                elif f == 'carbohidratos': detalles_lista.append(f"Carbohidratos: {r['carbohidratos']}g")
                elif f == 'calories': detalles_lista.append(f"Calorías: {r['calories']} kcal")
                elif f == 'precio': detalles_lista.append(f"Precio: {r['precio']}€")
            detalle = " | ".join(detalles_lista)
        else:
            # Vista general si piden "salud" o no especifican ningún nutriente
            detalle = (
                f"Proteínas: {r['proteinas']}g | Grasas: {r['grasas']}g | "
                f"Carbos: {r['carbohidratos']}g (Azúcares: {nutri.get('Azucares', '0g')}) | "
                f"Fibra: {nutri.get('Fibra alimentaria', '0g')} | Sal: {nutri.get('Sal', '0g')}"
            )

        linea = f"- **{r['titulo']}** | Precio: {r['precio']}€ | Salud: {int(r['score_nutricional'])}/100\n  > {detalle}\n"
        lineas_contexto.append(linea)

    contexto = "\n".join(lineas_contexto)
    return f"**Asistente Nutricional:** Para '{consulta}', he encontrado estas opciones:\n\n{contexto}"

def consultar(df):
    id = crear_indice(df)
    while True:
        consulta = input("\nIntroduce tu consulta (o 'salir' para terminar): ")
        if consulta.lower() == "salir":
            print("Hasta luego.")
            break
        respuesta = buscar_y_responder(consulta, df, id)
        print(respuesta)

if __name__ == "__main__":
    ruta_prueba = "data/clean/productos_limpios.json"
    if os.path.exists(ruta_prueba):
        print("Iniciando prueba aislada del motor RAG...")
        df_prueba = pd.read_json(ruta_prueba, orient="records")
        consultar(df_prueba)
    else:
        print(f"Error: No se encuentran los datos en {ruta_prueba}.")