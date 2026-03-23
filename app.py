import streamlit as st
import pandas as pd
import os
from src.rag import crear_indice, buscar_y_responder

# -------------------------------------------------------------------
# 1. CONFIGURACION DE LA PAGINA
# -------------------------------------------------------------------

st.set_page_config(page_title="Asistente Nutricional RAG", layout="centered")

st.title("Asistente Nutricional Inteligente")
st.write("Busca productos del supermercado filtrados por semantica, precio y salud.")

# -------------------------------------------------------------------
# 2. SISTEMA DE CACHE Y OPTIMIZACION
# -------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR, "data", "clean", "productos_limpios.json")
    
    if os.path.exists(ruta):
        return pd.read_json(ruta, orient="records")
    return None

@st.cache_resource
def iniciar_motor(_df):
    return crear_indice(_df)

# -------------------------------------------------------------------
# 3. INICIALIZACION DEL ENTORNO Y FILTROS LATERALES
# -------------------------------------------------------------------

df_limpio = cargar_datos()

if df_limpio is None:
    st.error("No se encontraron los datos limpios. Ejecuta 'python main.py' en tu terminal primero.")
else:
    st.sidebar.title("Informacion del Sistema")
    st.sidebar.write(f"Productos indexados: {len(df_limpio)}")
    st.sidebar.write("Motor: FAISS + Hibrido")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Estrictos")
    
    max_precio_df = float(df_limpio['precio_total'].max())
    precio_maximo = st.sidebar.slider("Precio Maximo (Euros)", 0.0, max_precio_df, max_precio_df)
    
    categorias_crudas = df_limpio['categorias'].dropna().tolist()
    lista_categorias = []
    for cat in categorias_crudas:
        if isinstance(cat, list):
            lista_categorias.extend(cat)
        else:
            lista_categorias.append(str(cat))
            
    categorias_disponibles = ["Todas"] + sorted(list(set([c.capitalize() for c in lista_categorias])))
    categoria_seleccionada = st.sidebar.selectbox("Categoria Especifica", categorias_disponibles)

    # -------------------------------------------------------------------
    # 4. INTERFAZ PRINCIPAL DE BUSQUEDA
    # -------------------------------------------------------------------

    with st.spinner("Iniciando motor de Inteligencia Artificial..."):
        indice = iniciar_motor(df_limpio)

    st.markdown("---")
    
    consulta = st.text_input(
        "Que estas buscando hoy?", 
        placeholder="Ej: top 3 de snacks con mucha proteina y poca grasa"
    )

    if st.button("Buscar"):
        if consulta:
            with st.spinner("Analizando opciones y aplicando filtros..."):
                respuesta = buscar_y_responder(
                    consulta, 
                    df_limpio, 
                    indice, 
                    precio_max=precio_maximo, 
                    categoria=categoria_seleccionada
                )
                st.success("Busqueda completada con exito.")
                st.markdown(respuesta)
        else:
            st.warning("Por favor, escribe una consulta antes de buscar.")