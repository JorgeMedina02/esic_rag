import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
import html
from typing import List, Dict

OUTPUT_PATH = "data/raw/productos_dia.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.dia.es/"
}

URLS_CATEGORIAS = {
    'Frescos': 'https://www.dia.es/frutas/c/L105',
    'Charcuteria': 'https://www.dia.es/charcuteria-y-quesos/c/L101',
    'Panaderia': 'https://www.dia.es/panaderia/c/L112',
    'Lacteos': 'https://www.dia.es/huevos-leche-y-mantequilla/c/L108',
    'Congelados': 'https://www.dia.es/congelados/c/L119',
    'Alimentacion': 'https://www.dia.es/arroz-pastas-y-legumbres/c/L106',
    'Platos_preparados': 'https://www.dia.es/platos-preparados/c/L103',
    'Bebidas': 'https://www.dia.es/bebidas/c/L107',
    'Desayunos_Dulces': 'https://www.dia.es/desayuno-y-dulces/c/L109'
}

def get_product_links(cat_url: str) -> List[str]:
    links = []
    try:
        resp = requests.get(cat_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200: return []
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split('?')[0]
            if "/p/" in href:
                full_url = "https://www.dia.es" + href if not href.startswith("http") else href
                if full_url not in links:
                    links.append(full_url)
    except Exception as e:
        print(f"Error obteniendo links: {e}")
    return links

def parse_nutrition_robust(soup) -> Dict[str, str]:
    """Extracción infalible buscando texto plano con expresiones regulares"""
    nutri = {}
    
    # Convertimos todo el HTML de la web en un solo texto gigante y en minúsculas
    texto = soup.get_text(separator=" ", strip=True).lower()
    
    # Patrones para cazar el nutriente y el número que le sigue justo después
    patrones = {
        "Grasas": r"grasas?\s*(?:totales)?\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Saturadas": r"saturadas?\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Hidratos de carbono": r"(?:hidratos de carbono|carbohidratos?)\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Azucares": r"az[uú]cares?\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Fibra alimentaria": r"fibras?\s*(?:alimentaria)?\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Proteinas": r"prote[ií]nas?\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr))",
        "Sal": r"(?:sal|sodio)\s*[:\-]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|mg))"
    }
    
    for clave, patron in patrones.items():
        match = re.search(patron, texto)
        if match:
            nutri[clave] = match.group(1).replace(",", ".")
            
    # La energía a veces es difícil, buscamos directamente el "kcal"
    match_kcal = re.search(r"(\d+(?:[.,]\d+)?\s*kcal)", texto)
    if match_kcal:
        nutri["Valor energetico"] = match_kcal.group(1).replace(",", ".")
        
    return nutri

def scrape_product_detail_debug(url: str, categoria: str) -> Dict:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        titulo_elem = soup.find("h1")
        titulo = html.unescape(titulo_elem.get_text(strip=True)) if titulo_elem else "Desconocido"
        
        peso_volumen = ""
        match_peso = re.search(r"(\d+(?:[.,]\d+)?\s*(?:g|kg|ml|l|cl|gr))", titulo, re.IGNORECASE)
        if match_peso:
            peso_volumen = match_peso.group(1).lower()

        precio_total = 0.0
        precio_por_cantidad = 0.0
        
        precio_elem = soup.select_one('p[class*="price"], span[class*="price"], .product-main-info__price')
        if precio_elem:
            precio_texto = precio_elem.get_text(strip=True)
            match = re.search(r"(\d+,\d+|\d+)", precio_texto)
            if match:
                precio_total = float(match.group(1).replace(",", "."))
                
        precio_unit_elem = soup.select_one('.price-per-unit, p[class*="unit"], span[class*="unit"]')
        if precio_unit_elem:
            unit_texto = precio_unit_elem.get_text(strip=True)
            match_unit = re.search(r"(\d+,\d+|\d+)", unit_texto)
            if match_unit:
                precio_por_cantidad = float(match_unit.group(1).replace(",", "."))

        nutricion = parse_nutrition_robust(soup)
        
        descripcion = ""
        desc_elem = soup.select_one('.product-description, p[class*="description"], div[class*="description"]')
        if desc_elem:
            descripcion = desc_elem.get_text(strip=True)

        print(f"  > {titulo[:30]}... | Precio: {precio_total} | Nutri: {len(nutricion)} macros encontrados")
        
        return {
            "url": url,
            "titulo": titulo,
            "valores_nutricionales_100_g": nutricion,
            "descripcion": descripcion,
            "categorias": [categoria.lower()],
            "precio_total": precio_total,
            "precio_por_cantidad": precio_por_cantidad,
            "peso_volumen": peso_volumen,
            "alergenos": [],
            "origen": "dia",
            "direccion_manufactura": []
        }
    except Exception as e:
        return {"url": url, "error": str(e)}

def main_acquisition_v3():
    os.makedirs("data/raw", exist_ok=True)
    all_products = []
    
    for cat_name, cat_url in URLS_CATEGORIAS.items():
        print(f"\n--- EXPLORANDO: {cat_name} ---")
        urls = get_product_links(cat_url)
        
        for p_url in urls[:50]: 
            data = scrape_product_detail_debug(p_url, cat_name)
            if "error" not in data:
                all_products.append(data)
            time.sleep(0.8) # Es vital mantener este tiempo para que no te bloqueen

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)
    
    print(f"\nProceso terminado. Total extraido: {len(all_products)} productos en '{OUTPUT_PATH}'")

if __name__ == "__main__":
    # Esto solo se ejecutará si corres acquisition.py directamente, 
    # pero no si lo importas desde main.py
    main_acquisition_v3()