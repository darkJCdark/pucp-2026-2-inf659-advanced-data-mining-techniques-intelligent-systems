"""Descarga las fuentes de fuentes/fuentes_doi.csv a partir de su DOI, usando un navegador real.

Uso:
    pip install playwright pandas && playwright install chromium
    python descargar_fuentes.py

Funcionamiento
1. Abre Chromium con un perfil propio (carpeta .perfil_navegador) para conservar la sesión.
2. Para cada DOI navega a https://doi.org/<doi>; el DOI redirige al editor (IEEE Xplore, Springer, etc.).
3. Si el editor entrega un PDF con la sesión actual, lo guarda. Si no, la ventana queda abierta
   para que inicies sesión tú (acceso institucional PUCP); luego pulsa Enter en la consola y reintenta.
El script no pide ni guarda usuario ni contraseña. Solo descarga lo que tu acceso permite.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent / "fuentes"
PERFIL = Path(__file__).resolve().parent / ".perfil_navegador"


def es_pdf(resp):
    return resp.ok and "application/pdf" in (resp.headers.get("content-type") or "")


def intentar_descarga(ctx, page, doi):
    """Devuelve los bytes del PDF si el editor lo entrega con la sesión actual; si no, None."""
    url = page.url
    candidatos = []
    m = re.search(r"ieeexplore\.ieee\.org/(?:document|abstract/document)/(\d+)", url)
    if m:
        candidatos.append(f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={m.group(1)}&ref=")
    # metadato estándar que publican la mayoría de editores
    meta = page.locator('meta[name="citation_pdf_url"]').first
    if meta.count():
        candidatos.append(meta.get_attribute("content"))
    for c in candidatos:
        r = ctx.request.get(c)
        if es_pdf(r):
            return r.body()
    return None


def main():
    tabla = pd.read_csv(RAIZ / "fuentes_doi.csv")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(PERFIL), headless=False, accept_downloads=True)
        page = ctx.new_page()
        for _, f in tabla.iterrows():
            destino = RAIZ / f["archivo"]
            if destino.exists():
                print(f"[ya existe] {destino.name}")
                continue
            page.goto(f"https://doi.org/{f['doi']}", wait_until="domcontentloaded")
            pdf = intentar_descarga(ctx, page, f["doi"])
            if pdf is None:
                input(f"\n{f['doi']}: el editor no entregó el PDF. Inicia sesión institucional en la ventana "
                      f"y pulsa Enter para reintentar... ")
                pdf = intentar_descarga(ctx, page, f["doi"])
            if pdf is None:
                print(f"[sin PDF] {f['doi']}: descárgalo a mano desde {page.url}")
                continue
            destino.write_bytes(pdf)
            print(f"[ok] {destino.name} ({len(pdf) / 1e6:.2f} MB)")
        ctx.close()


if __name__ == "__main__":
    sys.exit(main())
