# Vibe Quest S10 · Pregunta 3: SAE, monosemanticidad y steering

Pregunta: la polisemanticidad y los esfuerzos por lograr características monosemánticas con autoencoders; cómo dirigir modelos de lenguaje manipulando esas características podría transformar la auditoría, la alineación y la detección de sesgos, y qué riesgos éticos plantea.

## Mapa de carpetas

| Ruta | Contenido |
|---|---|
| `VibeQuest_S10_SAE_Monosemanticidad_Steering.ipynb` | Cuaderno interactivo (2 paneles con sliders/menús) para exponer en 5 minutos. 61 celdas (28 de código): una celda de texto antes de cada celda de código, secciones `N · Título`; librerías, parámetros, definición de funciones e invocaciones van en celdas separadas, y cada tabla, gráfico, impresión o archivo guardado está en su propia celda |
| `Guion_VibeQuest_5min.md` | Guion hablado, tabla de cifras y preguntas probables |
| `pipeline.md` | Historial, decisiones, resultados, límites y pendientes |
| `fuentes/` | Notas de lectura en palabras propias y `README.md` con enlaces y estado de lectura de cada fuente |
| `codigo/experimento_sae_steering.py` | Mismo experimento como script (entrena 3 modelos y 12 SAE, robustez de 5 semillas, exporta CSV y figuras) |
| `resultados/datos/` | 4 CSV generados por la ejecución en Colab |
| `resultados/capturas/` | Figuras de resultados (fig1 y fig2) |
| `resultados/interpretacion_resultados.md` | Lectura de cada resultado, con sus límites |

No hay dataset externo: los datos son sintéticos y se generan en el código (semillas fijas).

## Cómo ejecutarlo
Abrir el cuaderno en Google Colab (CPU basta) y ejecutar todas las celdas en orden: tarda unos 5 a 6 minutos. Los CSV y figuras se guardan en `resultados/` del entorno de Colab. Requiere `torch`, `numpy`, `pandas`, `matplotlib`, `ipywidgets`.

## Estado
- El código del cuaderno y el bloque principal del script se ejecutaron en Colab (T4; las redes corren en CPU); el archivo `.py` completo no se corrió de una sola vez como archivo y sus figuras `panelA/panelB` no se generaron. Las comprobaciones (`assert`) pasan y la tabla se reprodujo con los mismos valores en dos ejecuciones.
- Los 7 PDF de arXiv están en `fuentes/`. Bricken (2023) y Templeton (2024) son páginas web sin PDF oficial: URL en `fuentes/README.md`.
- Las figuras de `capturas/` están generadas desde los CSV; no son capturas de pantalla de los widgets.
