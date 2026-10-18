# Pipeline: cómo se hizo `Proyecto_Deforestacion_Que_se_quiere_hacer.md`

Fecha: 18 de octubre de 2026. Pedido: ponerse al día, revisar todo lo avanzado de la TA sobre deforestación y explicar en un archivo qué se quiere hacer.

## Qué hice

1. **Contexto.** Leí la memoria del proyecto (`inf659-deep-learning.md`, `preferences.md`) y la lista de documentos del Project "Data Mining". Hice una búsqueda en el Project sobre deforestación: solo devolvió la guía de estudio, la preparación del parcial, la teoría y el Vibe Quest; ninguno trata el proyecto.
2. **Inventario de la carpeta.** Listé recursivamente la carpeta del curso del portátil `laptop-lot53lo9` y el historial de git (últimos 8 commits).
3. **Lectura de los entregables.** Convertí con `pandoc` los cinco `.docx` (Propuesta, abstract A, abstract B, Datasets, chat con Cursor) a texto. Extraje el texto de los dos pósteres `.pptx` con `python-pptx`. De los dos cuadernos `.ipynb` leí las celdas de texto y el final de las salidas guardadas, sin ejecutarlos.
4. **Verificación de afirmaciones.** Contrasté cuatro afirmaciones del chat con Cursor contra sus fuentes: umbral de 30 % del Global Forest Review (página del GFR), área mínima de PRODES de 6,25 ha ([arXiv 2211.09806](https://arxiv.org/abs/2211.09806)), umbral de 80 % de Gasparini et al. 2019 ([MDPI](https://www.mdpi.com/1424-8220/19/22/5020); la versión de PMC pedía reCAPTCHA) y las 100 imágenes y 84 % de McCallum et al. 2023 ([MDPI](https://www.mdpi.com/2072-4292/15/21/5204)).
5. **Cálculo propio.** Un parche de 224 px a 10 m mide 2,24 km (5,02 km², ~502 ha). Una ventana de 55 km contiene (55/2,24)² ≈ 603 parches. Esas cifras se usan en los riesgos (b).
6. **Redacción.** Escribí el archivo y este pipeline en español.

## Qué no hice

- No ejecuté ningún cuaderno ni descargué datos. Las cifras de DFDC salen de las salidas guardadas en los cuadernos.
- No vi el video `2026-10-15 09-08-04.mkv`.
- No verifiqué las coordenadas de las ventanas sobre un mapa ni el conteo de parches en Earth Engine.
- No encontré el abstract final con las reglas (`Abstract_FrozenDINOv2_ForestLoss_Para_Peru.docx`).
- No sé si E1 se entregó, si se registró el póster ni si el profesor aprobó el tema.

## Fuentes usadas

- Carpeta del curso: `4. Tarea Académica/Entregables-…/Entregables` (Propuesta.docx, WAIMLAp_2026_Abstract.docx, [Cursor]Abstract_v1.docx, Chat con Cursor - 14.10.docx, Datasets.docx, E1_*.ipynb, pósteres).
- Memoria y Project "Data Mining".
- Las cuatro fuentes web del punto 4.

## Cómo reproducirlo

1. Listar la carpeta y leer los `.docx` con `pandoc archivo.docx -t plain --wrap=none`.
2. Leer los `.pptx` con `python-pptx` recorriendo `shape.text_frame`.
3. Leer los `.ipynb` como JSON y filtrar las celdas `markdown` y las salidas de texto.
4. Repetir las cuatro verificaciones de la sección anterior.
