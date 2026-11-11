# Vibe Quest S13 · Transformer sin Positional Encoding (pregunta 2)

**Pregunta:** si quitamos el Positional Encoding, "El gato persiguió al perro" y "El perro persiguió al gato" ¿producen la misma salida? Demostrarlo con código, comparar los embeddings de salida con y sin PE y visualizar por qué se pierde el orden.

**Respuesta corta:** sin PE, la salida de cada palabra no depende de su posición (error ~1e-15); permutar la frase solo permuta las filas. Con PE, el orden queda codificado y las salidas cambian.

## Cómo se usa
1. Sube `vibe_quest_s13.ipynb` a Google Colab y ejecuta todas las celdas (CPU alcanza). La sección 9 tiene el panel con sliders.
2. La última celda descarga `resultados_s13.zip` con las tablas.
3. Para las fuentes: `pip install playwright pandas && playwright install chromium` y `python codigo/descargar_fuentes.py`.

## Estructura
```
Vibe Quest/
  README.md                 este archivo
  pipeline.md               flujo, pasos, parámetros, resultados y límites
  guion_5min.md             guion de presentación con tiempos y preguntas posibles
  vibe_quest_s13.ipynb      notebook (texto + código separado, sliders)
  codigo/
    atencion_pe.py          funciones en PyTorch (las del notebook)
    referencia_numpy.py     misma matemática en NumPy
    ejecutar_referencia.py  corre los experimentos de referencia y guarda CSV y figura
    descargar_fuentes.py    baja los PDF desde fuentes_doi.csv
  fuentes/                  fuentes_doi.csv, README.md (qué respalda cada una) y los PDF
  resultados/
    datos/                  CSV de la referencia NumPy
    capturas/               figura de las matrices de atención
    google colab/           CSV de la ejecución en Colab
    interpretacion_resultados.md
```

## Fuentes (Scopus o IEEE, 2022-2026)
Zhang et al. (2024, ICASSP), Guo (2024, ICEIB), Banđur et al. (2026, TPAMI), Hayakawa y Sato (2025, ACL), Zhao et al. (2024, EMNLP Findings) y Wang et al. (2024, TPAMI). Detalle y DOI en `fuentes/README.md`.

## Límites
Pesos aleatorios sin entrenar, codificador bidireccional, un tipo de PE; los resúmenes de las fuentes están leídos, sus textos completos no.
