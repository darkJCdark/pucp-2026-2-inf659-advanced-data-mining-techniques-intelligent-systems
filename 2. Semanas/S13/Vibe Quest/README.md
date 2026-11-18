# Vibe Quest S13 · Transformer sin Positional Encoding (pregunta 2)

**Pregunta:** ¿qué le pasaría al Transformer si eliminamos el Positional Encoding y le damos "El gato persiguió al perro" y "El perro persiguió al gato"? Demostrarlo con código comparando los embeddings de salida con y sin PE y visualizando por qué se pierde el orden.

**Respuesta:** sin PE, cada palabra sale con el mismo vector en las dos frases (error de equivarianza ~1e-15 con pesos aleatorios y 6e-7 con modelos entrenados; coseno de "gato" = 1.0). Solo cambia la fila en que aparece, así que el modelo no distingue quién persigue a quién. Con PE las salidas cambian. La máscara causal es la excepción: da información de posición aunque no haya PE.

## Cómo se usa
1. Abre `vibe_quest_s13_transformer_sin_pe.ipynb` en Google Colab con GPU (Entorno de ejecución → Cambiar tipo → T4) y ejecuta todo (unos 3 minutos). El dataset se descarga solo.
2. Paneles con sliders: 2.10 (Parte A), 3.14, 3.16 y 3.17 (Parte B).
3. La última celda descarga `resultados_s13.zip` con todas las tablas.

## Estructura
```
Vibe Quest/
  README.md                 este archivo
  pipeline.md               flujo, pasos, decisiones, resultados y límites
  guion_5min.md             guion de presentación con tiempos y preguntas posibles
  vibe_quest_s13_transformer_sin_pe.ipynb   notebook (85 celdas, secciones con # y ##)
  descargar_fuentes.bat     atajo de Windows para codigo/descargar_fuentes.py
  codigo/
    atencion_pe.py          Parte A en PyTorch
    referencia_numpy.py     Parte A en NumPy (validación independiente)
    ejecutar_referencia.py  corre la referencia NumPy y guarda CSV y figuras
    parte_b_tatoeba.py      Parte B: datos, clasificador y entrenamiento
    descargar_fuentes.py    baja los PDF desde fuentes_doi.csv
  fuentes/                  6 PDF, fuentes_doi.csv y README.md
  resultados/
    datos/                  CSV de la referencia NumPy (Parte A)
    capturas/               figuras de la Parte A
    google colab/           CSV de la ejecución en Colab (Partes A y B)
    interpretacion_resultados.md
```

## Datos
Tatoeba español-inglés (tatoeba.org vía manythings.org/anki, CC-BY 2.0), copia de TensorFlow: `https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip`.

## Fuentes (Scopus o IEEE, 2022-2026)
Zhang et al. (2024, ICASSP), Guo (2024, ICEIB), Banđur et al. (2026, TPAMI), Hayakawa y Sato (2025, ACL), Zhao et al. (2024, EMNLP Findings) y Wang et al. (2024, TPAMI). Detalle en `fuentes/README.md`.
