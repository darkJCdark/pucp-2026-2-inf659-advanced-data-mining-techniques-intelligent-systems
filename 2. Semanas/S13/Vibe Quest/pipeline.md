# pipeline.md · Vibe Quest S13 (pregunta 2: Transformer sin Positional Encoding)

## Objetivo
Responder: *¿qué le pasaría al Transformer si eliminamos el PE y le damos "El gato persiguió al perro" y "El perro persiguió al gato"? Demostrarlo con código comparando los embeddings de salida con y sin PE y visualizando por qué se pierde el orden.*

## Flujo (entrada → modelos → salida)

```
PARTE A · estructural (float64)                        PARTE B · entrenada con Tatoeba (float32, GPU)
FRASE_A / FRASE_B + 2 frases extra                     spa-eng.zip (118 964 pares) → oraciones únicas de 4-10 palabras
   │ embeddings y pesos aleatorios (semilla)              │ tarea ORDEN: 20 000 oraciones, 50 % con 2 palabras intercambiadas
   ▼                                                      │ tarea IDIOMA: 10 000 es + 10 000 en
codificador mínimo: self-attention + Add & Norm + FFN     ▼ 80/10/10, vocabulario del train, <BOS> al inicio
   │ con / sin PE sinusoidal                           Clasificador: emb + PE opcional → TransformerEncoder (d=64, 2 capas,
   ▼                                                     4 cabezas, bidireccional o causal) → promedio → lineal
error de equivarianza, coseno de "gato",                  │ 2 tareas × 2 atenciones × con/sin PE × 3 semillas = 24 entrenamientos
atención, PCA, barrido, 100 permutaciones,                ▼
comparación con NumPy                                  exactitud en prueba, curvas, ablación del PE (escala 0-1),
                                                       la consigna con los modelos entrenados (3.18), chequeos
```

## Historial de pasos
1. Leí la Clase 12 (láminas y transcripción): self-attention, PE sinusoidal (lámina 120), Add & Norm (lámina 122). Elegí la pregunta 2 porque la clase ya explica su teoría y se puede demostrar sin entrenar.
2. Fuentes Scopus/IEEE 2022-2026: 5 citadas en `fuentes/` (DeepNet quedó descartada porque trata Add & Norm, no el PE), descargadas con acceso institucional PUCP. Leí sus textos con `pdftotext`; las páginas citadas están en `fuentes/README.md`.
3. Parte A: referencia NumPy (`codigo/referencia_numpy.py`) y versión PyTorch; coinciden a 1e-9. Validada en Colab.
4. Metodología de la Parte B, tomada de las fuentes: mismo modelo variando solo el PE con "sin PE" como control (Guo, 2024); bidireccional frente a causal (Zhang et al., 2024; Hayakawa y Sato, 2025); tarea donde el orden importa (intercambio de dos palabras) y otra donde no (idioma) (Guo, 2024); ablación del PE en evaluación y 3 semillas (Banđur et al., 2026). Dataset: Tatoeba español-inglés.
5. Prototipo en Colab: en CPU (1 hilo) los 24 entrenamientos tardaban más de 12 minutos; lo interrumpí y pasé a GPU T4: unos 80 s. Probé 25 épocas: la validación del modelo con PE se estanca cerca de 0.56-0.57, así que dejé 8 épocas.
6. Agregué la sección 3.18 para responder la consigna con los modelos entrenados (embeddings de salida de "el gato persiguió al perro" y "el perro persiguió al gato").
7. Notebook final en Google Colab (`vibe_quest_s13_transformer_sin_pe.ipynb`): 85 celdas (40 de texto, 45 de código), secciones con # y ##, 5 paneles interactivos. Ejecutado completo sin errores; los 6 chequeos de la Parte A y los 7 de la Parte B dan verdadero. Probé los paneles con 12 combinaciones de tarea/atención/semilla, 3 escalas de PE y 4 oraciones (incluidas una corta y una sin palabras conocidas). El contenido de Colab coincide celda por celda con el archivo local.
8. Salidas de Colab en `resultados/google colab/`.
9. Revisión crítica: probé clasificar con el vector de `<BOS>` en vez del promedio para ver si el modelo con PE mejoraba en la tarea de orden. Bidireccional: 0.503 sin PE y 0.565 con PE (peor que 0.588 con promedio). Causal leyendo la última palabra: 0.610 sin PE y 0.636 con PE. No mejoró, así que mantuve el promedio y lo dejé anotado en la sección 6.5. Recorté el guion para centrarlo en la Parte A y la sección 3.18; la exactitud de la tarea de orden (0.59 con PE) queda como respaldo en el notebook y en las preguntas posibles, no en la exposición.

## Parámetros
Parte A: D = 16, CAPAS = 2, semillas 0 (pesos), 1 (embeddings), 7 (permutaciones). Parte B: 4-10 palabras, N_ORDEN = 20 000, N_IDIOMA = 10 000 por lengua, d = 64, 2 capas, 4 cabezas, FFN 128, dropout 0.1, Adam lr 0.002, lote 256, 8 épocas, semillas 0-2.

## Resultados clave
- Parte A: sin PE, error de equivarianza 1.1e-15 y coseno de "gato" 1.0; con PE, 1.47 y 0.87.
- Parte B, consigna con modelos entrenados: bidireccional sin PE, error 6e-7, coseno 1.0 y la misma P(original) para las dos frases (0.462); con PE, coseno 0.81 y P(original) 0.567 contra 0.469.
- Tarea de orden (prueba, media ± desv.): bidireccional sin PE 0.508 ± 0.016; con PE 0.588 ± 0.008 y 0.499 al quitarle el PE; causal sin PE 0.626 ± 0.010; causal con PE 0.623 ± 0.002.
- Tarea de idioma: 0.995-0.996 en las 4 configuraciones.

## Limitaciones
Modelo pequeño (la tarea de orden llega a 0.59-0.63). Un tipo de PE, 3 semillas, un dataset. La Parte A usa pesos aleatorios. Dos ejecuciones en la misma T4 dieron cifras idénticas; en otro hardware pueden variar.

## Regla de mantenimiento
Si cambias parámetros de la Parte A, vuelve a correr `codigo/ejecutar_referencia.py` y actualiza los valores de la sección 2.14. Si cambias la Parte B, actualiza las cifras de la sección 6 y de `resultados/interpretacion_resultados.md`.
