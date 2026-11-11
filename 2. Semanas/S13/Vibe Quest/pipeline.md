# pipeline.md · Vibe Quest S13 (pregunta 2: Transformer sin Positional Encoding)

## Objetivo
Responder: *si quito el Positional Encoding de un Transformer y paso "El gato persiguió al perro" y "El perro persiguió al gato", ¿qué pasa con los embeddings de salida? Demostrarlo con código, comparar con y sin PE y visualizar por qué se pierde el orden.* Presentación de 5 minutos con sliders.

## Flujo (entrada → modelo → salida)

```
ENTRADA                       MODELO (codificador, PyTorch float64)              SALIDA
FRASE_A  El gato persiguió    embedding por palabra (NumPy, semilla 1)           tablas (resultados/datos y google colab)
         al perro             + PE sinusoidal (opcional, escala ajustable)         e1_consigna, e2_barrido, e3_permutaciones
FRASE_B  El perro persiguió   ┌ self-attention softmax(QKᵀ/√d)V                  figuras
         al gato             ─┤ Add & Norm → FFN → Add & Norm   × CAPAS (2)        atenciones capa 1 (A/B, con/sin PE)
FRASES_EXTRA  larga_7,        └ pesos aleatorios N(0, 1/d), semilla 0              PCA 2D de las salidas (○ A, × B)
         repetida (gato ×2)                                                         distancia por palabra
                              medidas: error de equivarianza, diferencia de         mapa de calor escala emb × escala PE
                              la frase (media de filas), coseno de "gato"           panel con sliders
```

## Historial de pasos
1. Leí las láminas y la transcripción de la Clase 12 (`2. Semanas/S13`): self-attention, PE sinusoidal (lámina 120), Add & Norm (lámina 122). Las 4 preguntas están en la lámina 123; las dejé con lo visto en clase en `S13/VibeQuest_S13_Preguntas.md`.
2. Elegí la pregunta 2 (acordado contigo): es la más corta de implementar, la clase ya explica su teoría y se ejecuta sin entrenar nada. Las otras exigen entrenar (P1, P3, P4) o implementar un encoder-decoder (P1, P3).
3. Fuentes: restricción acordada, solo Scopus o IEEE Xplore 2022-2026. Busqué en IEEE Xplore (API de búsqueda desde la sesión del navegador) y en Scopus; leí los resúmenes de 6 fuentes. Ver `fuentes/README.md`.
4. Código: referencia en NumPy (`codigo/referencia_numpy.py`, `ejecutar_referencia.py`) para fijar los valores esperados; luego el mismo experimento en PyTorch (`codigo/atencion_pe.py` y el notebook), con los pesos generados en NumPy para que ambas implementaciones coincidan.
5. Experimentos: (1) las dos frases de la consigna; (2) barrido de 64 configuraciones (d ∈ {8,16,32,64}, capas ∈ {1,2,4,6}, escala del embedding ∈ {0.1,1,3,10}), con y sin PE; (3) 3 frases × 100 permutaciones aleatorias.
6. Notebook `vibe_quest_s13.ipynb` (41 celdas: 21 de texto con `#` y 20 de código, cada código separado). Incluye verificaciones con `assert` (PE contra la fórmula, equivarianza sin PE, comparación con NumPy), 3 gráficos, panel con sliders, mapa de calor, chequeos de hipótesis True/False, tabla afirmación ↔ fuente y una celda final que guarda y comprime los resultados.
7. Validación en tu Google Colab (torch 2.14.1, CPU): el notebook corre sin errores y todos los `assert` pasan; PyTorch coincide con NumPy. Probé 6 combinaciones extremas del panel (d de 8 a 64, 1 a 6 capas, escala del embedding de 0.1 a 10, escala del PE de 0 a 5, con y sin PE) sin fallos.
8. Salidas de Colab guardadas en `resultados/google colab/`.

## Parámetros
D = 16, CAPAS = 2, SEMILLA de pesos = 0, semilla de embeddings = 1, semilla de permutaciones = 7, 100 permutaciones por frase, PE sinusoidal con base 10000, float64.

## Resultados clave
- Frases de la consigna: sin PE el error de equivarianza es 1.1e-15 y el coseno entre el "gato" de A y el de B es 1.0; con PE es 1.474 y 0.869.
- Barrido (64 configuraciones): sin PE nunca supera 3.6e-15; con PE va de 0.049 a 2.283 (media 0.80).
- 100 permutaciones × 3 frases: sin PE el error máximo es 2.7e-15; con PE la media está entre 1.20 y 1.31.
- Con PE el error no crece de forma monótona con la escala del embedding (0.97, 1.10, 0.74, 0.39 para 0.1, 1, 3, 10).

## Limitaciones
Pesos aleatorios sin entrenar (propiedad estructural, no rendimiento en una tarea). Codificador bidireccional: con máscara causal un modelo sin PE puede inferir posición (Hayakawa y Sato, 2025; Zhang et al., 2024). Un tipo de PE (sinusoidal) y una semilla de pesos. Fuentes leídas por su resumen; los PDF se descargan con `codigo/descargar_fuentes.py`.

## Regla de mantenimiento
Si cambias `D`, `CAPAS`, las frases o las semillas, vuelve a correr `codigo/ejecutar_referencia.py` y actualiza los valores de la sección 12 del notebook (REF_E1_CON_PE, REF_E2, REF_E3_MEDIA_CON_PE), o los `assert` fallarán.
