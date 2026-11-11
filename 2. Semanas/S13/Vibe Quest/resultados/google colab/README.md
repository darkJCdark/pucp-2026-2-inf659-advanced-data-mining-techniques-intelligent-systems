# resultados · google colab

Salidas de la ejecución del notebook `vibe_quest_s13.ipynb` en Google Colab (CPU, torch 2.14.1, numpy 2.4.6, pandas 2.2.3).

| Archivo | Contenido |
| --- | --- |
| `e1_consigna_torch.csv` | Experimento 1: las dos frases de la consigna, sin y con PE |
| `e2_resumen_torch.csv` | Experimento 2: resumen del barrido de 64 configuraciones (error de equivarianza por escala del embedding). La tabla completa de 128 filas la guarda la sección 16 del notebook (`e2_barrido_torch.csv` dentro de `resultados_s13.zip`) |
| `e3_permutaciones_torch.csv` | Experimento 3: 3 frases × 100 permutaciones aleatorias |
| `chequeos_hipotesis_torch.csv` | Los 6 chequeos de hipótesis (todos verdaderos) |

Los pesos y embeddings salen de NumPy con semilla fija y el cálculo es en float64, así que PyTorch coincide con la referencia (`../datos/*.csv`, hecha con `codigo/referencia_numpy.py`) con diferencias de 1e-15 en los valores sin PE y de 1e-9 o menos en los demás: las celdas de la sección 12 lo verifican con `assert`.
