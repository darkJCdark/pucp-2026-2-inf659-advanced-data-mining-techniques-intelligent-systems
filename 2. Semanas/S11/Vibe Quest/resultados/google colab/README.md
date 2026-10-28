# resultados/google colab

Salidas de la ejecución del notebook `vibe_quest_s11_gradientes_rnn_lstm.ipynb` en Google Colab (PyTorch 2.11.0, 3 modelos × 6 longitudes × 3 semillas, 803 s). Corresponden a la segunda de las dos ejecuciones que hice; con las mismas semillas, la primera dio cifras distintas (ver sección 15 del notebook).

| Archivo | Contenido |
| --- | --- |
| `resultados_entrenamiento_torch.csv` | 54 entrenamientos: pérdida y exactitud en prueba, norma máxima del gradiente y log10 de la razón de gradientes |
| `resumen_entrenamiento_torch.csv` | Media y mínimo de exactitud por modelo y longitud |
| `chequeos_hipotesis_torch.csv` | Los 9 chequeos de la sección 12 (True/False) |
| `comparacion_numpy_torch.csv` | Exactitud media en PyTorch frente a la referencia NumPy |

Las dos figuras (`fig_exactitud_vs_longitud_torch.png` y `fig_gradiente_por_distancia_torch.png`) se generan en `resultados_s11/` dentro de Colab y no se descargaron a esta carpeta. La sección 16 del notebook las comprime en `resultados_s11.zip` y la descarga.
