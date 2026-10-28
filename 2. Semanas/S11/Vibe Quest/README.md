# Vibe Quest S11 · Gradientes que se desvanecen y explotan en una RNN simple

**Pregunta 1 de la semana 11:** ¿cómo afectan los gradientes que se desvanecen al rendimiento de una RNN simple en una tarea con dependencias a largo plazo? ¿Puedes demostrar con código cómo se desvanecen o explotan frente a un modelo normal?

## Carpetas
| Carpeta | Contenido |
| --- | --- |
| `codigo/` | Código en Python (numpy): `modelos_numpy.py` (RNN, LSTM, BPTT a mano, Adam, recorte), `tarea_memoria.py`, `verificar_gradientes.py`, `ajuste_hiperparametros.py`, `experimento_gradientes.py`, `descargar_fuentes.py` |
| `fuentes/` | `README.md` con el estado de lectura, `fuentes_doi.csv` y los PDF descargados |
| `resultados/` | `datos/` (CSV), `capturas/` (figuras), `salida_experimento.txt` (lo que imprime el script) y `interpretacion_resultados.md` |
| raíz | `pipeline.md` (flujo y historial), este `README.md` |

El cuaderno `vibe_quest_s11_gradientes_rnn_lstm.ipynb` es la versión PyTorch para Google Colab, con celdas de texto (`#`) y código separadas y sliders. Ejecutado completo en Colab sin errores; los resultados con PyTorch son más ruidosos que los de NumPy (ver `pipeline.md`, paso 8).

## Cómo ejecutar
```bash
cd codigo
python verificar_gradientes.py       # BPTT a mano contra diferencias finitas
python ajuste_hiperparametros.py     # ~3 min, semilla 100
python experimento_gradientes.py     # ~9 min, 54 entrenamientos
python descargar_fuentes.py          # abre el navegador; inicia sesión PUCP si se pide
```
Requiere `numpy`, `pandas` y `matplotlib`; `descargar_fuentes.py` además `playwright`.

## Resumen de resultados
Tarea: recordar una clase mostrada en t = 0 tras T − 1 pasos de ruido. Exactitud media de 3 semillas (azar = 0.25):

| T | RNN simple | LSTM |
| --- | --- | --- |
| 5 y 10 | 1.00 | 1.00 |
| 20 | 0.58 | 1.00 |
| 30 | 0.69 | 1.00 |
| 50 | 0.26 | 0.75 |
| 80 | 0.33 | 1.00 |

Sin entrenar y con T = 50, el gradiente que llega al paso 1 es 7.4e-6 veces el del final en una RNN de radio 1.0, y 364 veces mayor con radio 2.0 (explosión). La LSTM no es infalible: falló una de tres semillas con T = 50.

Detalle e interpretación: `resultados/interpretacion_resultados.md`.
