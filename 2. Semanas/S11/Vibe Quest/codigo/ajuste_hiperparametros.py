"""Ajuste previo de hiperparámetros con una semilla distinta (100) a las de evaluación (0, 1, 2).

Se prueba T = 30 con las mismas reglas para ambos modelos y se elige la configuración
con mayor exactitud en prueba. Escribe ../resultados/datos/ajuste_hiperparametros.csv.
"""
import sys
import pandas as pd
import experimento_gradientes as e

T_AJUSTE, SEMILLA_AJUSTE = 30, 100
filas = []
for lr in (0.002, 0.005, 0.01, 0.02):
    for radio in (1.0, 1.2):
        e.LR = lr
        r = e.entrenar("RNN", T_AJUSTE, SEMILLA_AJUSTE, clip=False, radio=radio)
        filas.append({"modelo": "RNN", "lr": lr, "radio_Wh": radio, "sesgo_olvido": None, "exactitud_prueba": r["exactitud_prueba"], "norma_grad_max": r["norma_grad_max"]})
        print(filas[-1], flush=True)
for lr in (0.005, 0.01, 0.02):
    for sesgo in (1.0, 2.0):
        e.LR, e.SESGO_OLVIDO = lr, sesgo
        r = e.entrenar("LSTM", T_AJUSTE, SEMILLA_AJUSTE)
        filas.append({"modelo": "LSTM", "lr": lr, "radio_Wh": None, "sesgo_olvido": sesgo, "exactitud_prueba": r["exactitud_prueba"], "norma_grad_max": r["norma_grad_max"]})
        print(filas[-1], flush=True)
df = pd.DataFrame(filas)
df.round(4).to_csv(e.DATOS / "ajuste_hiperparametros.csv", index=False)
print(df.round(3).to_string(index=False))
