"""Ejecuta los experimentos de referencia (NumPy) y guarda CSV y figuras."""
import sys, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from referencia_numpy import *

SALIDA = sys.argv[1]
D, CAPAS, SEMILLA = 16, 2, 0
emb = crear_embeddings(FRASE_A, D)
capas = crear_pesos(D, CAPAS, SEMILLA)
PERM = [0, 4, 2, 3, 1]  # intercambia gato<->perro: A -> B
filas = []

# E1: frases de la consigna
for usar_pe in (False, True):
    sA, _ = ejecutar(FRASE_A, emb, capas, usar_pe)
    sB, _ = ejecutar(FRASE_B, emb, capas, usar_pe)
    err = float(np.abs(sB - sA[PERM]).max())
    pool = float(np.abs(sA.mean(0) - sB.mean(0)).max())
    cos_gato = coseno(sA[1], sB[4])          # 'gato' en A (pos 1) vs en B (pos 4)
    cos_pos = np.mean([coseno(sA[i], sB[i]) for i in range(5)])
    filas.append(dict(experimento="E1_consigna", usar_pe=usar_pe, d=D, capas=CAPAS,
                      error_equivarianza=err, dif_promedio_frase=pool,
                      cos_gato_A_vs_B=cos_gato, cos_misma_posicion=cos_pos))
pd.DataFrame(filas).to_csv(f"{SALIDA}/datos/e1_consigna.csv", index=False)
print(pd.DataFrame(filas).to_string())

# E2: barrido sobre d, capas y escala de embedding (relacion senal PE / embedding)
filas = []
for d, nc, esc in itertools.product([8, 16, 32, 64], [1, 2, 4, 6], [0.1, 1.0, 3.0, 10.0]):
    e = crear_embeddings(FRASE_A, d, escala=esc); c = crear_pesos(d, nc, SEMILLA)
    for usar_pe in (False, True):
        sA, _ = ejecutar(FRASE_A, e, c, usar_pe); sB, _ = ejecutar(FRASE_B, e, c, usar_pe)
        filas.append(dict(d=d, capas=nc, escala_emb=esc, usar_pe=usar_pe,
                          error_equivarianza=float(np.abs(sB - sA[PERM]).max()),
                          dif_promedio_frase=float(np.abs(sA.mean(0) - sB.mean(0)).max())))
e2 = pd.DataFrame(filas); e2.to_csv(f"{SALIDA}/datos/e2_barrido.csv", index=False)
print(e2.groupby("usar_pe")[["error_equivarianza", "dif_promedio_frase"]].describe().T.to_string())

# E3: frases adicionales y 100 permutaciones aleatorias
extra = {"larga_7": ["El", "gato", "negro", "persiguió", "al", "perro", "grande"],
         "repetida": ["El", "gato", "vio", "al", "gato", "gris"]}
rng = np.random.default_rng(7); filas = []
for nombre, fr in {"consigna": FRASE_A, **extra}.items():
    e = crear_embeddings(fr, D)
    for usar_pe in (False, True):
        errs, difs = [], []
        for _ in range(100):
            perm = list(rng.permutation(len(fr)))
            errs.append(error_equivarianza(fr, perm, e, capas, usar_pe))
            sal, _ = ejecutar(fr, e, capas, usar_pe)
            sal_p, _ = ejecutar([fr[j] for j in perm], e, capas, usar_pe)
            difs.append(float(np.abs(sal.mean(0) - sal_p.mean(0)).max()))
        filas.append(dict(frase=nombre, n_tokens=len(fr), usar_pe=usar_pe,
                          err_eq_max=max(errs), err_eq_media=float(np.mean(errs)),
                          dif_frase_max=max(difs), dif_frase_media=float(np.mean(difs))))
e3 = pd.DataFrame(filas); e3.to_csv(f"{SALIDA}/datos/e3_permutaciones.csv", index=False)
print(e3.to_string())

# Figuras: matrices de atencion capa 1
fig, ax = plt.subplots(2, 2, figsize=(9, 8))
for r, (fr, tit) in enumerate([(FRASE_A, "A"), (FRASE_B, "B")]):
    for c_, usar in enumerate((False, True)):
        _, att = ejecutar(fr, emb, capas, usar)
        ax[r, c_].imshow(att[0], cmap="viridis"); ax[r, c_].set_xticks(range(5)); ax[r, c_].set_yticks(range(5))
        ax[r, c_].set_xticklabels(fr, rotation=45); ax[r, c_].set_yticklabels(fr)
        ax[r, c_].set_title(f"Frase {tit} - {'con PE' if usar else 'sin PE'}")
fig.tight_layout(); fig.savefig(f"{SALIDA}/capturas/atencion_con_sin_pe.png", dpi=130)
