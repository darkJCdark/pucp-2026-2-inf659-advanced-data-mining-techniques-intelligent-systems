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

# Figura 2: PCA 2D de las salidas de A y B (circulo = A, cruz = B)
def proyectar(salidas):
    todo = np.concatenate(salidas); mu = todo.mean(0)
    _, _, vt = np.linalg.svd(todo - mu, full_matrices=False)
    return [(s - mu) @ vt[:2].T for s in salidas]

colores = {"El": "tab:gray", "gato": "tab:orange", "persiguió": "tab:green", "al": "tab:purple", "perro": "tab:blue"}
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a_, usar in zip(ax, (False, True)):
    sA, _ = ejecutar(FRASE_A, emb, capas, usar); sB, _ = ejecutar(FRASE_B, emb, capas, usar)
    pA, pB = proyectar([sA, sB])
    for i, w in enumerate(FRASE_A):
        a_.scatter(*pA[i], c=colores[w], marker="o", s=110); a_.annotate(f"{w} (A)", pA[i], textcoords="offset points", xytext=(6, 6))
    for i, w in enumerate(FRASE_B):
        a_.scatter(*pB[i], c=colores[w], marker="x", s=110)
    a_.set_title("Con PE" if usar else "Sin PE (círculo = A, cruz = B)"); a_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{SALIDA}/capturas/pca_salidas_con_sin_pe.png", dpi=130)

# Figura 3: distancia por palabra entre A y B
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, usar in zip(ax, (False, True)):
    sA, _ = ejecutar(FRASE_A, emb, capas, usar); sB, _ = ejecutar(FRASE_B, emb, capas, usar)
    a_.bar(FRASE_B, [np.linalg.norm(sA[PERM[j]] - sB[j]) for j in range(5)], color="tab:red" if usar else "tab:blue")
    a_.set_title("Con PE" if usar else "Sin PE"); a_.set_ylabel("‖salida(A) − salida(B)‖ por palabra")
fig.tight_layout(); fig.savefig(f"{SALIDA}/capturas/distancia_por_palabra.png", dpi=130)

# Figura 4: mapa de calor escala del embedding x escala del PE (d=16, 2 capas)
esc_emb, esc_pe = [0.1, 0.5, 1.0, 3.0, 10.0], [0.0, 0.1, 0.5, 1.0, 3.0]
mapa = np.zeros((5, 5))
for i, sp in enumerate(esc_pe):
    for j, se in enumerate(esc_emb):
        e = crear_embeddings(FRASE_A, 16, escala=se); c = crear_pesos(16, 2, SEMILLA)
        mapa[i, j] = error_equivarianza(FRASE_A, PERM, e, c, sp > 0, sp) if False else float(np.abs(ejecutar(FRASE_B, e, c, sp > 0, sp)[0] - ejecutar(FRASE_A, e, c, sp > 0, sp)[0][PERM]).max())
plt.figure(figsize=(6.5, 4.5)); plt.imshow(mapa, cmap="magma", aspect="auto"); plt.colorbar(label="error de equivarianza")
plt.xticks(range(5), esc_emb); plt.yticks(range(5), esc_pe); plt.xlabel("escala del embedding"); plt.ylabel("escala del PE")
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{mapa[i, j]:.2g}", ha="center", va="center", color="w", fontsize=8)
plt.tight_layout(); plt.savefig(f"{SALIDA}/capturas/mapa_calor_escalas.png", dpi=130)
print("mapa de calor (filas = escala PE):\n", np.round(mapa, 3))
