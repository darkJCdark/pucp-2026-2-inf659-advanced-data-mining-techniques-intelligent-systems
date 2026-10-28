"""Vibe Quest S11 · Gradientes que se desvanecen y explotan en una RNN simple frente a una LSTM.

Ejecutar: python experimento_gradientes.py
Genera tablas en ../resultados/datos y figuras en ../resultados/capturas.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelos_numpy import RNNSimple, LSTM, Adam, recortar
from tarea_memoria import generar

# Parámetros
SEMILLAS = [0, 1, 2]
LONGITUDES = [5, 10, 20, 30, 50, 80]
N_CLASES = 4
OCULTAS = 32
LOTE = 64
PASOS = 3000
LR = 0.005
UMBRAL_CLIP = 1.0
N_PRUEBA = 2000

BASE = Path(__file__).resolve().parent.parent / "resultados"
DATOS, CAPTURAS = BASE / "datos", BASE / "capturas"
DATOS.mkdir(parents=True, exist_ok=True)
CAPTURAS.mkdir(parents=True, exist_ok=True)


SESGO_OLVIDO = 2.0


def crear(nombre, n_in, rng, radio=1.0):
    if nombre == "LSTM":
        return LSTM(n_in, OCULTAS, N_CLASES, rng, sesgo_olvido=SESGO_OLVIDO)
    return RNNSimple(n_in, OCULTAS, N_CLASES, rng, radio=radio)


def figura_exactitud(res):
    """Media por modelo y longitud, con cada semilla como punto."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for mod, color in [("RNN", "tab:blue"), ("RNN + recorte", "tab:orange"), ("LSTM", "tab:green")]:
        sub = res[res.modelo == mod]
        media = sub.groupby("T").exactitud_prueba.mean()
        ax.plot(media.index, media.values, marker="o", color=color, label=mod)
        ax.scatter(sub["T"], sub.exactitud_prueba, color=color, alpha=0.35, s=18)
    ax.axhline(1 / N_CLASES, color="gray", ls="--", label="azar")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("longitud de la secuencia T")
    ax.set_ylabel("exactitud en prueba (puntos = semillas)")
    ax.set_title("Recordar la clase de t = 0 tras T − 1 pasos de ruido")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# 1 · Aritmética de la lámina (activación lineal: dh_T/dh_1 = w^(T-1))
def aritmetica():
    filas = []
    for w in (1.1, 0.9):
        for T in (50,):
            filas.append({"w": w, "T": T, "w^T": w ** T, "w^(T-1)": w ** (T - 1)})
    return pd.DataFrame(filas)


# 2 · Norma del gradiente por distancia temporal, sin entrenar
def normas_inicio(T=50, n=256, semillas=SEMILLAS):
    series = {}
    for nombre, radio in [("RNN radio 0.9", 0.9), ("RNN radio 1.0", 1.0), ("RNN radio 1.5", 1.5), ("RNN radio 2.0", 2.0), ("LSTM", None)]:
        acum = []
        for s in semillas:
            rng = np.random.default_rng(100 + s)
            X, y = generar(n, T, rng)
            m = crear("LSTM" if nombre == "LSTM" else "RNN", X.shape[2], rng, radio or 1.0)
            _, _, _, normas = m.paso(X, y)
            acum.append(normas)
        series[nombre] = np.mean(acum, axis=0)
    return series


# 3 · Entrenamiento
def entrenar(nombre, T, semilla, clip=False, radio=1.0):
    rng = np.random.default_rng(semilla)
    Xp, yp = generar(N_PRUEBA, T, np.random.default_rng(999))
    m = crear(nombre, Xp.shape[2], rng, radio)
    opt = Adam(m.p, lr=LR)
    perdidas, ratios, normas_g = [], [], []
    for paso in range(PASOS):
        X, y = generar(LOTE, T, rng)
        l, a, g, normas = m.paso(X, y)
        normas_g.append(recortar(g, UMBRAL_CLIP) if clip else recortar(g, 1e18))
        ratios.append(normas[0] / (normas[-1] + 1e-30))
        perdidas.append(l)
        opt.actualizar(m.p, g)
    lp, ap, _, _ = m.paso(Xp, yp, False)
    return {
        "perdida_final_entrenamiento": float(np.mean(perdidas[-100:])),
        "perdida_prueba": float(lp),
        "exactitud_prueba": float(ap),
        "log10_ratio_grad": float(np.median(np.log10(np.array(ratios) + 1e-30))),
        "norma_grad_max": float(np.max(normas_g)),
        "perdidas": perdidas,
    }


if __name__ == "__main__":
    t0 = time.time()
    print("1 · Aritmética de la lámina")
    ar = aritmetica()
    print(ar.to_string(index=False))
    ar.to_csv(DATOS / "aritmetica_lamina.csv", index=False)

    print("\n2 · Norma del gradiente por distancia (T=50, sin entrenar)")
    ser = normas_inicio()
    fila = {k: v[0] / v[-1] for k, v in ser.items()}
    print("||dL/dh_1|| / ||dL/dh_T||:", {k: f"{v:.3g}" for k, v in fila.items()})
    pd.DataFrame({"t": np.arange(1, 51), **ser}).to_csv(DATOS / "normas_gradiente_inicio.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in ser.items():
        ax.semilogy(np.arange(1, 51), v / v[-1], label=k)
    ax.set_xlabel("paso temporal t (la pérdida se calcula en t = 50)")
    ax.set_ylabel("‖dL/dh_t‖ / ‖dL/dh_50‖")
    ax.set_title("Gradiente según la distancia al final (sin entrenar)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(CAPTURAS / "fig1_gradiente_por_distancia.png", dpi=150)
    plt.close(fig)

    print("\n3 · Entrenamiento (RNN, RNN con recorte, LSTM)")
    filas, curvas = [], {}
    for T in LONGITUDES:
        for nombre, clip in [("RNN", False), ("RNN", True), ("LSTM", False)]:
            etiqueta = "RNN + recorte" if clip else nombre
            for s in SEMILLAS:
                r = entrenar(nombre, T, s, clip=clip)
                curvas[(etiqueta, T, s)] = r.pop("perdidas")
                filas.append({"modelo": etiqueta, "T": T, "semilla": s, **r})
                print(f"  T={T:>2} {etiqueta:<14} sem={s} exactitud={r['exactitud_prueba']:.3f} "
                      f"log10(ratio)={r['log10_ratio_grad']:.1f}  [{time.time() - t0:.0f}s]", flush=True)
    res = pd.DataFrame(filas)
    res.round(6).to_csv(DATOS / "resultados_entrenamiento.csv", index=False)
    resumen = res.groupby(["modelo", "T"]).agg(
        exactitud_media=("exactitud_prueba", "mean"), exactitud_std=("exactitud_prueba", "std"),
        log10_ratio=("log10_ratio_grad", "median"), norma_grad_max=("norma_grad_max", "max")).round(4)
    resumen.to_csv(DATOS / "resumen_entrenamiento.csv")
    print("\nResumen (3 semillas)")
    print(resumen.to_string())

    figura_exactitud(res).savefig(CAPTURAS / "fig2_exactitud_vs_longitud.png", dpi=150)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")
