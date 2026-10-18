# Experimento con varias semillas: ¿el autoencoder no lineal mejora a PCA de forma consistente?
# Se repite el experimento del notebook con N semillas distintas. En cada semilla cambian:
#   (1) el rollo suizo y el ruido, (2) la división train/test y (3) la inicialización de los autoencoders.
# Salida: tabla en consola, resultados_semillas_pytorch.csv y grafico_semillas_pytorch.png (en la carpeta del script).
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
SEMILLAS = list(range(10))                  # 10 semillas
NIVELES_RUIDO = [0.0, 0.3, 0.6]
DIMENSIONES = [1, 2]                        # k = 3 no comprime (los datos ya son 3D)
SALIDA = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")   # también corre pegado en una celda de Colab


class Autoencoder:
    """Autoencoder en PyTorch (misma arquitectura que el notebook PyTorch)."""
    def __init__(self, k, tipo, semilla, epocas=2500, lr=0.005):
        self.epocas, self.lr = epocas, lr
        torch.manual_seed(semilla)
        if tipo == "lineal":
            self.red = nn.Sequential(nn.Linear(3, k), nn.Linear(k, 3))
        else:
            self.red = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, k), nn.Tanh(),
                                     nn.Linear(k, 32), nn.Tanh(), nn.Linear(32, 3))

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        opt = torch.optim.Adam(self.red.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
        perdida = nn.MSELoss()
        for _ in range(self.epocas):
            opt.zero_grad()
            perdida(self.red(X), X).backward()
            opt.step()
            sch.step()
        return self

    def reconstruir(self, X):
        with torch.no_grad():
            return self.red(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def error(X, Xrec):
    return float(np.mean((X - Xrec) ** 2))


filas = []
inicio = time.time()
for semilla in SEMILLAS:
    X0, t = make_swiss_roll(n_samples=1500, noise=0.0, random_state=semilla)
    X0 = StandardScaler().fit_transform(X0)
    ruido_base = np.random.default_rng(semilla).normal(size=X0.shape)
    for ruido in NIVELES_RUIDO:
        X = X0 + ruido * ruido_base
        Xtr, Xte = train_test_split(X, test_size=0.2, random_state=semilla)
        for k in DIMENSIONES:
            pca = PCA(n_components=k).fit(Xtr)
            e_pca = error(Xte, pca.inverse_transform(pca.transform(Xte)))
            e_lin = error(Xte, Autoencoder(k, "lineal", semilla).fit(Xtr).reconstruir(Xte))
            e_nol = error(Xte, Autoencoder(k, "no lineal", semilla).fit(Xtr).reconstruir(Xte))
            filas.append(dict(semilla=semilla, ruido=ruido, k=k, PCA=e_pca, AE_lineal=e_lin, AE_no_lineal=e_nol))
    print(f"semilla {semilla} lista ({time.time() - inicio:.0f} s)", flush=True)

df = pd.DataFrame(filas)
df["mejora_%"] = 100 * (1 - df["AE_no_lineal"] / df["PCA"])      # positivo = el no lineal reconstruye mejor
df["dif_lineal_vs_PCA_%"] = 100 * (df["AE_lineal"] / df["PCA"] - 1)
df.to_csv(SALIDA / "resultados_semillas_pytorch.csv", index=False)

resumen = df.groupby(["k", "ruido"]).agg(
    PCA=("PCA", "mean"), AE_lineal=("AE_lineal", "mean"), AE_no_lineal=("AE_no_lineal", "mean"),
    mejora_media=("mejora_%", "mean"), mejora_sd=("mejora_%", "std"),
    mejora_min=("mejora_%", "min"), gana_no_lineal=("mejora_%", lambda s: int((s > 0).sum())),
    dif_lineal_max_abs=("dif_lineal_vs_PCA_%", lambda s: float(s.abs().max())),
).round(3)
print(f"\nResumen sobre {len(SEMILLAS)} semillas (error de reconstrucción en test; mejora = % menos error que PCA)")
print(resumen.to_string())

# Gráfico: mejora porcentual por semilla y media (k = 2)
fig, ax = plt.subplots(1, len(DIMENSIONES), figsize=(10, 3.8), sharey=True)
for a, k in zip(ax, DIMENSIONES):
    sub = df[df.k == k]
    for ruido, color in zip(NIVELES_RUIDO, ["#1b9e77", "#d95f02", "#7570b3"]):
        s = sub[sub.ruido == ruido]
        a.scatter([ruido] * len(s), s["mejora_%"], color=color, alpha=0.5, s=18)
        a.plot([ruido], [s["mejora_%"].mean()], "k_", markersize=22, mew=2)
    a.axhline(0, color="gray", lw=1)
    a.set_title(f"k = {k}"); a.set_xlabel("ruido"); a.set_xticks(NIVELES_RUIDO)
ax[0].set_ylabel("mejora del AE no lineal vs PCA (%)")
plt.tight_layout(); plt.savefig(SALIDA / "grafico_semillas_pytorch.png", dpi=110); plt.show()
print("\nArchivos guardados: resultados_semillas_pytorch.csv, grafico_semillas_pytorch.png")

print("\nCorridas donde el AE no lineal mejora menos de 5 %:")
print(df[df["mejora_%"] < 5][["semilla", "ruido", "k", "PCA", "AE_no_lineal", "mejora_%"]].round(3).to_string())
