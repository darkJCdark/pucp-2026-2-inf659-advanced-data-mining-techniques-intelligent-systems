"""Vibe Quest S10 (pregunta 3): superposición, autoencoder disperso (SAE) y steering.

Experimento autónomo (PyTorch, CPU, ~6 min). Misma lógica que el cuaderno
VibeQuest_S10_SAE_Monosemanticidad_Steering.ipynb. Escribe en ../resultados/datos
y ../resultados/capturas (o en la carpeta actual si no existen).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Parámetros del experimento ----------------
SEMILLA = 0
N_RASGOS = 12            # rasgos "verdaderos" que genera el mundo sintético
N_DIM = 4                # neuronas de la capa oculta del modelo de juguete (menos que rasgos)
K_DICT = 24              # latentes del autoencoder disperso (factor de expansión 6)
ESPARSIDADES = [0.5, 0.9, 0.97]   # probabilidad de que un rasgo esté apagado
ALPHAS = [0.0, 0.1, 0.3, 1.0]     # peso de la penalización L1 del autoencoder disperso
PASOS_MODELO = 4000
PASOS_SAE = 5000
LOTE = 2048

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)


def muestrear(n, esparsidad, generador):
    """Cada rasgo está activo con prob. (1 - esparsidad); si está activo, vale U(0,1)."""
    activo = (torch.rand(n, N_RASGOS, generator=generador) > esparsidad).float()
    valor = torch.rand(n, N_RASGOS, generator=generador)
    return activo * valor


class ModeloJuguete(nn.Module):
    """x (12 rasgos) -> h = W x (4 neuronas) -> ReLU(W^T h + b): comprime y reconstruye."""

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(N_DIM, N_RASGOS) * 0.5)
        self.b = nn.Parameter(torch.zeros(N_RASGOS))

    def oculta(self, x):
        return x @ self.W.T

    def salida(self, h):
        return torch.relu(h @ self.W + self.b)

    def forward(self, x):
        return self.salida(self.oculta(x))


class SAE(nn.Module):
    """Autoencoder disperso: h (4) -> ReLU -> f (24 latentes) -> h' (4)."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(N_DIM, K_DICT)
        self.dec = nn.Parameter(torch.randn(N_DIM, K_DICT))
        self.bd = nn.Parameter(torch.zeros(N_DIM))

    def direcciones(self):
        # columnas del decodificador con norma 1: así L1 no se "evade" achicando f
        return self.dec / self.dec.norm(dim=0, keepdim=True)

    def codificar(self, h):
        return torch.relu(self.enc(h - self.bd))

    def decodificar(self, f):
        return f @ self.direcciones().T + self.bd

    def forward(self, h):
        f = self.codificar(h)
        return self.decodificar(f), f


def entrenar_modelo(esparsidad, semilla=SEMILLA):
    torch.manual_seed(semilla)
    g = torch.Generator().manual_seed(semilla)
    m = ModeloJuguete()
    opt = torch.optim.Adam(m.parameters(), lr=0.005)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    for _ in range(PASOS_MODELO):
        x = muestrear(LOTE, esparsidad, g)
        perdida = ((m(x) - x) ** 2).mean()
        opt.zero_grad(); perdida.backward(); opt.step(); sch.step()
    return m


def entrenar_sae(modelo, esparsidad, alpha, semilla=SEMILLA):
    torch.manual_seed(semilla)
    g = torch.Generator().manual_seed(semilla + 1)
    sae = SAE()
    opt = torch.optim.Adam(sae.parameters(), lr=0.002)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
    for _ in range(PASOS_SAE):
        with torch.no_grad():
            h = modelo.oculta(muestrear(LOTE, esparsidad, g))
        h_rec, f = sae(h)
        rec = ((h_rec - h) ** 2).sum(1).mean()
        disp = f.sum(1).mean()
        perdida = rec + alpha * disp
        opt.zero_grad(); perdida.backward(); opt.step(); sch.step()
    return sae


def corr_max(A, B):
    """Para cada columna de A, mayor correlación con alguna columna de B (0 si es constante)."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-8)
    B = (B - B.mean(0)) / (B.std(0) + 1e-8)
    C = (A.T @ B) / A.shape[0]
    return C.max(1).values, C


@torch.no_grad()
def evaluar(modelo, sae, esparsidad, n=5000, semilla=123):
    g = torch.Generator().manual_seed(semilla)
    x = muestrear(n, esparsidad, g)
    h = modelo.oculta(x)
    f = sae.codificar(h)
    h_rec = sae.decodificar(f)
    W = modelo.W.detach()
    normas = W.norm(dim=0)
    repr_ = normas > 0.5                                   # rasgos que el modelo sí representa
    Wn = W / (normas + 1e-8)
    cos = Wn.T @ sae.direcciones()                         # (12 rasgos x 24 latentes)
    recupera = (cos.max(1).values > 0.9)[repr_].float().mean().item() if repr_.any() else float("nan")
    # pureza: correlación máxima de cada unidad (neurona / latente) con algún rasgo verdadero
    p_neu, _ = corr_max(h, x)
    vivos = (f > 1e-6).float().mean(0) > 0.005             # latentes que se activan alguna vez
    p_lat, _ = corr_max(f[:, vivos], x) if vivos.any() else (torch.zeros(1), None)
    err = ((h_rec - h) ** 2).sum(1).mean().item() / (h ** 2).sum(1).mean().item()
    return dict(
        representados=int(repr_.sum()),
        recupera=recupera,
        pureza_neurona=p_neu.mean().item(),
        pureza_latente=p_lat.mean().item(),
        l0=(f > 1e-6).float().sum(1).mean().item(),
        vivos=int(vivos.sum()),
        error_rel=err,
    )


@torch.no_grad()
def dirigir(modelo, sae, esparsidad, rasgo, factor, metodo, n=5000, semilla=321):
    """Fija (clamp) una unidad a factor x su activación máxima y mide el cambio medio en las 12 salidas."""
    g = torch.Generator().manual_seed(semilla)
    x = muestrear(n, esparsidad, g)
    h = modelo.oculta(x)
    base = modelo.salida(h)
    if metodo == "SAE":
        f = sae.codificar(h)
        W = modelo.W.detach()
        Wn = W / (W.norm(dim=0) + 1e-8)
        j = int((Wn[:, rasgo] @ sae.direcciones()).argmax())      # latente más alineado al rasgo
        f2 = f.clone()
        f2[:, j] = factor * f[:, j].max()
        h2 = sae.decodificar(f2)
    else:
        k = int(modelo.W.detach()[:, rasgo].abs().argmax())       # neurona que más pesa en ese rasgo
        h2 = h.clone()
        h2[:, k] = factor * h[:, k].abs().max()
    delta = (modelo.salida(h2) - base).mean(0)
    otros = torch.ones(N_RASGOS, dtype=torch.bool); otros[rasgo] = False
    selectividad = delta[rasgo].abs().item() / (delta[otros].abs().sum().item() + 1e-8)
    return delta.numpy(), selectividad


# ---------------- Ejecución ----------------
def carpeta(nombre):
    base = Path(__file__).resolve().parent.parent / "resultados" if "__file__" in globals() else Path(".")
    destino = base / nombre
    destino.mkdir(parents=True, exist_ok=True)
    return destino


if __name__ == "__main__":
    DATOS, CAPTURAS = carpeta("datos"), carpeta("capturas")
    t0 = time.time()
    modelos = {S: entrenar_modelo(S) for S in ESPARSIDADES}
    saes = {(S, a): entrenar_sae(modelos[S], S, a) for S in ESPARSIDADES for a in ALPHAS}
    print(f"Entrenados {len(modelos)} modelos y {len(saes)} SAE en {time.time() - t0:.0f} s")

    filas = []
    for S in ESPARSIDADES:
        for a in ALPHAS:
            filas.append({"esparsidad": S, "alpha": a, **evaluar(modelos[S], saes[(S, a)], S)})
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(DATOS / "resultados_tabla_sae.csv", index=False)
    print(tabla.round(3).to_string(index=False))

    def fila(S, a):
        return tabla[(tabla.esparsidad == S) & (tabla.alpha == a)].iloc[0]
    for S in (0.9, 0.97):
        f = fila(S, 0.3)
        assert f.recupera >= 0.9 and f.pureza_latente > f.pureza_neurona + 0.2, f"fallo comprobación S={S}"
    assert fila(0.5, 0.3).recupera < 0.5, "con rasgos densos no debería recuperar casi nada"
    print("Comprobaciones superadas")

    # Robustez: 5 semillas en la configuración principal
    out = []
    for s in [1, 2, 3, 4, 5]:
        m = entrenar_modelo(0.97, semilla=s)
        z = entrenar_sae(m, 0.97, 0.3, semilla=s)
        r = evaluar(m, z, 0.97)
        sa, ne = [], []
        for i in range(N_RASGOS):
            if m.W.detach()[:, i].norm() > 0.5:
                sa.append(dirigir(m, z, 0.97, i, 3.0, "SAE")[1])
                ne.append(dirigir(m, z, 0.97, i, 3.0, "Neurona")[1])
        out.append({"semilla": s, **r, "selectividad_SAE": np.mean(sa), "selectividad_neurona": np.mean(ne)})
    robustez = pd.DataFrame(out)
    robustez.round(4).to_csv(DATOS / "resultados_robustez_semillas.csv", index=False)
    print(robustez.round(3).to_string(index=False))
    resumen = robustez[["recupera", "pureza_neurona", "pureza_latente", "selectividad_SAE", "selectividad_neurona"]].agg(["mean", "std"])
    resumen.round(4).to_csv(DATOS / "resultados_robustez_resumen.csv")
    print(resumen.round(3))

    # Steering: tabla para la semilla 0 (todos los rasgos, factor 3)
    filas = []
    for S in (0.9, 0.97):
        for i in range(N_RASGOS):
            if modelos[S].W.detach()[:, i].norm() > 0.5:
                filas.append({"esparsidad": S, "rasgo": i,
                              "selectividad_SAE": dirigir(modelos[S], saes[(S, 0.3)], S, i, 3.0, "SAE")[1],
                              "selectividad_neurona": dirigir(modelos[S], saes[(S, 0.3)], S, i, 3.0, "Neurona")[1]})
    steer = pd.DataFrame(filas)
    steer.round(4).to_csv(DATOS / "resultados_steering_semilla0.csv", index=False)
    print(steer.groupby("esparsidad")[["selectividad_SAE", "selectividad_neurona"]].agg(["mean", "std"]).round(3))

    # Figuras: panel A (S=0.97, alpha=0.3) y panel B (rasgo 1)
    for S, a in [(0.97, 0.3), (0.5, 0.3)]:
        m, z = modelos[S], saes[(S, a)]
        W = m.W.detach(); Wn = W / (W.norm(dim=0) + 1e-8)
        cos = (Wn.T @ z.direcciones()).detach(); orden = cos.argmax(1).argsort()
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].imshow(W.abs(), aspect="auto", cmap="viridis"); ax[0].set_title(f"|W| neuronas x rasgos (esparsidad {S})")
        ax[1].imshow(cos[orden].clamp(min=0), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[1].set_title(f"coseno rasgo x latente SAE (alpha {a})")
        plt.tight_layout(); plt.savefig(CAPTURAS / f"panelA_S{S}_a{a}.png", dpi=110); plt.close()
    for metodo in ("SAE", "Neurona"):
        delta, sel = dirigir(modelos[0.97], saes[(0.97, 0.3)], 0.97, 1, 3.0, metodo)
        plt.figure(figsize=(7, 3)); plt.bar(range(N_RASGOS), delta, color=["#e76f51" if i == 1 else "#8d99ae" for i in range(N_RASGOS)])
        plt.title(f"Dirigir rasgo 1 vía {metodo}: selectividad {sel:.2f}"); plt.tight_layout()
        plt.savefig(CAPTURAS / f"panelB_{metodo}_rasgo1.png", dpi=110); plt.close()
    print(f"Terminado en {time.time() - t0:.0f} s")
