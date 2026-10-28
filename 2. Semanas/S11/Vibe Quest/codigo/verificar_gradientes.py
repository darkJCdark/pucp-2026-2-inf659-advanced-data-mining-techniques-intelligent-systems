"""Comprueba el gradiente analítico (BPTT a mano) contra diferencias finitas."""
import numpy as np
from modelos_numpy import RNNSimple, LSTM
from tarea_memoria import generar


def revisar(modelo, X, y, nombre, eps=1e-5):
    _, _, g, _ = modelo.paso(X, y)
    peor = 0.0
    rng = np.random.default_rng(1)
    for k, v in modelo.p.items():
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in v.shape)
            v0 = v[idx]
            v[idx] = v0 + eps
            lp = modelo.paso(X, y, False)[0]
            v[idx] = v0 - eps
            lm = modelo.paso(X, y, False)[0]
            v[idx] = v0
            num = (lp - lm) / (2 * eps)
            err = abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx]))
            peor = max(peor, err)
    print(f"{nombre}: error relativo máximo = {peor:.2e}")
    assert peor < 1e-5, f"gradiente incorrecto en {nombre}"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, y = generar(8, 6, rng)
    revisar(RNNSimple(X.shape[2], 7, 4, rng), X, y, "RNN simple")
    revisar(LSTM(X.shape[2], 7, 4, rng), X, y, "LSTM")
