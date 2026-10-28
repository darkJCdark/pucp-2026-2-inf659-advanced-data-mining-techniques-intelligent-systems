"""Tarea con dependencia a largo plazo: recordar la clase mostrada en t=0 tras T-1 pasos de ruido."""
import numpy as np


def generar(n, T, rng, n_clases=4, ruido=0.5):
    """X: (T, n, n_clases+1). En t=0: one-hot de la clase + indicador. En t>0: ruido gaussiano."""
    y = rng.integers(0, n_clases, n)
    X = ruido * rng.standard_normal((T, n, n_clases + 1))
    X[0] = 0.0
    X[0, np.arange(n), y] = 1.0
    X[0, :, n_clases] = 1.0
    X[1:, :, n_clases] = 0.0
    return X, y
