"""Referencia NumPy del experimento P2 (S13): Transformer con y sin Positional Encoding.
Misma matematica que atencion_pe.py (PyTorch). Sirve para validar y fijar valores esperados."""
import numpy as np

FRASE_A = ["El", "gato", "persiguió", "al", "perro"]
FRASE_B = ["El", "perro", "persiguió", "al", "gato"]


def pe_sinusoidal(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / np.power(10000.0, i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang[:, : d // 2])
    return pe


def crear_pesos(d, n_capas, semilla=0, d_ff=None):
    rng = np.random.default_rng(semilla)
    d_ff = d_ff or 4 * d
    capas = []
    for _ in range(n_capas):
        capas.append({
            "Wq": rng.normal(0, d ** -0.5, (d, d)), "Wk": rng.normal(0, d ** -0.5, (d, d)),
            "Wv": rng.normal(0, d ** -0.5, (d, d)), "Wo": rng.normal(0, d ** -0.5, (d, d)),
            "W1": rng.normal(0, d ** -0.5, (d, d_ff)), "W2": rng.normal(0, d_ff ** -0.5, (d_ff, d)),
        })
    return capas


def crear_embeddings(vocab, d, semilla=1, escala=1.0):
    rng = np.random.default_rng(semilla)
    return {w: escala * rng.normal(0, 1, d) for w in sorted(set(vocab))}


def layer_norm(x, eps=1e-5):
    m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def codificador(x, capas):
    """x: (n, d). Devuelve salida y lista de matrices de atencion (una por capa)."""
    atenciones = []
    for p in capas:
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        a = softmax(q @ k.T / np.sqrt(x.shape[1]))
        atenciones.append(a)
        x = layer_norm(x + (a @ v) @ p["Wo"])
        x = layer_norm(x + np.maximum(x @ p["W1"], 0) @ p["W2"])
    return x, atenciones


def ejecutar(frase, emb, capas, usar_pe, escala_pe=1.0):
    x = np.stack([emb[w] for w in frase])
    if usar_pe:
        x = x + escala_pe * pe_sinusoidal(len(frase), x.shape[1])
    return codificador(x, capas)


def error_equivarianza(frase, perm, emb, capas, usar_pe, escala_pe=1.0):
    """Permuta la frase con perm. Si el modelo es equivariante: salida(perm)=salida[perm]."""
    sal, _ = ejecutar(frase, emb, capas, usar_pe, escala_pe)
    frase_p = [frase[j] for j in perm]
    sal_p, _ = ejecutar(frase_p, emb, capas, usar_pe, escala_pe)
    return float(np.abs(sal_p - sal[perm]).max())


def coseno(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
