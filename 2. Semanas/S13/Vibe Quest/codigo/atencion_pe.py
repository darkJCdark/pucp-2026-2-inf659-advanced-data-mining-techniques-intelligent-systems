"""Funciones del experimento P2 (S13) en PyTorch: codificador Transformer con y sin Positional Encoding.
Son las mismas funciones del notebook vibe_quest_s13.ipynb (secciones 3, 4 y 11)."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

D = 16
CAPAS = 2
SEMILLA = 0


def pe_sinusoidal(n_pos, d):
    pos = torch.arange(n_pos)[:, None].double()
    i = torch.arange(0, d, 2)[None, :].double()
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(n_pos, d)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


def crear_pesos(d, n_capas, semilla=0):
    rng = np.random.default_rng(semilla)
    d_ff = 4 * d
    capas = []
    for _ in range(n_capas):
        capas.append({
            "Wq": rng.normal(0, d ** -0.5, (d, d)), "Wk": rng.normal(0, d ** -0.5, (d, d)),
            "Wv": rng.normal(0, d ** -0.5, (d, d)), "Wo": rng.normal(0, d ** -0.5, (d, d)),
            "W1": rng.normal(0, d ** -0.5, (d, d_ff)), "W2": rng.normal(0, d_ff ** -0.5, (d_ff, d)),
        })
    return [{k: torch.tensor(v) for k, v in capa.items()} for capa in capas]


def crear_embeddings(vocab, d, semilla=1, escala=1.0):
    rng = np.random.default_rng(semilla)
    return {w: torch.tensor(escala * rng.normal(0, 1, d)) for w in sorted(set(vocab))}


def codificador(x, capas):
    d = x.shape[1]
    atenciones = []
    for p in capas:
        q, k, v = x @ p["Wq"], x @ p["Wk"], x @ p["Wv"]
        a = torch.softmax(q @ k.T / d ** 0.5, dim=-1)
        atenciones.append(a)
        x = F.layer_norm(x + (a @ v) @ p["Wo"], (d,))
        x = F.layer_norm(x + torch.relu(x @ p["W1"]) @ p["W2"], (d,))
    return x, atenciones


def ejecutar(frase, emb, capas, usar_pe, escala_pe=1.0):
    x = torch.stack([emb[w] for w in frase])
    if usar_pe:
        x = x + escala_pe * pe_sinusoidal(len(frase), x.shape[1])
    return codificador(x, capas)


def permutacion(frase_a, frase_b):
    return [frase_a.index(w) for w in frase_b]


def coseno(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


def medir(frase_a, frase_b, emb, capas, usar_pe, escala_pe=1.0):
    perm = permutacion(frase_a, frase_b)
    sa, _ = ejecutar(frase_a, emb, capas, usar_pe, escala_pe)
    sb, _ = ejecutar(frase_b, emb, capas, usar_pe, escala_pe)
    return {
        "error_equivarianza": (sb - sa[perm]).abs().max().item(),
        "dif_promedio_frase": (sa.mean(0) - sb.mean(0)).abs().max().item(),
    }


def experimento_permutaciones(frases, n_perm=100, semilla_perm=7):
    rng = np.random.default_rng(semilla_perm)
    filas = []
    for nombre, fr in frases.items():
        e = crear_embeddings(fr, D)
        for usar_pe in (False, True):
            errs, difs = [], []
            for _ in range(n_perm):
                perm = list(rng.permutation(len(fr)))
                sal, _ = ejecutar(fr, e, capas, usar_pe)
                sal_p, _ = ejecutar([fr[j] for j in perm], e, capas, usar_pe)
                errs.append((sal_p - sal[perm]).abs().max().item())
                difs.append((sal.mean(0) - sal_p.mean(0)).abs().max().item())
            filas.append({"frase": nombre, "n_tokens": len(fr), "usar_pe": usar_pe,
                          "err_eq_max": max(errs), "err_eq_media": float(np.mean(errs)),
                          "dif_frase_max": max(difs), "dif_frase_media": float(np.mean(difs))})
    return pd.DataFrame(filas)


capas = crear_pesos(D, CAPAS, SEMILLA)  # pesos globales que usa experimento_permutaciones
