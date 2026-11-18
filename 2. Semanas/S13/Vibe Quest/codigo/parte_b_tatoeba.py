"""Parte B del Vibe Quest S13: Transformer entrenado con Tatoeba (español-inglés), con y sin PE,
atención bidireccional y causal. Son las funciones de las secciones 3.2 a 3.10 y 3.18 del notebook.
Requiere pandas, numpy y torch; usa pe_sinusoidal de atencion_pe.py."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from atencion_pe import pe_sinusoidal

import collections
import io
import os
import random
import re
import time
import urllib.request
import zipfile

import torch.nn as nn

torch.set_default_dtype(torch.float32)
torch.set_num_threads(os.cpu_count() or 1)
DISP = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL_DATOS = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
LARGO_MIN, LARGO_MAX = 4, 10      # palabras por oración
N_ORDEN, N_IDIOMA = 20000, 10000  # oraciones para cada tarea (idioma: 10 000 por lengua)
D_MODELO, CAPAS_B, CABEZAS = 64, 2, 4
EPOCAS, LOTE, LR = 8, 256, 2e-3
SEMILLAS = (0, 1, 2)
print("dispositivo:", DISP, "| hilos CPU:", torch.get_num_threads())


def tokenizar(texto):
    return re.findall(r"[a-záéíóúüñ']+", texto.lower())


def oraciones_unicas(lineas, columna):
    unicas = {tuple(tokenizar(l.split("\t")[columna])) for l in lineas}
    return sorted(f for f in unicas if LARGO_MIN <= len(f) <= LARGO_MAX)


def intercambiar(frase, rng):
    pares = [(i, j) for i in range(len(frase)) for j in range(i + 1, len(frase)) if frase[i] != frase[j]]
    i, j = rng.choice(pares)
    f = list(frase)
    f[i], f[j] = f[j], f[i]
    return tuple(f)


def crear_tarea_orden(oraciones, semilla=1):
    rng = random.Random(semilla)
    ejemplos = []
    for f in oraciones:
        ejemplos.append((f, 1) if rng.random() < 0.5 else (intercambiar(f, rng), 0))
    return ejemplos


def dividir(ejemplos):
    n = len(ejemplos)
    a, b = int(0.8 * n), int(0.9 * n)
    return ejemplos[:a], ejemplos[a:b], ejemplos[b:]


def vocabulario(entrenamiento, min_frec=2):
    cuenta = collections.Counter(w for f, _ in entrenamiento for w in f)
    return {w: i + 3 for i, w in enumerate(sorted(w for w, k in cuenta.items() if k >= min_frec))}


def tensorizar(ejemplos, voc):
    X = torch.zeros(len(ejemplos), LARGO_MAX + 1, dtype=torch.long)
    y = torch.tensor([e[1] for e in ejemplos])
    for k, (f, _) in enumerate(ejemplos):
        X[k, 0] = 1
        for p, w in enumerate(f):
            X[k, p + 1] = voc.get(w, 2)
    return X.to(DISP), y.to(DISP)


class Clasificador(nn.Module):
    def __init__(self, n_vocab, usar_pe, causal, d=D_MODELO, capas=CAPAS_B, cabezas=CABEZAS):
        super().__init__()
        self.usar_pe, self.causal = usar_pe, causal
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        capa = nn.TransformerEncoderLayer(d, cabezas, 2 * d, dropout=0.1, batch_first=True)
        self.cod = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.salida = nn.Linear(d, 2)
        self.register_buffer("pe", pe_sinusoidal(LARGO_MAX + 1, d).float())

    def forward(self, x, escala_pe=1.0):
        relleno = x == 0
        L = x.shape[1]
        h = self.emb(x)
        if self.usar_pe:
            h = h + escala_pe * self.pe[:L]
        mascara = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1) if self.causal else None
        h = self.cod(h, mask=mascara, src_key_padding_mask=relleno)
        h = h.masked_fill(relleno[..., None], 0).sum(1) / (~relleno).sum(1, keepdim=True)
        return self.salida(h)


@torch.no_grad()
def diferencia_por_permutacion(modelo, X, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    Xp = X.clone()
    for k in range(len(X)):
        n = int((X[k] != 0).sum()) - 1
        idx = torch.randperm(n, generator=g) + 1
        Xp[k, 1:n + 1] = X[k, idx.to(X.device)]
    modelo.eval()
    return (modelo(X) - modelo(Xp)).abs().max().item()


@torch.no_grad()
def exactitud(modelo, datos, escala_pe=1.0):
    modelo.eval()
    X, y = datos
    return (modelo(X, escala_pe).argmax(1) == y).float().mean().item()


def entrenar(tarea, usar_pe, causal, semilla):
    torch.manual_seed(semilla)
    d = DATOS[tarea]
    modelo = Clasificador(len(d["voc"]) + 3, usar_pe, causal).to(DISP)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    X, y = d["train"]
    gen = torch.Generator().manual_seed(semilla)
    curva = []
    for ep in range(EPOCAS):
        modelo.train()
        orden = torch.randperm(len(y), generator=gen).to(DISP)
        total = 0.0
        for b in range(0, len(y), LOTE):
            i = orden[b:b + LOTE]
            perdida = F.cross_entropy(modelo(X[i]), y[i])
            opt.zero_grad()
            perdida.backward()
            opt.step()
            total += perdida.item() * len(i)
        curva.append({"epoca": ep + 1, "perdida": total / len(y), "acc_val": exactitud(modelo, d["val"])})
    resultado = {"tarea": tarea, "pe": usar_pe, "atencion": "causal" if causal else "bidireccional", "semilla": semilla,
                 "acc_test": exactitud(modelo, d["test"]),
                 "acc_test_quitando_pe": exactitud(modelo, d["test"], 0.0) if usar_pe else float("nan")}
    return resultado, curva, modelo


FRASE_A_B = tokenizar("El gato persiguió al perro")
FRASE_B_B = tokenizar("El perro persiguió al gato")


@torch.no_grad()
def vectores_salida(modelo, frase):
    X, _ = tensorizar([(tuple(frase), 1)], DATOS["orden"]["voc"])
    modelo.eval()
    relleno = X == 0
    L = X.shape[1]
    h = modelo.emb(X)
    if modelo.usar_pe:
        h = h + modelo.pe[:L]
    mascara = torch.triu(torch.ones(L, L, dtype=torch.bool, device=X.device), 1) if modelo.causal else None
    h = modelo.cod(h, mask=mascara, src_key_padding_mask=relleno)
    return h[0, 1:len(frase) + 1].cpu()
