"""RNN simple (tanh) y LSTM en numpy, con retropropagación a través del tiempo escrita a mano."""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def ortogonal(n, rng, escala=1.0):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return escala * q * np.sign(np.diag(r))


class RNNSimple:
    """h_t = tanh(x_t Wx + h_{t-1} Wh + b); logits = h_T Wy + by."""

    def __init__(self, n_in, n_oc, n_sal, rng, radio=1.0):
        self.p = {
            "Wx": rng.standard_normal((n_in, n_oc)) / np.sqrt(n_in),
            "Wh": ortogonal(n_oc, rng, radio),
            "b": np.zeros(n_oc),
            "Wy": rng.standard_normal((n_oc, n_sal)) / np.sqrt(n_oc),
            "by": np.zeros(n_sal),
        }
        self.n_oc = n_oc

    def paso(self, X, y, calcular_grad=True):
        p = self.p
        T, B, _ = X.shape
        hs = np.zeros((T + 1, B, self.n_oc))
        for t in range(T):
            hs[t + 1] = np.tanh(X[t] @ p["Wx"] + hs[t] @ p["Wh"] + p["b"])
        prob = softmax(hs[T] @ p["Wy"] + p["by"])
        perdida = -np.log(prob[np.arange(B), y] + 1e-12).mean()
        acc = (prob.argmax(1) == y).mean()
        if not calcular_grad:
            return perdida, acc, None, None
        d = prob.copy()
        d[np.arange(B), y] -= 1
        d /= B
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wy"] = hs[T].T @ d
        g["by"] = d.sum(0)
        dh = d @ p["Wy"].T
        normas = np.zeros(T)  # normas[t] = ||dL/dh_{t+1}|| (promedio por muestra)
        for t in range(T, 0, -1):
            normas[t - 1] = np.linalg.norm(dh, axis=1).mean()
            dz = dh * (1 - hs[t] ** 2)
            g["Wx"] += X[t - 1].T @ dz
            g["Wh"] += hs[t - 1].T @ dz
            g["b"] += dz.sum(0)
            dh = dz @ p["Wh"].T
        return perdida, acc, g, normas


class LSTM:
    """LSTM estándar con compuertas i, f, o y candidato g; sesgo de olvido configurable."""

    def __init__(self, n_in, n_oc, n_sal, rng, sesgo_olvido=1.0):
        H = n_oc
        b = np.zeros(4 * H)
        b[H:2 * H] = sesgo_olvido
        self.p = {
            "W": rng.standard_normal((n_in + H, 4 * H)) / np.sqrt(n_in + H),
            "b": b,
            "Wy": rng.standard_normal((H, n_sal)) / np.sqrt(H),
            "by": np.zeros(n_sal),
        }
        self.n_oc = H

    def paso(self, X, y, calcular_grad=True):
        p, H = self.p, self.n_oc
        T, B, D = X.shape
        hs = np.zeros((T + 1, B, H))
        cs = np.zeros((T + 1, B, H))
        cache = []
        for t in range(T):
            xh = np.concatenate([X[t], hs[t]], axis=1)
            a = xh @ p["W"] + p["b"]
            i, f, o = sigmoide(a[:, :H]), sigmoide(a[:, H:2 * H]), sigmoide(a[:, 2 * H:3 * H])
            g = np.tanh(a[:, 3 * H:])
            cs[t + 1] = f * cs[t] + i * g
            tc = np.tanh(cs[t + 1])
            hs[t + 1] = o * tc
            cache.append((xh, i, f, o, g, tc))
        prob = softmax(hs[T] @ p["Wy"] + p["by"])
        perdida = -np.log(prob[np.arange(B), y] + 1e-12).mean()
        acc = (prob.argmax(1) == y).mean()
        if not calcular_grad:
            return perdida, acc, None, None
        d = prob.copy()
        d[np.arange(B), y] -= 1
        d /= B
        g_ = {k: np.zeros_like(v) for k, v in p.items()}
        g_["Wy"] = hs[T].T @ d
        g_["by"] = d.sum(0)
        dh = d @ p["Wy"].T
        dc = np.zeros((B, H))
        normas = np.zeros(T)
        for t in range(T, 0, -1):
            normas[t - 1] = np.linalg.norm(dh, axis=1).mean()
            xh, i, f, o, g, tc = cache[t - 1]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, dg, df = dc * g, dc * i, dc * cs[t - 1]
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            g_["W"] += xh.T @ da
            g_["b"] += da.sum(0)
            dxh = da @ p["W"].T
            dh = dxh[:, D:]
            dc = dc * f
        return perdida, acc, g_, normas


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def actualizar(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def norma_global(grads):
    return float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))


def recortar(grads, umbral):
    n = norma_global(grads)
    if n > umbral:
        for k in grads:
            grads[k] = grads[k] * (umbral / n)
    return n
