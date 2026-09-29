"""Reference solutions: scaling-law fitting and compute-optimal allocation (page 13)."""
import math
from typing import Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

HOFFMANN_2022 = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
EPOCH_2024 = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)


def chinchilla_loss(N: ArrayLike, D: ArrayLike, E: float, A: float, B: float,
                    alpha: float, beta: float) -> ArrayLike:
    """L(N, D) = E + A / N**alpha + B / D**beta (broadcasts over N and D)."""
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    out = E + A / N**alpha + B / D**beta
    return float(out) if out.ndim == 0 else out


def _check_positive(**kw) -> None:
    for name, v in kw.items():
        if not v > 0:
            raise ValueError(f"{name} must be positive, got {v}")


def compute_optimal(C: float, E: float, A: float, B: float, alpha: float,
                    beta: float) -> Tuple[float, float]:
    """Closed-form argmin of L(N, D) s.t. 6 N D = C.

    Substituting D = C / (6 N) and setting dL/dN = 0 gives alpha * A / N**alpha = beta * B / D**beta,
    which solves to N_opt = G (C/6)**a, D_opt = G**-1 (C/6)**b with
    G = (alpha A / (beta B))**(1 / (alpha + beta)), a = beta / (alpha + beta), b = alpha / (alpha + beta).
    """
    _check_positive(C=C, A=A, B=B, alpha=alpha, beta=beta)
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta / (alpha + beta)
    N = G * (C / 6.0) ** a
    D = C / (6.0 * N)          # enforces 6 N D = C exactly, and equals G**-1 (C/6)**b
    return float(N), float(D)


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """y = k x**p by least squares on (ln x, ln y); returns (p, k)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("x and y must be 1-D, equal length, with at least 2 points")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fitting needs strictly positive x and y")
    lx, ly = np.log(x), np.log(y)
    x0 = lx.mean()                                   # centering keeps the fit well conditioned
    p, c = np.polyfit(lx - x0, ly, 1)                # ln y = p (ln x - x0) + c
    k = math.exp(c - p * x0)
    return float(p), float(k)


def isoflop_optimum(sizes: Sequence[float], losses: Sequence[float]) -> Tuple[float, float]:
    """Vertex of the least-squares parabola in ln N through one IsoFLOP curve."""
    N = np.asarray(sizes, dtype=np.float64)
    L = np.asarray(losses, dtype=np.float64)
    if N.shape != L.shape or N.ndim != 1 or np.unique(N).size < 3:
        raise ValueError("need at least 3 distinct sizes, with one loss per size")
    lx = np.log(N)
    x0 = lx.mean()
    c2, c1, c0 = np.polyfit(lx - x0, L, 2)
    if not c2 > 0:
        raise ValueError("the fitted parabola has no minimum (c2 <= 0)")
    t = -c1 / (2.0 * c2)                             # vertex in the centered variable
    return float(math.exp(x0 + t)), float(c0 - c1 * c1 / (4.0 * c2))


def effective_tokens(unique_tokens: float, total_tokens: float,
                     r_star: float = 15.387756) -> float:
    """D' = U + U R* (1 - exp(-R / R*)), with R = D / U - 1 repetitions."""
    U, D = float(unique_tokens), float(total_tokens)
    if U <= 0:
        raise ValueError("unique_tokens must be positive")
    if not r_star > 0:
        raise ValueError("r_star must be positive (math.inf is allowed)")
    if D <= U or math.isinf(r_star):                 # no repeats, or repeats as good as fresh data
        return D
    R = D / U - 1.0
    return U + U * r_star * (1.0 - math.exp(-R / r_star))


def inference_aware_optimum(loss: float, d_inf: float, E: float, A: float, B: float,
                            alpha: float, beta: float) -> Tuple[float, float]:
    """Minimize 6 N D + 2 N d_inf subject to L(N, D) = loss, by a 1-D search over ln N."""
    _check_positive(A=A, B=B, alpha=alpha, beta=beta)
    if not loss > E:
        raise ValueError("target loss must exceed the irreducible loss E")
    if d_inf < 0:
        raise ValueError("d_inf must be non-negative")
    gap = loss - E
    N_min = (A / gap) ** (1.0 / alpha)               # below this size even infinite data cannot reach `loss`

    def d_train(N: float) -> float:
        r = gap - A / N**alpha                       # loss budget left for the data term
        return math.inf if r <= 0 else (B / r) ** (1.0 / beta)

    def cost(ln_n: float) -> float:
        N = math.exp(ln_n)
        D = d_train(N)
        return math.inf if math.isinf(D) else 6.0 * N * D + 2.0 * N * d_inf

    # With d_inf = 0 the optimum is the compute-optimal size, N_c = N_min (1 + alpha/beta)**(1/alpha)
    # (at the optimum B/D**beta = (alpha/beta) A/N**alpha). Inference cost only pushes N down,
    # so the answer lies in (N_min, N_c]; search slightly beyond N_c to be safe.
    lo = math.log(N_min) + 1e-9
    hi = math.log(N_min) + math.log(1.0 + alpha / beta) / alpha + 0.5
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = cost(c), cost(d)
    for _ in range(200):                             # golden-section search (cost is unimodal in ln N)
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = cost(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = cost(d)
    N = math.exp(0.5 * (lo + hi))
    return float(N), float(d_train(N))
