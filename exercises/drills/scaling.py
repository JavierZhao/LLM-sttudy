"""Drill (page 13): fit a scaling law and compute the compute-optimal allocation.

Implement the functions below. Run:
    pytest exercises/tests/test_scaling.py

Conventions: N is the parameter count, D the number of training tokens, C the training compute
in FLOPs with C = 6 * N * D, and the loss is the Chinchilla parametric form
L(N, D) = E + A / N**alpha + B / D**beta (natural-log cross entropy, nats per token).
All functions are pure NumPy/Python; no autograd is needed.
"""
import math
from typing import Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# Two published fits of the parametric form, for you to experiment with (the tests use their own copies).
HOFFMANN_2022 = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)          # Hoffmann et al., Approach 3 as printed
EPOCH_2024 = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)    # Besiroglu et al., replication


def chinchilla_loss(N: ArrayLike, D: ArrayLike, E: float, A: float, B: float,
                    alpha: float, beta: float) -> ArrayLike:
    """Parametric loss L(N, D) = E + A / N**alpha + B / D**beta.

    Args:
        N: parameter count(s), a float or an array of any shape.
        D: training token count(s), broadcastable against N.
        E, A, B, alpha, beta: fitted constants, all positive.
    Returns:
        The loss with the broadcast shape of N and D: a Python/NumPy float for scalar inputs,
        an ndarray otherwise. Inputs must not be modified.
    """
    raise NotImplementedError


def compute_optimal(C: float, E: float, A: float, B: float, alpha: float,
                    beta: float) -> Tuple[float, float]:
    """Closed-form minimizer of L(N, D) subject to 6 * N * D = C.

    Args:
        C: training compute in FLOPs (> 0).
        E, A, B, alpha, beta: constants of the parametric loss, all > 0. E does not
            influence the answer but is accepted so the signature matches chinchilla_loss.
    Returns:
        (N_opt, D_opt) as floats with 6 * N_opt * D_opt == C (to floating-point accuracy).
        No numerical optimization: evaluate the closed form.
    Raises:
        ValueError if C, A, B, alpha or beta is not positive.
    """
    raise NotImplementedError


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Least-squares fit of y = k * x**p, done as a straight line in log-log space.

    Args:
        x, y: equal-length sequences (lists or arrays) of positive numbers, at least 2 points.
    Returns:
        (p, k): the exponent p (negative for a decreasing law) and the coefficient k > 0, as
        Python floats. The fit minimizes the sum of squared errors of ln y against ln x.
    Raises:
        ValueError if the lengths differ, there are fewer than 2 points, or any value is <= 0.
    """
    raise NotImplementedError


def isoflop_optimum(sizes: Sequence[float], losses: Sequence[float]) -> Tuple[float, float]:
    """Estimate the loss-minimizing model size on one IsoFLOP curve (Chinchilla's Approach 2).

    Fit a parabola L = c2 * (ln N)**2 + c1 * ln N + c0 to the measured (N, L) pairs by least
    squares and return its vertex.

    Args:
        sizes: model sizes N (positive), any order, at least 3 distinct values.
        losses: final training losses, same length and order as sizes.
    Returns:
        (N_opt, L_min): N_opt = exp(-c1 / (2 * c2)) and L_min = the parabola's value at its vertex,
        both Python floats.
    Raises:
        ValueError if fewer than 3 points are given, the lengths differ, or the fitted
        parabola has no minimum (c2 <= 0).
    """
    raise NotImplementedError


def effective_tokens(unique_tokens: float, total_tokens: float,
                     r_star: float = 15.387756) -> float:
    """Effective data D' after repeating a corpus (Muennighoff et al., data-constrained scaling).

    With U = unique_tokens, D = total_tokens processed and R = D / U - 1 repetitions:
        D' = U + U * R_star * (1 - exp(-R / R_star)).
    Args:
        unique_tokens: size of the unique corpus U (> 0).
        total_tokens: total tokens processed D, counting repeats.
        r_star: the fitted decay constant R*_D (> 0); math.inf means repeated tokens are worth as much as fresh ones.
    Returns:
        D' as a float. If total_tokens <= unique_tokens there is no repetition and D' = total_tokens.
    """
    raise NotImplementedError


def inference_aware_optimum(loss: float, d_inf: float, E: float, A: float, B: float,
                            alpha: float, beta: float) -> Tuple[float, float]:
    """Cheapest (N, D_train) that reaches a target loss when the model will also serve d_inf tokens.

    Minimize the total FLOPs 6 * N * D_train + 2 * N * d_inf over N, where for each N the
    training tokens D_train are the ones that make chinchilla_loss(N, D_train, ...) equal
    `loss` exactly (Sardana and Frankle, 2023).

    Args:
        loss: target loss, must exceed E.
        d_inf: total tokens (prompt plus output) the model will process at inference, >= 0.
        E, A, B, alpha, beta: constants of the parametric loss, all > 0.
    Returns:
        (N, D_train) as floats. With d_inf = 0 this is the compute-optimal model for that loss.
        Larger d_inf must give a smaller N and a larger D_train.
    Raises:
        ValueError if loss <= E or any constant is not positive.
    Notes:
        There is no closed form for d_inf > 0; a robust 1-D search over ln N is expected.
        Only sizes with A / N**alpha < loss - E can reach the target at all.
    """
    raise NotImplementedError
