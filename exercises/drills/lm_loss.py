"""Drill (page 01): the language-modeling loss and its metrics.

Implement the four functions below. Run:
    pytest exercises/tests/test_lm_loss.py

Conventions used throughout: logits are (B, T, V) with position t predicting token t + 1,
labels are (B, T) int64, and the value -100 means "ignore this position".
"""
import math
from typing import Tuple

import torch
from torch import Tensor

IGNORE_INDEX = -100


def shift_for_lm(tokens: Tensor, pad_id: int) -> Tuple[Tensor, Tensor]:
    """Build (inputs, labels) for teacher-forced next-token prediction.

    Args:
        tokens: (B, T) int64 token ids, right-padded with ``pad_id``.
        pad_id: id used for padding. Padding is identified by value.
    Returns:
        inputs: (B, T) int64, identical to ``tokens`` (a copy, the argument must not be
            modified).
        labels: (B, T) int64 with ``labels[b, t] = tokens[b, t + 1]`` for ``t < T - 1``.
            The final position has no next token, so ``labels[:, T - 1] = -100``. Every
            label whose value equals ``pad_id`` is also set to -100.
    Notes:
        Shapes match the logits (B, T, V), so no further slicing is needed before the loss.
        ``T == 1`` is allowed: all labels are then -100.
    """
    raise NotImplementedError


def lm_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Mean negative log-likelihood over the non-ignored positions.

    Args:
        logits: (B, T, V) floating point (float32 or float64), position t scores token t + 1.
        labels: (B, T) int64 in [0, V) or equal to ``ignore_index``.
        ignore_index: label value that marks positions to drop from the average.
    Returns:
        A 0-dim tensor: sum of -log softmax(logits)[label] over positions whose label is not
        ``ignore_index``, divided by the number of such positions. The result must support
        ``backward()``. If every label is ignored, return a 0-dim zero tensor (still
        connected to ``logits`` so that ``backward()`` works and gives zero gradients). The
        result has the dtype of ``logits``.
    Constraints:
        Do not use ``F.cross_entropy``, ``F.nll_loss``, ``F.log_softmax``, ``Tensor.log_softmax``
        or ``torch.logsumexp`` (a test scans the module source for these names).
        Write the log-softmax yourself so that it is numerically stable for logits of magnitude
        1e4 (no overflow in exp, no underflow to log(0)).
    """
    raise NotImplementedError


def perplexity(mean_nll):
    """Perplexity = exp(mean negative log-likelihood in nats per token).

    Args:
        mean_nll: Python float or 0-dim tensor, the mean NLL in nats.
    Returns:
        The same kind of object (float in, float out; tensor in, tensor out).
    """
    raise NotImplementedError


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Tokenizer-independent score: compressed bits per raw UTF-8 byte of the evaluation text.

    Args:
        total_nll_nats: the SUM (not the mean) of the negative log-likelihood, in nats, that the
            model assigns to the whole evaluation text.
        n_bytes: length of that same text in UTF-8 bytes. Must be positive.
    Returns:
        total_nll_nats / (n_bytes * ln 2).
    Raises:
        ValueError: if ``n_bytes <= 0``.
    """
    raise NotImplementedError
