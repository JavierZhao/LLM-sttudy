"""Reference solutions: web-data pipeline primitives (page 12)."""
from __future__ import annotations

import hashlib
import random
import re

import numpy as np

MERSENNE_P = (1 << 61) - 1  # prime modulus for the universal hash family


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def shingles(text: str, n: int) -> set[str]:
    if n < 1:
        raise ValueError("n must be >= 1")
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}  # a short document is one shingle
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _stable_hash(s: str) -> int:
    # blake2b is deterministic across processes; Python's hash() is salted per process.
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") % MERSENNE_P


def minhash_signature(shingle_set: set[str], n_perm: int, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)  # same seed -> same permutations for every document
    a = [rng.randrange(1, MERSENNE_P) for _ in range(n_perm)]
    b = [rng.randrange(0, MERSENNE_P) for _ in range(n_perm)]
    if not shingle_set:
        return np.full(n_perm, MERSENNE_P, dtype=np.uint64)
    xs = [_stable_hash(s) for s in shingle_set]
    # Python ints: a_i * x can reach 2^122, so uint64 arithmetic would overflow.
    sig = [min((ai * x + bi) % MERSENNE_P for x in xs) for ai, bi in zip(a, b)]
    return np.array(sig, dtype=np.uint64)


def lsh_collision_probability(s: float, bands: int, rows: int) -> float:
    # P[a band of `rows` minhashes all agree] = s^rows; P[at least one of `bands` agrees] = 1 - (1 - s^rows)^bands
    return 1.0 - (1.0 - s**rows) ** bands


def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    signatures = np.asarray(signatures)
    if signatures.ndim != 2 or signatures.shape[1] < bands * rows:
        raise ValueError("need signatures of shape (N, n_perm) with n_perm >= bands * rows")
    n = signatures.shape[0]
    pairs: set[tuple[int, int]] = set()
    for k in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = np.ascontiguousarray(signatures[:, k * rows:(k + 1) * rows])
        for i in range(n):
            buckets.setdefault(block[i].tobytes(), []).append(i)  # the band content is the bucket key
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))  # members are appended in increasing i
    return pairs


def choose_bands_rows(
    n_perm: int, threshold: float, fp_weight: float = 0.5, fn_weight: float = 0.5
) -> tuple[int, int]:
    trapz = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2.0
    lo = np.linspace(0.0, threshold, 401)
    hi = np.linspace(threshold, 1.0, 401)
    best, best_err = (1, 1), float("inf")
    for bands in range(1, n_perm + 1):
        for rows in range(1, n_perm // bands + 1):
            fp = trapz(1 - (1 - lo**rows) ** bands, lo)   # false-positive area: P(s) for s below the threshold
            fn = trapz((1 - hi**rows) ** bands, hi)       # false-negative area: 1 - P(s) for s above it
            err = fp_weight * fp + fn_weight * fn
            if err < best_err:
                best, best_err = (bands, rows), err
    return best


_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}
_BULLETS = ("•", "●", "-", "*")
_ELLIPSES = ("...", "…")


def gopher_quality_filter(doc: str) -> bool:
    words = doc.split()
    n = len(words)
    if not 50 <= n <= 100_000:                                       # rule 1
        return False
    if not 3 <= sum(len(w) for w in words) / n <= 10:                # rule 2
        return False
    if doc.count("#") / n > 0.1:                                     # rule 3: hash symbols
        return False
    if (doc.count("...") + doc.replace("...", "").count("…")) / n > 0.1:  # rule 3: ellipses
        return False
    lines = [ln for ln in doc.splitlines() if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(_BULLETS) for ln in lines) / len(lines) > 0.9:   # rule 4
            return False
        if sum(ln.rstrip().endswith(_ELLIPSES) for ln in lines) / len(lines) > 0.3:    # rule 5
            return False
    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:    # rule 6
        return False
    stripped = {w.lower().strip(".,;:!?\"'()[]") for w in words}
    return len(stripped & _STOP) >= 2                                # rule 7
