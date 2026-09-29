"""Drill (page 12): web-data pipeline primitives (Gopher quality filter, MinHash, banded LSH).

Implement the functions below. Run:
    pytest exercises/tests/test_data_pipeline.py

Conventions (the tests rely on all of them):
- Documents are plain strings. "Words" for shingling are lowercase runs of Unicode word
  characters (letters, digits and underscore in any script;
  re.findall(r"\\w+", text.lower())). The Gopher filter uses its own, simpler word definition
  (whitespace split), documented on that function.
- Use only the standard library and numpy. Do not use Python's built-in hash() for strings:
  it is salted per process, so signatures would not be reproducible.
"""
from __future__ import annotations

import numpy as np

MERSENNE_P = (1 << 61) - 1  # prime modulus for the universal hash family


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity |a & b| / |a | b|.

    Two empty sets have similarity 1.0. If exactly one set is empty the similarity is 0.0.
    """
    raise NotImplementedError


def shingles(text: str, n: int) -> set[str]:
    """Word n-gram shingles of a document.

    Lowercase the text, extract words with re.findall(r"\\w+", ...), and return the set of all
    windows of n consecutive words, each joined with a single space.
    - If the document has at least one but fewer than n words, return {" ".join(words)}
      (a short document is a single shingle).
    - Empty text (no words) returns the empty set.
    - n must be >= 1, otherwise raise ValueError.
    """
    raise NotImplementedError


def minhash_signature(shingle_set: set[str], n_perm: int, seed: int = 0) -> np.ndarray:
    """MinHash signature of a shingle set.

    Returns an array of shape (n_perm,) and dtype uint64.

    Hash family: map each shingle s to an integer
        x = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") % MERSENNE_P
    and define permutation i by  h_i(x) = (a_i * x + b_i) % MERSENNE_P  with
    a_i in [1, MERSENNE_P - 1] and b_i in [0, MERSENNE_P - 1] drawn from a generator seeded with
    `seed` (the SAME (a_i, b_i) must be used for every document when `seed` is the same; that is what
    makes signatures of different documents comparable). Use exact integer arithmetic: a_i * x can
    exceed 64 bits.

    signature[i] = min over all shingles of h_i(x). For an empty shingle set every entry is MERSENNE_P.
    """
    raise NotImplementedError


def lsh_collision_probability(s: float, bands: int, rows: int) -> float:
    """Probability that two documents with Jaccard similarity s become a candidate pair
    under banded LSH with `bands` bands of `rows` rows each (the S-curve).
    """
    raise NotImplementedError


def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """Banded LSH over MinHash signatures.

    Args:
        signatures: (N, n_perm) integer array, one signature per row, n_perm >= bands * rows.
            Only the first bands * rows columns are used; band k covers columns
            [k * rows, (k + 1) * rows).
        bands, rows: LSH parameters.
    Returns:
        The set of pairs (i, j), i < j, such that documents i and j agree on ALL rows of at least
        one band. Do not compare all N^2 pairs: bucket documents by band content and pair up
        only documents that share a bucket. N = 0 or 1 returns the empty set.
    Raises:
        ValueError if n_perm < bands * rows.
    """
    raise NotImplementedError


def choose_bands_rows(
    n_perm: int, threshold: float, fp_weight: float = 0.5, fn_weight: float = 0.5
) -> tuple[int, int]:
    """Pick (bands, rows) with bands * rows <= n_perm for a target Jaccard threshold.

    With P(s) = lsh_collision_probability(s, bands, rows), minimize
        fp_weight * FP + fn_weight * FN,
        FP = integral_0^threshold P(s) ds        (pairs below the threshold that still collide)
        FN = integral_threshold^1 (1 - P(s)) ds  (pairs above the threshold that are missed)
    over all bands >= 1, rows >= 1 with bands * rows <= n_perm. Integrate numerically
    (for example the trapezoid rule on at least 200 grid points per side).
    """
    raise NotImplementedError


def gopher_quality_filter(doc: str) -> bool:
    """Return True if the document passes the Gopher heuristic quality filters (keep it).

    Rules (Rae et al. 2021, Appendix A.1 "Quality Filtering"). Let words = doc.split()
    (whitespace tokens, punctuation attached). A document is REJECTED if any of these fails:
      1. number of words is not in [50, 100_000];
      2. mean word length (characters per whitespace token) is not in [3, 10];
      3. (count of "#" characters) / (number of words) > 0.1, or
         (count of ellipses) / (number of words) > 0.1, where an ellipsis is "..." or the single
         character "…", counted without overlap;
      4. more than 90% of the non-empty lines start with a bullet (after lstrip, one of
         "•", "●", "-", "*");
      5. more than 30% of the non-empty lines end with an ellipsis (after rstrip, "..." or "…");
      6. fewer than 80% of the words contain at least one alphabetic character (str.isalpha on
         any character);
      7. fewer than 2 DISTINCT stop words from {the, be, to, of, and, that, have, with} occur among
         the words (compare lowercased words with the surrounding characters .,;:!?"'()[] stripped).
    A whitespace-only or empty document is rejected. Rules 4 and 5 are evaluated over
    doc.splitlines() after dropping lines that are empty once stripped.
    """
    raise NotImplementedError
