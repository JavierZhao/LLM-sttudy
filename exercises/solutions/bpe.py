"""Reference solutions: byte-level BPE (page 02)."""
from __future__ import annotations

Pair = tuple[int, int]


def pair_counts(ids: list[int]) -> dict[Pair, int]:
    counts: dict[Pair, int] = {}
    for p in zip(ids, ids[1:]):          # overlapping pairs all count
        counts[p] = counts.get(p, 0) + 1
    return counts


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2                       # skip both merged tokens: no overlapping merges
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, vocab_size: int) -> list[Pair]:
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 (the byte tokens)")
    ids = list(text.encode("utf-8"))
    merges: list[Pair] = []
    for new_id in range(256, vocab_size):
        counts = pair_counts(ids)
        if not counts:                   # fewer than 2 tokens left
            break
        # highest count first; ties -> smallest (id1, id2), so training is deterministic
        best = min(counts, key=lambda p: (-counts[p], p))
        ids = merge_pair(ids, best, new_id)
        merges.append(best)
    return merges


def encode(text: str, merges: list[Pair]) -> list[int]:
    rank = {p: i for i, p in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = pair_counts(ids)
        pair = min(counts, key=lambda p: rank.get(p, float("inf")))
        if pair not in rank:             # nothing mergeable is left
            break
        # a merge's new token id is 256 + its rank; tokens created later have higher ranks,
        # so taking the lowest rank each time replays the training merges in order
        ids = merge_pair(ids, pair, 256 + rank[pair])
    return ids


def decode(ids: list[int], merges: list[Pair]) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def bytes_to_unicode() -> dict[int, str]:
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
    mapping: dict[int, str] = {b: chr(b) for b in keep}
    n = 0
    for b in range(256):
        if b not in mapping:             # whitespace/control/unprintable: shift above 255
            mapping[b] = chr(256 + n)
            n += 1
    return mapping
