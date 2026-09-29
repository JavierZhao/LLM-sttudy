"""Drill (page 02): byte-level BPE from scratch.

Implement the functions below. Run:
    pytest exercises/tests/test_bpe.py

Conventions (the tests rely on all of them):
- Token ids 0..255 are the 256 byte values. The i-th merge (0-based) creates token id 256 + i.
- A merge is a pair of existing token ids (id1, id2). Its bytes are bytes(id1) + bytes(id2).
- There is NO pre-tokenization here: the whole text is one UTF-8 byte sequence, so merges may
  cross word and punctuation boundaries. (Real tokenizers split with a regex first, see the page.)
- Pair counts include overlapping occurrences: in [a, a, a] the pair (a, a) occurs twice.
  Merging scans left to right without overlap: [a, a, a] with (a, a) -> new becomes [new, a].
"""
from __future__ import annotations

Pair = tuple[int, int]


def pair_counts(ids: list[int]) -> dict[Pair, int]:
    """Count adjacent pairs.

    Args:
        ids: token ids.
    Returns:
        Dict mapping (ids[i], ids[i+1]) to the number of positions i where it occurs
        (overlapping occurrences all count). Empty dict if len(ids) < 2.
    """
    raise NotImplementedError


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace occurrences of `pair` in `ids` by `new_id`.

    Scan left to right; once two tokens are merged, neither takes part in another merge of
    this call. Returns a new list (does not modify `ids`).
    Example: merge_pair([1, 1, 1], (1, 1), 9) == [9, 1];
             merge_pair([1, 2, 1, 2, 1], (1, 2), 9) == [9, 9, 1].
    """
    raise NotImplementedError


def train_bpe(text: str, vocab_size: int) -> list[Pair]:
    """Train byte-level BPE on `text` and return the merges in order.

    Start from the UTF-8 bytes of `text`. Repeat until the vocabulary has `vocab_size` entries
    (that is, vocab_size - 256 merges): count all adjacent pairs, take the most frequent one,
    give it the next free id, replace it everywhere.

    Deterministic tie rule: highest count wins; among equal counts the smallest (id1, id2)
    tuple wins.

    Args:
        text: training text.
        vocab_size: target size including the 256 byte tokens. Must be >= 256, otherwise
            raise ValueError. vocab_size == 256 returns [].
    Returns:
        merges[i] = the pair merged into token 256 + i. If fewer than 2 tokens remain before
        the target is reached (for example a very short text), stop early and return fewer
        merges. Pairs with count 1 are still merged.
    """
    raise NotImplementedError


def encode(text: str, merges: list[Pair]) -> list[int]:
    """Encode `text` with a trained merge list.

    Start from the UTF-8 bytes. Repeatedly find, among the adjacent pairs currently present,
    the one with the LOWEST merge index (rank) and merge all its occurrences (left to right,
    no overlap). Stop when no present pair is in `merges`.

    Returns:
        list of token ids. Never longer than len(text.encode("utf-8")). encode("") == [].
    """
    raise NotImplementedError


def decode(ids: list[int], merges: list[Pair]) -> str:
    """Decode token ids back to a string.

    Build the byte string of every token from `merges`, concatenate the bytes of `ids`, and
    decode as UTF-8 with errors="replace" (a model can emit a partial multi-byte sequence).
    """
    raise NotImplementedError


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character map.

    The 188 bytes that are already printable and not whitespace/control characters keep their
    own code point: '!'..'~' (33..126), '¡'..'¬' (161..172), '®'..'ÿ' (174..255).
    The other 68 bytes (whitespace, control characters, 127..160, 173) are assigned code points
    256, 257, ... in increasing order of the byte value.

    Returns:
        dict with 256 entries, byte value -> one-character str, all distinct.
        For example b2u[ord("A")] == "A" and b2u[32] == "Ġ" (U+0120).
    """
    raise NotImplementedError
