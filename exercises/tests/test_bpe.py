import random

import pytest

MODULE = "bpe"

TEXTS = [
    "the quick brown fox jumps over the lazy dog. the dog barks; the fox runs away.",
    "Hello, 世界! 你好，世界。 Emoji: 😀😀🎉 café naïve über\n\ttabs and  double  spaces.",
    "低资源语言的分词器需要更多的词元。低资源语言的分词器需要更多的词元。",
]


def _ref_train(text, vocab_size):
    """Naive reference: recount everything at every step, tie -> smallest pair."""
    seq = list(text.encode("utf-8"))
    merges = []
    while 256 + len(merges) < vocab_size and len(seq) >= 2:
        counts = {}
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] = counts.get((seq[i], seq[i + 1]), 0) + 1
        top = max(counts.values())
        best = sorted(p for p, c in counts.items() if c == top)[0]
        new_id = 256 + len(merges)
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
        merges.append(best)
    return merges, seq


def test_pair_counts_and_merge_pair(impl):
    assert impl.pair_counts([]) == {}
    assert impl.pair_counts([7]) == {}
    assert impl.pair_counts([1, 1, 1]) == {(1, 1): 2}             # overlapping pairs count
    assert impl.pair_counts([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}
    assert impl.merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]         # left to right, no overlap
    assert impl.merge_pair([1, 2, 1, 2, 1], (1, 2), 9) == [9, 9, 1]
    assert impl.merge_pair([5, 6], (1, 2), 9) == [5, 6]
    ids = [1, 2, 3]
    impl.merge_pair(ids, (1, 2), 9)
    assert ids == [1, 2, 3]                                        # input not modified


def test_hand_verified_training(impl):
    # "aaabdaaabac": (a,a) x4 -> 256; then (a,b) and (256,a) tie at 2, smaller tuple (97,98) wins
    # -> 257; then (256,257) x2 -> 258.
    text = "aaabdaaabac"
    assert impl.train_bpe(text, 259) == [(97, 97), (97, 98), (256, 257)]
    # continuing: every remaining pair has count 1, so the smallest pair wins each time
    assert impl.train_bpe(text, 262) == [(97, 97), (97, 98), (256, 257), (97, 99), (100, 258), (258, 260)]


def test_hand_verified_encoding(impl):
    merges = impl.train_bpe("aaabdaaabac", 259)
    assert impl.encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]
    merges = impl.train_bpe("aaabdaaabac", 262)
    assert impl.encode("aaabdaaabac", merges) == [261, 259]


def test_tie_break_is_smallest_pair(impl):
    assert impl.train_bpe("abcd", 257) == [(97, 98)]               # all counts 1
    assert impl.train_bpe("abab", 258) == [(97, 98), (256, 256)]


def test_vocab_size_256_returns_no_merges(impl):
    assert impl.train_bpe("some text to train on", 256) == []
    assert impl.encode("héllo", []) == list("héllo".encode("utf-8"))
    with pytest.raises(ValueError):
        impl.train_bpe("abc", 255)


def test_early_stop_on_short_text(impl):
    assert impl.train_bpe("", 300) == []
    assert impl.train_bpe("a", 300) == []
    assert impl.train_bpe("ab", 300) == [(97, 98)]                 # then 1 token is left


@pytest.mark.parametrize("text", TEXTS)
def test_training_matches_naive_reference(impl, text):
    for vs in (256, 260, 300, 400):
        ref_merges, ref_seq = _ref_train(text, vs)
        merges = impl.train_bpe(text, vs)
        assert merges == ref_merges
        assert impl.encode(text, merges) == ref_seq                # encode replays training


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip_unicode(impl, text):
    merges = impl.train_bpe(" ".join(TEXTS), 320)
    ids = impl.encode(text, merges)
    assert impl.decode(ids, merges) == text
    assert impl.decode([], merges) == ""
    assert impl.encode("", merges) == []


def test_roundtrip_on_unseen_text(impl):
    merges = impl.train_bpe(TEXTS[0], 300)
    unseen = "Zürich 東京 🚀 never appeared in training, but bytes always work"
    assert impl.decode(impl.encode(unseen, merges), merges) == unseen


def test_encoding_never_longer_than_bytes_and_more_merges_never_hurt(impl):
    text = " ".join(TEXTS)
    n_bytes = len(text.encode("utf-8"))
    lengths = []
    for vs in (256, 270, 300, 350, 500):
        merges = impl.train_bpe(text, vs)
        lengths.append(len(impl.encode(text, merges)))
    assert all(n <= n_bytes for n in lengths)
    assert lengths == sorted(lengths, reverse=True)                # non-increasing
    assert lengths[0] == n_bytes and lengths[-1] < n_bytes


def test_encode_applies_lowest_rank_first(impl):
    assert impl.encode("abc", [(97, 98), (98, 99)]) == [256, 99]   # (a,b) has rank 0
    assert impl.encode("abc", [(98, 99), (97, 98)]) == [97, 256]   # (b,c) has rank 0
    assert impl.encode("aaaa", [(97, 97)]) == [256, 256]
    assert impl.encode("aaaa", [(97, 97), (256, 256)]) == [257]


def test_decode_partial_utf8_is_replaced_not_raised(impl):
    assert impl.decode([0xE4], []) == "�"                     # first byte of a 3-byte character
    assert impl.decode(list("世".encode("utf-8")), []) == "世"


def test_random_bytes_roundtrip(impl):
    rng = random.Random(0)
    text = "".join(rng.choice("abc def\n世界é😀") for _ in range(300))
    merges = impl.train_bpe(text, 290)
    assert impl.decode(impl.encode(text, merges), merges) == text


def test_bytes_to_unicode(impl):
    b2u = impl.bytes_to_unicode()
    assert sorted(b2u) == list(range(256))
    assert len(set(b2u.values())) == 256                           # reversible
    assert all(len(c) == 1 and c.isprintable() and not c.isspace() for c in b2u.values())
    assert sum(ord(c) == b for b, c in b2u.items()) == 188         # printable bytes map to themselves
    assert b2u[ord("A")] == "A" and b2u[ord("~")] == "~" and b2u[0xFF] == "ÿ"
    assert b2u[0] == "Ā" and b2u[32] == "Ġ" and b2u[10] == "Ċ"   # Ā, Ġ, Ċ
    assert b2u[127] == "ġ" and b2u[173] == "Ń"           # first and last shifted after 32
