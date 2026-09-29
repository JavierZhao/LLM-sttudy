import random

import numpy as np

MODULE = "data_pipeline"

# ---------------------------------------------------------------------------------------------
# helpers: deterministic synthetic documents (no external data, fixed seeds)
# ---------------------------------------------------------------------------------------------

PROSE = (
    "The committee met on Tuesday to discuss the future of the old railway station, which has stood "
    "empty for nearly two decades. Several residents argued that the building should be restored and "
    "turned into a museum, while others preferred to sell the land to developers who promised new "
    "housing. After a long debate, the members agreed to commission a study of the costs and to "
    "report back to the public in the spring. Whatever they decide, the station will remain a "
    "reminder of the days when the town depended on the railway for its trade and its growth."
)


def _vocab(prefix: str, size: int = 400) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(size)]


def _random_doc(seed: int, prefix: str, n_words: int = 200) -> str:
    rng = random.Random(seed)
    words = _vocab(prefix)
    return " ".join(rng.choice(words) for _ in range(n_words))


def _perturb(doc: str, n_changes: int, seed: int) -> str:
    rng = random.Random(seed)
    words = doc.split()
    for pos in rng.sample(range(len(words)), n_changes):
        words[pos] = "zzchanged" + str(pos)
    return " ".join(words)


# ---------------------------------------------------------------------------------------------
# jaccard / shingles
# ---------------------------------------------------------------------------------------------


def test_jaccard_basic(impl):
    assert impl.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert impl.jaccard({1, 2}, {1, 2}) == 1.0
    assert impl.jaccard({1}, {2}) == 0.0
    assert impl.jaccard(set(), set()) == 1.0
    assert impl.jaccard(set(), {1}) == 0.0


def test_shingles_exact(impl):
    got = impl.shingles("The quick brown fox jumps.", 2)
    assert got == {"the quick", "quick brown", "brown fox", "fox jumps"}
    # case and punctuation are normalized, repeated shingles collapse in the set
    assert impl.shingles("a b, A B; a b", 2) == {"a b", "b a"}
    assert impl.shingles("one two three", 3) == {"one two three"}


def test_shingles_edge_cases(impl):
    assert impl.shingles("", 3) == set()
    assert impl.shingles("   ...  ", 3) == set()
    assert impl.shingles("hello world", 5) == {"hello world"}  # fewer than n words: one shingle
    assert impl.shingles("x y z", 1) == {"x", "y", "z"}
    try:
        impl.shingles("x y z", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("n = 0 must raise ValueError")


# ---------------------------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------------------------


def test_minhash_shape_dtype_determinism(impl):
    s = {f"s{i}" for i in range(50)}
    a = impl.minhash_signature(s, 64, seed=3)
    b = impl.minhash_signature(set(s), 64, seed=3)
    c = impl.minhash_signature(s, 64, seed=4)
    assert a.shape == (64,) and a.dtype == np.uint64
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert int(a.max()) < impl.MERSENNE_P


def test_minhash_empty_set(impl):
    sig = impl.minhash_signature(set(), 16, seed=0)
    assert sig.shape == (16,)
    assert np.all(sig == impl.MERSENNE_P)


def test_minhash_agreement_estimates_jaccard(impl):
    n_perm = 1024  # standard error sqrt(J(1-J)/1024) <= 0.016
    a = {f"s{i}" for i in range(100)}
    cases = [
        ({f"s{i}" for i in range(50, 150)}, 50 / 150),   # J = 1/3
        ({f"s{i}" for i in range(10, 110)}, 90 / 110),   # J = 0.818...
        ({f"t{i}" for i in range(100)}, 0.0),            # disjoint
        (set(a), 1.0),                                   # identical
    ]
    sig_a = impl.minhash_signature(a, n_perm, seed=7)
    for other, true_j in cases:
        assert abs(impl.jaccard(a, other) - true_j) < 1e-12
        sig_b = impl.minhash_signature(other, n_perm, seed=7)
        est = float(np.mean(sig_a == sig_b))
        assert abs(est - true_j) < 0.08, (true_j, est)


# ---------------------------------------------------------------------------------------------
# LSH
# ---------------------------------------------------------------------------------------------


def test_collision_probability_formula(impl):
    f = impl.lsh_collision_probability
    assert f(0.0, 20, 5) == 0.0
    assert f(1.0, 20, 5) == 1.0
    assert abs(f(0.5, 20, 5) - (1 - (1 - 0.5**5) ** 20)) < 1e-12
    assert abs(f(0.5, 20, 5) - 0.470051) < 1e-5
    assert abs(f(0.75, 14, 8) - 0.771634) < 1e-5  # the FineWeb setting
    xs = [i / 20 for i in range(21)]
    ps = [f(x, 14, 8) for x in xs]
    assert all(p <= q + 1e-15 for p, q in zip(ps, ps[1:]))  # monotone S-curve


def test_lsh_candidates_semantics(impl):
    sigs = np.array(
        [
            [1, 2, 3, 4, 5, 6],   # 0
            [1, 2, 9, 9, 9, 9],   # 1: shares only columns 0-1 with doc 0
            [7, 2, 3, 8, 8, 8],   # 2: shares single columns with doc 0 but never a whole band
            [0, 0, 0, 4, 5, 6],   # 3: shares columns 3-5 with doc 0
            [5, 5, 5, 5, 5, 5],   # 4: agrees with nobody on a whole band
        ],
        dtype=np.uint64,
    )
    # bands = 2, rows = 3: band 0 = columns 0-2, band 1 = columns 3-5. Only docs 0 and 3 agree on a whole band.
    assert impl.lsh_candidates(sigs, bands=2, rows=3) == {(0, 3)}
    # bands = 3, rows = 2: bands are columns 0-1, 2-3, 4-5. Docs 0 and 1 agree on the first band, docs 0 and 3 on the last.
    got2 = impl.lsh_candidates(sigs, bands=3, rows=2)
    assert got2 == {(0, 1), (0, 3)}
    assert impl.lsh_candidates(sigs[:1], 2, 3) == set()
    assert impl.lsh_candidates(sigs[:0], 2, 3) == set()
    assert all(i < j for i, j in got2)
    # transitivity is NOT the job of this function: three docs sharing one bucket give all three pairs
    triple = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint64)
    assert impl.lsh_candidates(triple, 1, 2) == {(0, 1), (0, 2), (1, 2)}


def test_lsh_candidates_ignores_extra_columns_and_validates(impl):
    sigs = np.array([[1, 2, 3, 100], [1, 2, 3, 200]], dtype=np.uint64)
    assert impl.lsh_candidates(sigs, bands=1, rows=3) == {(0, 1)}   # column 3 unused
    assert impl.lsh_candidates(sigs, bands=1, rows=4) == set()
    try:
        impl.lsh_candidates(sigs, bands=3, rows=2)
    except ValueError:
        pass
    else:
        raise AssertionError("n_perm < bands * rows must raise ValueError")


def test_lsh_candidate_rate_follows_s_curve(impl):
    # Build pairs of "signatures" whose columns agree independently with probability s, so the
    # candidate rate must equal 1 - (1 - s^rows)^bands. This checks banding without any MinHash.
    bands, rows, n_pairs = 20, 5, 2000
    rng = np.random.default_rng(0)
    for s, lo, hi in [(0.3, 0.0475 - 0.03, 0.0475 + 0.03), (0.5, 0.47 - 0.05, 0.47 + 0.05), (0.8, 0.99, 1.0)]:
        a = rng.integers(0, 2**40, size=(n_pairs, bands * rows), dtype=np.uint64)
        b = rng.integers(0, 2**40, size=(n_pairs, bands * rows), dtype=np.uint64)
        agree = rng.random((n_pairs, bands * rows)) < s
        b = np.where(agree, a, b)
        sigs = np.empty((2 * n_pairs, bands * rows), dtype=np.uint64)
        sigs[0::2], sigs[1::2] = a, b
        cands = impl.lsh_candidates(sigs, bands, rows)
        rate = sum((2 * k, 2 * k + 1) in cands for k in range(n_pairs)) / n_pairs
        assert lo <= rate <= hi, (s, rate)


def test_near_duplicates_become_candidates_unrelated_do_not(impl):
    base = _random_doc(0, "a")
    near = _perturb(base, 2, seed=1)                # 2 of 200 words changed: 3-shingle Jaccard ~ 0.9
    other1 = _random_doc(1, "b")
    other2 = _random_doc(2, "c")
    docs = [base, near, other1, other2]
    n_perm, bands, rows = 100, 20, 5
    sets = [impl.shingles(d, 3) for d in docs]
    assert impl.jaccard(sets[0], sets[1]) > 0.85
    sigs = np.stack([impl.minhash_signature(s, n_perm, seed=42) for s in sets])
    cands = impl.lsh_candidates(sigs, bands, rows)
    assert (0, 1) in cands
    assert not any(pair in cands for pair in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def test_lsh_pipeline_on_prose(impl):
    variant = PROSE.replace("Tuesday", "Wednesday")  # one word changed in ~100
    unrelated = _random_doc(5, "q", n_words=120)
    sets = [impl.shingles(t, 2) for t in (PROSE, variant, unrelated)]
    sigs = np.stack([impl.minhash_signature(s, 128, seed=0) for s in sets])
    cands = impl.lsh_candidates(sigs, bands=32, rows=4)
    assert (0, 1) in cands and (0, 2) not in cands and (1, 2) not in cands


# ---------------------------------------------------------------------------------------------
# choosing bands and rows
# ---------------------------------------------------------------------------------------------


def _err(impl, n_perm, t, b, r, wfp, wfn):
    fine = np.linspace(0, 1, 4001)
    p = 1 - (1 - fine**r) ** b
    below, above = fine <= t, fine >= t
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return wfp * trapz(p[below], fine[below]) + wfn * trapz(1 - p[above], fine[above])


def test_choose_bands_rows_is_near_optimal(impl):
    for n_perm, t in [(64, 0.6), (100, 0.5), (60, 0.8)]:
        b, r = impl.choose_bands_rows(n_perm, t)
        assert b >= 1 and r >= 1 and b * r <= n_perm
        best = min(
            _err(impl, n_perm, t, bb, rr, 0.5, 0.5)
            for bb in range(1, n_perm + 1)
            for rr in range(1, n_perm // bb + 1)
        )
        assert _err(impl, n_perm, t, b, r, 0.5, 0.5) <= best * 1.03 + 1e-4, (n_perm, t, b, r)


def test_choose_bands_rows_shape_of_curve(impl):
    f = impl.lsh_collision_probability
    for n_perm, t in [(128, 0.5), (128, 0.8), (256, 0.9)]:
        b, r = impl.choose_bands_rows(n_perm, t)
        assert f(min(1.0, t + 0.2), b, r) > 0.9
        assert f(max(0.0, t - 0.2), b, r) < 0.1


def test_choose_bands_rows_weights_move_the_threshold(impl):
    # Penalizing false positives more should give a stricter S-curve: larger implied threshold (1/b)^(1/r).
    def implied(b, r):
        return (1.0 / b) ** (1.0 / r)

    strict = implied(*impl.choose_bands_rows(128, 0.6, fp_weight=0.9, fn_weight=0.1))
    lenient = implied(*impl.choose_bands_rows(128, 0.6, fp_weight=0.1, fn_weight=0.9))
    assert strict > lenient


# ---------------------------------------------------------------------------------------------
# Gopher quality filter
# ---------------------------------------------------------------------------------------------


def test_gopher_accepts_normal_prose(impl):
    assert len(PROSE.split()) >= 50
    assert impl.gopher_quality_filter(PROSE) is True
    two = PROSE + "\n\n" + PROSE.replace("Tuesday", "Friday")
    assert impl.gopher_quality_filter(two) is True


def test_gopher_rejects_boilerplate(impl):
    nav = " | ".join(["Home", "About us", "Products", "Contact", "Login", "Register", "Cart"] * 10)
    assert impl.gopher_quality_filter(nav) is False        # no stop words, mostly symbols/short words
    assert impl.gopher_quality_filter("Click here to read more") is False   # far too short
    assert impl.gopher_quality_filter("") is False
    assert impl.gopher_quality_filter("   \n  ") is False


def test_gopher_word_count_range(impl):
    unit = "the farmers have planted wheat and barley in the fields "  # 10 words, stop words present
    assert len(unit.split()) == 10
    assert impl.gopher_quality_filter(unit * 4) is False   # 40 words
    assert impl.gopher_quality_filter(unit * 5) is True    # 50 words: lower bound is inclusive
    assert impl.gopher_quality_filter(unit * 10_000) is True    # 100,000 words: upper bound is inclusive
    assert impl.gopher_quality_filter(unit * 10_001) is False


def test_gopher_mean_word_length(impl):
    short = "a b c the of " * 20                          # mean word length ~ 1.6
    assert impl.gopher_quality_filter(short) is False
    long_words = "the internationalization of institutionalization and characteristically " * 10
    assert impl.gopher_quality_filter(long_words) is False   # mean length > 10


def test_gopher_symbol_ratios(impl):
    body = PROSE.split()
    with_hashes = " ".join(w + " #" if i % 5 == 0 else w for i, w in enumerate(body))
    assert impl.gopher_quality_filter(with_hashes) is False   # '#' / words ~ 0.2
    sparse = PROSE + " #once"
    assert impl.gopher_quality_filter(sparse) is True
    dots = " ".join(w + "..." if i % 4 == 0 else w for i, w in enumerate(body))
    assert impl.gopher_quality_filter(dots) is False          # ellipsis / words ~ 0.25
    unicode_dots = " ".join(w + "…" if i % 4 == 0 else w for i, w in enumerate(body))
    assert impl.gopher_quality_filter(unicode_dots) is False


def test_gopher_line_rules(impl):
    sentences = [s.strip() + "." for s in PROSE.split(".") if s.strip()]
    bullets = "\n".join("- " + s for s in sentences)
    assert impl.gopher_quality_filter(bullets) is False       # 100% of lines are bullets
    mixed = "\n".join(("- " + s) if i == 0 else s for i, s in enumerate(sentences))
    assert impl.gopher_quality_filter(mixed) is True          # 1 of 4 lines
    trailing = "\n".join(s[:-1] + "..." for s in sentences)
    assert impl.gopher_quality_filter(trailing) is False      # every line ends with an ellipsis
    partly = "\n".join((s[:-1] + "...") if i == 0 else s for i, s in enumerate(sentences))
    assert impl.gopher_quality_filter(partly) is True         # 1 of 4 = 25% <= 30%


def test_gopher_alphabetic_fraction_and_stop_words(impl):
    numbers = " ".join(str(1000 + i) for i in range(80)) + " the and of the"
    assert impl.gopher_quality_filter(numbers) is False       # < 80% of words have a letter
    keyword_salad = ("cheap watches online discount price shipping worldwide bestseller quality " * 8).strip()
    assert impl.gopher_quality_filter(keyword_salad) is False  # no stop words at all
    one_stop = ("cheap watches online discount price shipping worldwide bestseller quality " * 8) + "the the the"
    assert impl.gopher_quality_filter(one_stop) is False       # only ONE distinct stop word
    two_stop = ("cheap watches online discount price shipping worldwide bestseller quality " * 8) + "the (and) the"
    assert impl.gopher_quality_filter(two_stop) is True        # punctuation stripped, two distinct stop words
