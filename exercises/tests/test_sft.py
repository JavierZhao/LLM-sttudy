import math
import re

import pytest
import torch
import torch.nn.functional as F

MODULE = "sft"

IM_START, IM_END = 1, 2
SPECIALS = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def toy_tokenize(text):
    """Special markers are single tokens; every other character is one token (id = ord + 100)."""
    out = []
    for piece in _SPECIAL_RE.split(text):
        if piece in SPECIALS:
            out.append(SPECIALS[piece])
        else:
            out.extend(ord(c) + 100 for c in piece)
    return out


def chars(s):
    return [ord(c) + 100 for c in s]


def make_template(impl, **kw):
    return impl.ChatTemplate(
        header={"system": "<|im_start|>system\n", "user": "<|im_start|>user\n", "assistant": "<|im_start|>assistant\n"},
        end="<|im_end|>", sep="\n", **kw)


CONVO = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "and 3+3?"},
    {"role": "assistant", "content": "six"},
]


def trained_ids(ids, mask):
    return [t for t, m in zip(ids, mask) if m]


# ---------------------------------------------------------------- build_example
def test_build_example_matches_whole_string_tokenization(impl):
    tpl = make_template(impl)
    ids, mask = impl.build_example(CONVO, toy_tokenize, tpl)
    full = "".join(tpl.header[m["role"]] + m["content"] + tpl.end + tpl.sep for m in CONVO)
    assert ids == toy_tokenize(full)
    assert len(mask) == len(ids)


def test_build_example_trains_assistant_content_and_end_token_only(impl):
    tpl = make_template(impl)
    ids, mask = impl.build_example(CONVO, toy_tokenize, tpl)
    assert trained_ids(ids, mask) == chars("4") + [IM_END] + chars("six") + [IM_END]
    # headers (starting with <|im_start|>), the separator newline after <|im_end|>, and other roles are masked
    for i, (t, m) in enumerate(zip(ids, mask)):
        if t == IM_START:
            assert not m
        if i > 0 and ids[i - 1] == IM_END:
            assert not m                      # the "\n" after the end token is never a target


def test_build_example_train_on_last_only(impl):
    tpl = make_template(impl)
    ids, mask = impl.build_example(CONVO, toy_tokenize, tpl, train_on="last")
    assert trained_ids(ids, mask) == chars("six") + [IM_END]
    ids2, mask2 = impl.build_example(CONVO[:-1], toy_tokenize, tpl, train_on="last")   # last message is a user turn
    assert not any(mask2)
    with pytest.raises(ValueError):
        impl.build_example(CONVO, toy_tokenize, tpl, train_on="first")


def test_build_example_bos_and_edge_cases(impl):
    tpl = make_template(impl, bos="<|im_end|>")            # a recognizable one-token bos
    ids, mask = impl.build_example(CONVO[1:3], toy_tokenize, tpl)
    assert ids[0] == IM_END and mask[0] is False           # bos is prepended once and never trained
    assert sum(1 for t in ids if t == IM_END) == 3         # bos + user end + assistant end
    # empty assistant content: no content tokens, but the end token is still a target
    ids, mask = impl.build_example([{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
                                   toy_tokenize, make_template(impl))
    assert trained_ids(ids, mask) == [IM_END]
    # only a user turn: nothing to learn from
    ids, mask = impl.build_example([{"role": "user", "content": "hi"}], toy_tokenize, make_template(impl))
    assert len(ids) == len(mask) and not any(mask)


# ---------------------------------------------------------------- pack / document mask
def _ex(n, base, trained_from=None):
    ids = list(range(base, base + n))
    mask = [i >= (trained_from if trained_from is not None else n) for i in range(n)]
    return ids, mask


def test_pack_next_fit_order_preserved(impl):
    exs = [_ex(5, 10), _ex(4, 20), _ex(3, 30), _ex(2, 40), _ex(6, 50)]
    rows = impl.pack(exs, max_len=8, pad_id=-7)
    assert len(rows) == 3                                   # first-fit would give [5,3] [4,2] [6]; next-fit does not
    assert [int((r.doc_ids >= 0).sum()) for r in rows] == [5, 7, 8]
    assert rows[1].input_ids.tolist() == [20, 21, 22, 23, 30, 31, 32, -7]
    assert rows[2].input_ids.tolist() == [40, 41, 50, 51, 52, 53, 54, 55]
    for r in rows:
        assert r.input_ids.shape == r.loss_mask.shape == r.position_ids.shape == r.doc_ids.shape == (8,)
        assert r.input_ids.dtype == torch.long and r.loss_mask.dtype == torch.bool


def test_pack_positions_docs_and_masks(impl):
    a = ([1, 2, 3, 4], [False, False, True, True])
    b = ([5, 6, 7], [False, True, True])
    (row,) = impl.pack([a, b], max_len=9, pad_id=0)
    assert row.position_ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 0]       # reset per document, 0 on padding
    assert row.doc_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, -1, -1]
    assert row.loss_mask.tolist() == [False, False, True, True, False, True, True, False, False]
    assert row.input_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 0, 0]


def test_pack_errors_and_empties(impl):
    with pytest.raises(ValueError):
        impl.pack([_ex(5, 0)], max_len=4)
    assert impl.pack([], max_len=4) == []
    rows = impl.pack([([], []), _ex(2, 0)], max_len=4)
    assert len(rows) == 1 and rows[0].doc_ids.tolist() == [0, 0, -1, -1]


def test_document_causal_mask_is_block_diagonal_causal(impl):
    doc = torch.tensor([0, 0, 0, 1, 1, 2, -1, -1])
    m = impl.document_causal_mask(doc)
    T = doc.numel()
    assert m.shape == (T, T) and m.dtype == torch.bool
    for i in range(T):
        for j in range(T):
            assert bool(m[i, j]) == (j <= i and doc[i] == doc[j])
    assert m[3, 2].item() is False                          # second document cannot see the first
    assert m[4, 3].item() is True
    assert m.any(dim=-1).all()                              # no empty rows (padding sees padding)
    batched = impl.document_causal_mask(torch.stack([doc, torch.zeros_like(doc)]))
    assert batched.shape == (2, T, T)
    assert torch.equal(batched[0], m)
    assert torch.equal(batched[1], torch.ones(T, T, dtype=torch.bool).tril())


def test_packed_attention_equals_separate_documents(impl):
    # With the document mask, attention over a packed row must equal attention run on each document alone.
    lens = [5, 3, 4]
    T = sum(lens)
    q, k, v = (torch.randn(1, 2, T, 8) for _ in range(3))
    doc = torch.cat([torch.full((n,), d) for d, n in enumerate(lens)])
    mask = impl.document_causal_mask(doc)
    packed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    off = 0
    for n in lens:
        sl = slice(off, off + n)
        alone = F.scaled_dot_product_attention(q[:, :, sl], k[:, :, sl], v[:, :, sl], is_causal=True)
        torch.testing.assert_close(packed[:, :, sl], alone, atol=1e-5, rtol=1e-5)
        off += n
    naive = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # plain causal mask leaks across documents
    assert not torch.allclose(naive[:, :, lens[0]:], packed[:, :, lens[0]:], atol=1e-4)


# ---------------------------------------------------------------- packed_lm_loss
def _logits_for_nll(nll_per_target, V=2):
    """logits at position t such that the NLL of token id 0 equals nll_per_target[t]; V = 2."""
    a = -torch.log(torch.exp(torch.as_tensor(nll_per_target, dtype=torch.float64)) - 1.0)   # log(1 + e^-a) = nll
    logits = torch.zeros(len(nll_per_target), V, dtype=torch.float64)
    logits[:, 0] = a
    return logits


def test_token_vs_sequence_normalization_worked_example(impl):
    # three documents: 4, 6 and 90 trained tokens with mean NLL 0.5, 1.0, 0.4 (each doc has 2 masked prompt tokens)
    spec = [(4, 0.5), (6, 1.0), (90, 0.4)]
    ids, doc, mask, tgt_nll = [], [], [], []
    for d, (n, m) in enumerate(spec):
        ids += [0] * (2 + n)
        doc += [d] * (2 + n)
        mask += [False] * 2 + [True] * n
        tgt_nll += [1.0] * 2 + [m] * n                       # NLL of predicting each token from its predecessor
    T = len(ids)
    nll_at_pos = tgt_nll[1:] + [1.0]                         # logits at position t predict token t + 1
    logits = _logits_for_nll(nll_at_pos).unsqueeze(0)
    ids_t = torch.tensor(ids).unsqueeze(0)
    mask_t = torch.tensor(mask).unsqueeze(0)
    doc_t = torch.tensor(doc).unsqueeze(0)
    tok = impl.packed_lm_loss(logits, ids_t, mask_t, doc_t, "token")
    seq = impl.packed_lm_loss(logits, ids_t, mask_t, doc_t, "sequence")
    assert math.isclose(tok.item(), 44.0 / 100.0, rel_tol=1e-9)              # (2 + 6 + 36) / 100
    assert math.isclose(seq.item(), (0.5 + 1.0 + 0.4) / 3, rel_tol=1e-9)     # 1.9 / 3
    with pytest.raises(ValueError):
        impl.packed_lm_loss(logits, ids_t, mask_t, doc_t, "mean")


def test_packed_lm_loss_never_predicts_across_document_boundary(impl):
    torch.manual_seed(1)
    B, T, V = 2, 9, 11
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 1, -1, -1]])
    mask = torch.ones(B, T, dtype=torch.bool)                # even document starts are (wrongly) marked as targets
    got = impl.packed_lm_loss(logits, ids, mask, doc, "token")
    total, count = 0.0, 0
    for b in range(B):
        for t in range(T - 1):
            if doc[b, t + 1] >= 0 and doc[b, t] == doc[b, t + 1]:
                total += F.cross_entropy(logits[b, t:t + 1], ids[b, t + 1:t + 2]).item()
                count += 1
    assert math.isclose(got.item(), total / count, rel_tol=1e-5)
    assert count == 6 + 5                                     # 2+3+1 targets in row 0, 1+4 in row 1
    # sequence mode: mean of per-document means, over rows and documents
    per_doc = {}
    for b in range(B):
        for t in range(T - 1):
            if doc[b, t + 1] >= 0 and doc[b, t] == doc[b, t + 1]:
                per_doc.setdefault((b, int(doc[b, t + 1])), []).append(
                    F.cross_entropy(logits[b, t:t + 1], ids[b, t + 1:t + 2]).item())
    expect = sum(sum(v) / len(v) for v in per_doc.values()) / len(per_doc)
    got_seq = impl.packed_lm_loss(logits, ids, mask, doc, "sequence")
    assert math.isclose(got_seq.item(), expect, rel_tol=1e-5)


def test_packed_lm_loss_empty_and_gradient(impl):
    logits = torch.randn(1, 5, 7, requires_grad=True)
    ids = torch.randint(0, 7, (1, 5))
    doc = torch.zeros(1, 5, dtype=torch.long)
    zero = impl.packed_lm_loss(logits, ids, torch.zeros(1, 5, dtype=torch.bool), doc, "token")
    assert zero.item() == 0.0
    loss = impl.packed_lm_loss(logits, ids, torch.tensor([[False, False, True, True, False]]), doc, "token")
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 4].abs().sum() == 0 and logits.grad[0, 0].abs().sum() == 0   # nothing predicts token 5 / masked
    assert logits.grad[0, 1].abs().sum() > 0 and logits.grad[0, 2].abs().sum() > 0     # positions 1, 2 predict trained tokens 2, 3


# ---------------------------------------------------------------- kd_loss
def _manual_kl(p, q):
    return float(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0))


def test_kd_hand_example_matches_the_page(impl):
    p = torch.tensor([0.48, 0.02, 0.02, 0.48]).log()          # teacher logits = log-probabilities (they normalize to 1)
    smeared = torch.full((4,), 0.25).log()
    collapsed = torch.tensor([0.94, 0.02, 0.02, 0.02]).log()
    assert math.isclose(impl.kd_loss(smeared, p, kind="forward").item(), 0.5252, abs_tol=5e-4)
    assert math.isclose(impl.kd_loss(smeared, p, kind="reverse").item(), 0.9367, abs_tol=5e-4)
    assert math.isclose(impl.kd_loss(collapsed, p, kind="forward").item(), 1.2029, abs_tol=5e-4)
    assert math.isclose(impl.kd_loss(collapsed, p, kind="reverse").item(), 0.5682, abs_tol=5e-4)
    # forward KL prefers the smeared student, reverse KL prefers the collapsed one (mode-seeking)
    assert impl.kd_loss(smeared, p, kind="forward") < impl.kd_loss(collapsed, p, kind="forward")
    assert impl.kd_loss(collapsed, p, kind="reverse") < impl.kd_loss(smeared, p, kind="reverse")


@pytest.mark.parametrize("kind", ["forward", "reverse"])
@pytest.mark.parametrize("T", [1.0, 2.0])
def test_kd_matches_manual_kl_with_mask_and_temperature(impl, kind, T):
    zs, zt = torch.randn(3, 4, 9), torch.randn(3, 4, 9)
    mask = torch.rand(3, 4) > 0.4
    mask[0, 0] = True
    got = impl.kd_loss(zs, zt, T=T, kind=kind, mask=mask)
    ps, pt = F.softmax(zs / T, -1), F.softmax(zt / T, -1)
    vals = []
    for b in range(3):
        for t in range(4):
            if mask[b, t]:
                vals.append(_manual_kl(pt[b, t].tolist(), ps[b, t].tolist()) if kind == "forward"
                            else _manual_kl(ps[b, t].tolist(), pt[b, t].tolist()))
    assert math.isclose(got.item(), T * T * sum(vals) / len(vals), rel_tol=1e-4)
    assert impl.kd_loss(zs, zt, T=T, kind=kind, mask=torch.zeros(3, 4, dtype=torch.bool)).item() == 0.0


@pytest.mark.parametrize("kind", ["forward", "reverse"])
def test_kd_zero_when_equal_and_nonnegative(impl, kind):
    z = torch.randn(5, 7)
    assert abs(impl.kd_loss(z, z.clone(), kind=kind).item()) < 1e-6
    assert impl.kd_loss(z, torch.randn(5, 7), kind=kind).item() > 0
    with pytest.raises(ValueError):
        impl.kd_loss(z, z, kind="jsd")


def test_kd_gradients(impl):
    zs = torch.randn(6, 8, requires_grad=True)
    zt = torch.randn(6, 8, requires_grad=True)
    N = zs.shape[0]
    q, p = F.softmax(zs.detach(), -1), F.softmax(zt.detach(), -1)
    # forward KL at T = 1: d/dz_s = (q - p) / N   (soft-label cross-entropy)
    impl.kd_loss(zs, zt, kind="forward").backward()
    torch.testing.assert_close(zs.grad, (q - p) / N, atol=1e-6, rtol=1e-4)
    assert zt.grad is None or zt.grad.abs().sum() == 0        # the teacher is a constant
    # reverse KL at T = 1: d/dz_s = q * (log(q/p) - KL) / N
    zs.grad = None
    impl.kd_loss(zs, zt, kind="reverse").backward()
    ell = (q / p).log()
    kl = (q * ell).sum(-1, keepdim=True)
    torch.testing.assert_close(zs.grad, q * (ell - kl) / N, atol=1e-6, rtol=1e-4)


def test_kd_temperature_scale(impl):
    zt = torch.tensor([[4.0, 2.0, 0.0, -2.0]])
    zs = torch.tensor([[3.0, 3.0, 0.0, -3.0]])
    vals = {T: impl.kd_loss(zs, zt, T=T, kind="forward").item() for T in (1.0, 2.0, 4.0)}
    assert math.isclose(vals[1.0], 0.3239, abs_tol=5e-4)
    assert math.isclose(vals[2.0], 0.4017, abs_tol=5e-4)      # raw KL is 0.1004; times T^2 = 4
    assert math.isclose(vals[4.0], 0.3902, abs_tol=5e-4)      # raw KL is 0.0244; times T^2 = 16
