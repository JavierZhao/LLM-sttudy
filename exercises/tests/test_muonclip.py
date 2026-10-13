import math

import pytest
import torch

MODULE = "muonclip"

D, H, DH = 16, 4, 8      # model width, heads, head dim


def _mha_setup(scale_heads=(6.0, 1.0, 3.0, 0.5), B=2, T=7, dtype=torch.float64):
    """Random inputs and per-head weights; head h's weights are scaled so heads differ a lot in logit size."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, D, generator=g, dtype=dtype)
    Wq = torch.randn(H, D, DH, generator=g, dtype=dtype)
    Wk = torch.randn(H, D, DH, generator=g, dtype=dtype)
    s = torch.tensor(scale_heads, dtype=dtype).view(H, 1, 1)
    return x, Wq * s, Wk * s


def _qk(x, Wq, Wk):
    q = torch.einsum("btd,hde->bhte", x, Wq)
    k = torch.einsum("btd,hde->bhte", x, Wk)
    return q, k


def _brute_max(q, k, causal):
    B, Hh, T, d = q.shape
    out = torch.full((Hh,), -float("inf"), dtype=q.dtype)
    for b in range(B):
        for h in range(Hh):
            for i in range(T):
                for j in range(i + 1 if causal else T):
                    out[h] = max(out[h], (q[b, h, i] @ k[b, h, j]) / math.sqrt(d))
    return out


# ------------------------------------------------------------ max_attention_logit
@pytest.mark.parametrize("causal", [True, False])
def test_max_logit_matches_bruteforce(impl, causal):
    x, Wq, Wk = _mha_setup()
    q, k = _qk(x, Wq, Wk)
    torch.testing.assert_close(impl.max_attention_logit(q, k, causal=causal), _brute_max(q, k, causal))


def test_max_logit_is_signed_and_causal_mask_matters(impl):
    # every logit is negative: the max must be the least negative one, not the largest magnitude
    q = -torch.ones(1, 1, 3, 4, dtype=torch.float64)
    k = torch.ones(1, 1, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(impl.max_attention_logit(q, k), torch.tensor([-2.0], dtype=torch.float64))
    # a huge logit only on a future pair (i=0, j=2) must be ignored when causal
    q = torch.zeros(1, 1, 3, 2, dtype=torch.float64)
    k = torch.zeros(1, 1, 3, 2, dtype=torch.float64)
    q[0, 0, 0] = torch.tensor([10.0, 0.0])
    k[0, 0, 2] = torch.tensor([10.0, 0.0])
    assert impl.max_attention_logit(q, k, causal=True).item() == 0.0
    assert math.isclose(impl.max_attention_logit(q, k, causal=False).item(), 100.0 / math.sqrt(2))


# ------------------------------------------------------------------- qk_clip (MHA)
def test_heads_below_tau_untouched(impl):
    x, Wq, Wk = _mha_setup()
    q, k = _qk(x, Wq, Wk)
    S = impl.max_attention_logit(q, k)
    tau = float(S.sort().values[1]) + 1e-6      # heads with the two smallest logits are at or below tau
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, S, tau)
    below = S <= tau
    assert below.sum() == 2 and (~below).sum() == 2
    assert torch.equal(gamma[below], torch.ones(2, dtype=gamma.dtype))
    assert torch.equal(Wq2[below], Wq[below]) and torch.equal(Wk2[below], Wk[below])


@pytest.mark.parametrize("alpha", [0.5, 0.25, 0.0, 1.0])
def test_clipped_heads_hit_tau_on_same_inputs(impl, alpha):
    x, Wq, Wk = _mha_setup()
    q, k = _qk(x, Wq, Wk)
    S = impl.max_attention_logit(q, k)
    tau = 0.5 * float(S.max())                  # forces at least the largest head to clip
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, S, tau, alpha=alpha)
    S2 = impl.max_attention_logit(*_qk(x, Wq2, Wk2))
    over = S > tau
    assert over.any()
    torch.testing.assert_close(S2[over], torch.full((int(over.sum()),), tau, dtype=S2.dtype))
    torch.testing.assert_close(S2[~over], S[~over])            # the rest are untouched
    torch.testing.assert_close(gamma[over], tau / S[over])


def test_alpha_splits_the_factor_between_q_and_k(impl):
    x, Wq, Wk = _mha_setup()
    S = impl.max_attention_logit(*_qk(x, Wq, Wk))
    tau = 0.4 * float(S.max())
    alpha = 0.3
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, S, tau, alpha=alpha)
    over = S > tau
    for h in over.nonzero().flatten().tolist():
        torch.testing.assert_close(Wq2[h], Wq[h] * gamma[h] ** alpha)
        torch.testing.assert_close(Wk2[h], Wk[h] * gamma[h] ** (1 - alpha))


def test_worked_example_250_over_100(impl):
    # the page's example: max logit 250, tau 100 -> gamma 0.4, each of W_q and W_k shrinks by sqrt(0.4)
    Wq = torch.ones(1, 3, 2, dtype=torch.float64)
    Wk = torch.ones(1, 3, 2, dtype=torch.float64)
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, torch.tensor([250.0], dtype=torch.float64), 100.0)
    assert math.isclose(gamma.item(), 0.4)
    assert math.isclose(Wq2.flatten()[0].item(), math.sqrt(0.4)) and math.isclose(Wk2.flatten()[0].item(), math.sqrt(0.4))


def test_per_head_not_per_layer(impl):
    # one head explodes, the others are fine: only that head's weights may change
    x, Wq, Wk = _mha_setup(scale_heads=(20.0, 1.0, 1.0, 1.0))
    S = impl.max_attention_logit(*_qk(x, Wq, Wk))
    tau = float(S[1:].max()) * 1.5
    assert S[0] > tau
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, S, tau)
    assert not torch.equal(Wq2[0], Wq[0])
    assert torch.equal(Wq2[1:], Wq[1:]) and torch.equal(Wk2[1:], Wk[1:])
    assert gamma[0] < 1 and torch.equal(gamma[1:], torch.ones(3, dtype=gamma.dtype))


def test_edge_cases_and_no_mutation(impl):
    x, Wq, Wk = _mha_setup()
    Wq0, Wk0 = Wq.clone(), Wk.clone()
    S = torch.tensor([250.0, 100.0, 0.0, -5.0], dtype=torch.float64)   # exactly tau, zero and negative logits
    Wq2, Wk2, gamma = impl.qk_clip(Wq, Wk, S, 100.0)
    assert torch.equal(Wq, Wq0) and torch.equal(Wk, Wk0)                # inputs not modified in place
    assert torch.isfinite(Wq2).all() and torch.isfinite(gamma).all()
    torch.testing.assert_close(gamma, torch.tensor([0.4, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_clip_is_idempotent_once_at_tau(impl):
    x, Wq, Wk = _mha_setup()
    S = impl.max_attention_logit(*_qk(x, Wq, Wk))
    tau = 0.5 * float(S.max())
    Wq2, Wk2, _ = impl.qk_clip(Wq, Wk, S, tau)
    S2 = impl.max_attention_logit(*_qk(x, Wq2, Wk2))
    Wq3, Wk3, gamma3 = impl.qk_clip(Wq2, Wk2, S2, tau)
    torch.testing.assert_close(gamma3, torch.ones_like(gamma3), atol=1e-9, rtol=0)
    torch.testing.assert_close(Wq3, Wq2)


def test_rejects_bad_arguments(impl):
    x, Wq, Wk = _mha_setup()
    S = torch.ones(H, dtype=torch.float64)
    with pytest.raises(ValueError):
        impl.qk_clip(Wq, Wk, S, 0.0)
    with pytest.raises(ValueError):
        impl.qk_clip(Wq, Wk, S, 1.0, alpha=1.5)
    x, Wqc, Wkc, Wqr, Wkr = _mla_setup()
    with pytest.raises(ValueError):
        impl.qk_clip_mla(Wqc, Wkc, Wqr, Wkr, torch.ones(3, dtype=torch.float64), 0.0)


# ----------------------------------------------------------------- qk_clip_mla
def _mla_setup(dtype=torch.float64, B=2, T=6, d_nope=8, d_rope=4, scale=(9.0, 1.0, 2.0)):
    g = torch.Generator().manual_seed(1)
    Hh = len(scale)
    x = torch.randn(B, T, D, generator=g, dtype=dtype)
    s = torch.tensor(scale, dtype=dtype).view(Hh, 1, 1)
    Wqc = torch.randn(Hh, D, d_nope, generator=g, dtype=dtype) * s
    Wkc = torch.randn(Hh, D, d_nope, generator=g, dtype=dtype) * s
    Wqr = torch.randn(Hh, D, d_rope, generator=g, dtype=dtype) * s
    Wkr = torch.randn(D, d_rope, generator=g, dtype=dtype)
    return x, Wqc, Wkc, Wqr, Wkr


def _mla_qk(x, Wqc, Wkc, Wqr, Wkr):
    """Concatenate [content; rotary] parts so that max_attention_logit's 1/sqrt(d_nope + d_rope) is right.
    (The rotation of RoPE is skipped: it is linear and commutes with the scalar rescaling.)"""
    qc = torch.einsum("btd,hde->bhte", x, Wqc)
    kc = torch.einsum("btd,hde->bhte", x, Wkc)
    qr = torch.einsum("btd,hde->bhte", x, Wqr)
    kr = (x @ Wkr).unsqueeze(1).expand(-1, Wqc.shape[0], -1, -1)     # one shared key, broadcast to all heads
    return torch.cat([qc, qr], -1), torch.cat([kc, kr], -1)


def test_mla_factors_are_sqrt_sqrt_gamma_and_shared_key_untouched(impl):
    x, Wqc, Wkc, Wqr, Wkr = _mla_setup()
    S = impl.max_attention_logit(*_mla_qk(x, Wqc, Wkc, Wqr, Wkr))
    tau = 0.5 * float(S.max())
    Wqc2, Wkc2, Wqr2, Wkr2, gamma = impl.qk_clip_mla(Wqc, Wkc, Wqr, Wkr, S, tau)
    over = S > tau
    assert over.any() and (~over).any()
    for h in range(len(S)):
        g = float(gamma[h])
        torch.testing.assert_close(Wqc2[h], Wqc[h] * math.sqrt(g))
        torch.testing.assert_close(Wkc2[h], Wkc[h] * math.sqrt(g))
        torch.testing.assert_close(Wqr2[h], Wqr[h] * g)
    assert torch.equal(Wkr2, Wkr)                                    # the shared rotary key is never rescaled


def test_mla_max_logit_hits_tau_after_clip(impl):
    x, Wqc, Wkc, Wqr, Wkr = _mla_setup()
    S = impl.max_attention_logit(*_mla_qk(x, Wqc, Wkc, Wqr, Wkr))
    tau = 0.5 * float(S.max())
    out = impl.qk_clip_mla(Wqc, Wkc, Wqr, Wkr, S, tau)
    S2 = impl.max_attention_logit(*_mla_qk(x, *out[:4]))
    over = S > tau
    torch.testing.assert_close(S2[over], torch.full((int(over.sum()),), tau, dtype=S2.dtype))
    torch.testing.assert_close(S2[~over], S[~over])
    # the two terms shrink by the same factor, so their ratio inside a head is preserved
    torch.testing.assert_close(out[4][over], tau / S[over])


def test_mla_below_tau_bitwise_unchanged(impl):
    x, Wqc, Wkc, Wqr, Wkr = _mla_setup()
    S = impl.max_attention_logit(*_mla_qk(x, Wqc, Wkc, Wqr, Wkr))
    Wqc2, Wkc2, Wqr2, Wkr2, gamma = impl.qk_clip_mla(Wqc, Wkc, Wqr, Wkr, S, float(S.max()) + 1.0)
    assert torch.equal(gamma, torch.ones_like(gamma))
    assert torch.equal(Wqc2, Wqc) and torch.equal(Wkc2, Wkc) and torch.equal(Wqr2, Wqr) and torch.equal(Wkr2, Wkr)
