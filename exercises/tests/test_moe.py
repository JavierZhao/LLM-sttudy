import math

import pytest
import torch

MODULE = "moe"


# ----------------------------------------------------------------------------- helpers
def _scores(x, W_r, score):
    logits = x @ W_r
    return logits.softmax(-1) if score == "softmax" else logits.sigmoid()


def _naive_route(x, W_r, k, score, bias, norm_topk, route_scale):
    """Per-token reference router: returns a list of (indices, gates) with indices in selection order."""
    s = _scores(x, W_r, score)
    out = []
    for t in range(x.shape[0]):
        sel = s[t] if bias is None else s[t] + bias
        idx = sorted(range(s.shape[1]), key=lambda i: -float(sel[i].detach()))[:k]
        g = torch.stack([s[t, i] for i in idx])
        if norm_topk:
            g = g / g.sum()
        out.append((idx, g * route_scale))
    return out


def _naive_moe(layer, x):
    """Naive per-token MoE forward using the layer's own parameters (differentiable)."""
    xf = x.reshape(-1, x.shape[-1])
    routes = _naive_route(xf, layer.W_r, layer.top_k, layer.score, layer.bias, layer.norm_topk, layer.route_scale)
    rows = []
    for t, (idx, g) in enumerate(routes):
        xt = xf[t : t + 1]
        y = torch.zeros_like(xt)
        for sh in layer.shared:
            y = y + sh(xt)
        for j, i in enumerate(idx):
            y = y + g[j] * layer.experts[i](xt)
        rows.append(y)
    return torch.cat(rows, 0).reshape(x.shape)


def _grads(out, w, params):
    loss = (out * w).sum()
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, params)]


# ----------------------------------------------------------------------------- router
def test_router_softmax_matches_topk_then_softmax(impl):
    # softmax over all experts + renormalize over the top-k == softmax over the top-k logits (Mixtral)
    x, W = torch.randn(9, 12), torch.randn(12, 8)
    idx, g = impl.topk_router(x, W, 2, score="softmax")
    assert idx.shape == (9, 2) and g.shape == (9, 2) and idx.dtype == torch.int64
    logits = x @ W
    top = logits.topk(2, dim=-1)
    assert torch.equal(idx, top.indices)
    torch.testing.assert_close(g, top.values.softmax(-1))


def test_router_softmax_without_normalization(impl):
    x, W = torch.randn(7, 10), torch.randn(10, 6)
    idx, g = impl.topk_router(x, W, 3, score="softmax", norm_topk=False)
    p = (x @ W).softmax(-1)
    torch.testing.assert_close(g, p.gather(1, idx))
    assert (g.sum(-1) < 1).all()   # gates do not sum to 1 (Switch, DeepSeek-V2)


def test_router_sigmoid_normalized_and_scaled(impl):
    x, W = torch.randn(11, 16), torch.randn(16, 12)
    idx, g = impl.topk_router(x, W, 4, score="sigmoid", route_scale=2.5)
    s = (x @ W).sigmoid()
    chosen = s.gather(1, idx)
    torch.testing.assert_close(g, 2.5 * chosen / chosen.sum(-1, keepdim=True))
    assert torch.equal(idx, s.topk(4, dim=-1).indices)


def test_router_bias_changes_selection_not_gates(impl):
    x, W = torch.randn(10, 12), torch.randn(12, 8)
    bias = torch.zeros(8)
    bias[5] = 10.0                                   # expert 5 wins every top-k
    idx, g = impl.topk_router(x, W, 2, score="sigmoid", bias=bias)
    assert (idx[:, 0] == 5).all()
    s = (x @ W).sigmoid()                            # gates use the UNBIASED scores
    chosen = s.gather(1, idx)
    torch.testing.assert_close(g, chosen / chosen.sum(-1, keepdim=True))
    # negative bias removes an expert that would otherwise be picked
    bias2 = torch.zeros(8)
    top_without = impl.topk_router(x, W, 2, score="sigmoid")[0]
    e = int(top_without[0, 0])
    bias2[e] = -10.0
    idx2, _ = impl.topk_router(x, W, 2, score="sigmoid", bias=bias2)
    assert (idx2 != e).all()


def test_router_matches_naive_loop(impl):
    x, W = torch.randn(13, 10), torch.randn(10, 9)
    bias = torch.randn(9) * 0.2
    for score, norm in [("softmax", True), ("softmax", False), ("sigmoid", True)]:
        idx, g = impl.topk_router(x, W, 3, score=score, bias=bias, norm_topk=norm, route_scale=1.7)
        for t, (ref_idx, ref_g) in enumerate(_naive_route(x, W, 3, score, bias, norm, 1.7)):
            assert idx[t].tolist() == ref_idx
            torch.testing.assert_close(g[t], ref_g)


def test_router_group_limited(impl):
    torch.manual_seed(1)
    T, d, E, k, n_group, topk_group = 64, 16, 16, 4, 4, 2
    x, W = torch.randn(T, d), torch.randn(d, E) * 1.5
    bias = torch.randn(E) * 0.1
    free_idx, _ = impl.topk_router(x, W, k, score="sigmoid", bias=bias)
    span_free = torch.tensor([len(set((free_idx[t] // (E // n_group)).tolist())) for t in range(T)])
    assert (span_free > topk_group).any(), "test premise: unrestricted routing must sometimes span > topk_group groups"
    idx, g = impl.topk_router(x, W, k, score="sigmoid", bias=bias, n_group=n_group, topk_group=topk_group)
    s = (x @ W).sigmoid()
    sel = s + bias
    gsz = E // n_group
    for t in range(T):
        # reference: group score = sum of the top (k // topk_group) biased scores in the group
        gs = [float(sel[t, j * gsz : (j + 1) * gsz].topk(k // topk_group).values.sum()) for j in range(n_group)]
        keep = sorted(range(n_group), key=lambda j: -gs[j])[:topk_group]
        allowed = [e for e in range(E) if e // gsz in keep]
        ref = sorted(allowed, key=lambda e: -float(sel[t, e]))[:k]
        assert idx[t].tolist() == ref
        assert len(set((idx[t] // gsz).tolist())) <= topk_group
    chosen = s.gather(1, idx)
    torch.testing.assert_close(g, chosen / chosen.sum(-1, keepdim=True))


def test_router_gradients_reach_x_and_router(impl):
    x = torch.randn(6, 8, requires_grad=True)
    W = torch.randn(8, 5, requires_grad=True)
    _, g = impl.topk_router(x, W, 2, score="softmax", norm_topk=False)
    (g * torch.randn(6, 2)).sum().backward()
    assert x.grad is not None and W.grad is not None and W.grad.abs().sum() > 0


# ----------------------------------------------------------------------------- MoE layer
@pytest.mark.parametrize("score,norm,n_shared", [("sigmoid", True, 1), ("softmax", False, 0), ("softmax", True, 2)])
def test_moe_matches_naive_loop_values_and_grads(impl, score, norm, n_shared):
    d, h, E, k = 16, 8, 8, 2
    layer = impl.MoELayer(d, h, E, n_shared, k, score=score, norm_topk=norm, route_scale=1.3).double()
    layer.bias.copy_(torch.randn(E, dtype=torch.float64) * 0.05)
    x = torch.randn(11, d, dtype=torch.float64)
    w = torch.randn(11, d, dtype=torch.float64)
    params = list(layer.parameters())
    out = layer(x)
    ref = _naive_moe(layer, x)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=1e-9)
    for a, b in zip(_grads(out, w, params), _grads(_naive_moe(layer, x), w, params)):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_moe_batched_input_and_empty_experts(impl):
    d, E, k = 12, 32, 2
    layer = impl.MoELayer(d, 6, E, 1, k, score="sigmoid")
    x = torch.randn(2, 3, d)                         # 6 tokens, 12 assignments, at most 12 of 32 experts used
    rows = [0] * E

    def count(e):
        def hook(module, args, output):
            rows[e] += args[0].shape[0]
        return hook

    for e in range(E):
        layer.experts[e].register_forward_hook(count(e))
    out = layer(x)
    assert rows == layer.last_load.tolist()          # each expert ran on exactly its own tokens, never on all of them
    assert out.shape == x.shape
    assert layer.last_indices.shape == (6, k)
    assert layer.last_load.shape == (E,) and int(layer.last_load.sum()) == 6 * k
    assert torch.equal(layer.last_load, torch.bincount(layer.last_indices.reshape(-1), minlength=E))
    assert (layer.last_load == 0).any()
    torch.testing.assert_close(out, _naive_moe(layer, x), atol=1e-5, rtol=1e-5)


def test_moe_last_probs_is_a_distribution_with_grad(impl):
    layer = impl.MoELayer(10, 4, 6, 0, 2, score="sigmoid")
    x = torch.randn(7, 10)
    layer(x)
    p = layer.last_probs
    assert p.shape == (7, 6)
    torch.testing.assert_close(p.sum(-1), torch.ones(7))
    s = (x @ layer.W_r).sigmoid()
    torch.testing.assert_close(p, s / s.sum(-1, keepdim=True))
    (p * torch.randn(7, 6)).sum().backward()         # the graph must reach the router weights
    assert layer.W_r.grad is not None and layer.W_r.grad.abs().sum() > 0


def test_moe_bias_only_steers_selection(impl):
    d, E, k = 12, 6, 2
    layer = impl.MoELayer(d, 5, E, 0, k, score="sigmoid", norm_topk=True)
    x = torch.randn(9, d)
    layer.bias[3] = 10.0
    out = layer(x)
    assert (layer.last_indices[:, 0] == 3).all()
    torch.testing.assert_close(out, _naive_moe(layer, x), atol=1e-5, rtol=1e-5)
    assert not any(p is layer.bias for p in layer.parameters())   # the bias is a buffer, not a Parameter


# ----------------------------------------------------------------------------- balance losses and bias
def test_switch_aux_loss_reference_values(impl):
    N = 4
    uniform_idx = torch.arange(8).remainder(N).unsqueeze(1)
    assert math.isclose(float(impl.switch_aux_loss(torch.full((8, N), 0.25), uniform_idx, N)), 1.0, rel_tol=1e-6)
    collapsed = torch.zeros(8, 1, dtype=torch.int64)
    onehot = torch.zeros(8, N)
    onehot[:, 0] = 1.0
    assert math.isclose(float(impl.switch_aux_loss(onehot, collapsed, N)), float(N), rel_tol=1e-6)
    # hand example: f = (.5, .25, .25, 0), P = (.4, .3, .2, .1)  ->  4 * 0.325 = 1.3
    probs = torch.tensor([[.6, .2, .1, .1], [.2, .4, .3, .1], [.3, .4, .2, .1], [.5, .2, .2, .1]])
    idx = torch.tensor([[0], [0], [1], [2]])
    assert math.isclose(float(impl.switch_aux_loss(probs, idx, N)), 1.3, rel_tol=1e-6)


def test_switch_aux_loss_gradient_flows_only_through_P(impl):
    N, T, k = 4, 6, 2
    logits = torch.randn(T, N, requires_grad=True)
    probs = logits.softmax(-1)
    idx = torch.tensor([[0, 1], [0, 2], [0, 1], [3, 0], [1, 0], [0, 2]])
    probs.retain_grad()
    impl.switch_aux_loss(probs, idx, N).backward()
    f = torch.bincount(idx.reshape(-1), minlength=N).float() / (T * k)
    # d loss / d probs[t, i] = N * f_i / T : the same for every token, largest for the overloaded expert
    torch.testing.assert_close(probs.grad, (N * f / T).expand(T, N).contiguous())
    assert int(f.argmax()) == 0 and torch.equal(probs.grad[0].argmax(), torch.tensor(0))


def test_router_z_loss(impl):
    E = 8
    assert math.isclose(float(impl.router_z_loss(torch.zeros(5, E))), math.log(E) ** 2, rel_tol=1e-6)
    z = torch.randn(4, E) * 3
    torch.testing.assert_close(impl.router_z_loss(z), torch.logsumexp(z, -1).pow(2).mean())
    z.requires_grad_(True)
    impl.router_z_loss(z).backward()
    lse = torch.logsumexp(z.detach(), -1, keepdim=True)
    # d/dz mean(lse^2) = (2 / T) * lse * softmax(z): descent shrinks the logits whenever lse > 0
    torch.testing.assert_close(z.grad, 2.0 / 4 * lse * z.detach().softmax(-1))


def test_update_balance_bias_signs(impl):
    b = torch.zeros(5)
    load = torch.tensor([10, 2, 4, 4, 0])           # mean 4: over, under, exact, exact, under
    new = impl.update_balance_bias(b, load, 0.01)
    torch.testing.assert_close(new, torch.tensor([-0.01, 0.01, 0.0, 0.0, 0.01]))
    assert torch.equal(b, torch.zeros(5))            # not modified in place
    assert new.dtype == b.dtype
    load2 = torch.tensor([400, 4, 4, 4, 4])          # size of the violation is irrelevant (sign update)
    torch.testing.assert_close(impl.update_balance_bias(b, load2, 0.01), torch.tensor([-0.01, 0.01, 0.01, 0.01, 0.01]))


def test_aux_loss_free_bias_rebalances_a_skewed_router(impl):
    torch.manual_seed(0)
    T, E, k, gamma = 2048, 8, 2, 0.01
    skew = torch.linspace(1.5, -1.5, E)              # expert 0 is strongly preferred, expert 7 is nearly dead
    logits = torch.randn(T, E) * 0.5 + skew
    s = logits.sigmoid()
    bias = torch.zeros(E)

    def max_violation(b):
        load = torch.bincount((s + b).topk(k, dim=-1).indices.reshape(-1), minlength=E).float()
        return float((load.max() - load.mean()) / load.mean()), load

    v0, load0 = max_violation(bias)
    assert v0 > 0.5 and load0.min() < load0.mean() * 0.5
    for _ in range(300):
        _, load = max_violation(bias)
        bias = impl.update_balance_bias(bias, load, gamma)
    v1, _ = max_violation(bias)
    assert v1 < 0.1 * v0 and v1 < 0.15
    assert bias[0] < 0 < bias[-1]                    # busy experts were penalized, idle ones boosted


# ----------------------------------------------------------------------------- capacity
def test_expert_capacity(impl):
    assert impl.expert_capacity(4096, 64, 2, 1.25) == 160
    assert impl.expert_capacity(10, 4, 1, 1.0) == 3      # ceil(2.5)
    assert impl.expert_capacity(64, 8, 1, 1.0) == 8


def test_capacity_mask_slot_major_priority(impl):
    # capacity 1. Slot-major: both first choices win their expert, both second choices are dropped.
    idx = torch.tensor([[0, 1], [1, 0]])
    keep = impl.capacity_dispatch_mask(idx, 2, 1)
    assert keep.dtype == torch.bool and keep.shape == (2, 2)
    assert keep.tolist() == [[True, False], [True, False]]
    # hand example with capacity 2
    idx = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 2]])
    assert impl.capacity_dispatch_mask(idx, 3, 2).tolist() == [[True, True], [True, True], [False, True], [False, True]]


def test_capacity_mask_matches_naive(impl):
    torch.manual_seed(3)
    T, E, k = 50, 6, 2
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
    for cap in (0, 5, 10, 17, 100):
        keep = impl.capacity_dispatch_mask(idx, E, cap)
        ref = torch.zeros(T, k, dtype=torch.bool)
        count = [0] * E
        for slot in range(k):
            for t in range(T):
                e = int(idx[t, slot])
                ref[t, slot] = count[e] < cap
                count[e] += 1
        assert torch.equal(keep, ref)
        kept_per_expert = torch.bincount(idx[keep], minlength=E)
        assert (kept_per_expert <= cap).all()
    assert impl.capacity_dispatch_mask(idx, E, T * k).all()
