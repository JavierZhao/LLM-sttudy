import ast
import inspect

import pytest
import torch
import torch.nn.functional as F

MODULE = "layernorm_backward"
D64 = torch.float64


# ---------------------------------------------------------------- references (written here, not imported)
def _ln_ref(x, g, b, eps):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def _rms_ref(x, g, eps):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * g


def _data(shape, d, dtype=D64):
    x = torch.randn(*shape, d, dtype=dtype) * 2 + 0.5
    g = torch.randn(d, dtype=dtype)
    b = torch.randn(d, dtype=dtype)
    dy = torch.randn(*shape, d, dtype=dtype)
    return x, g, b, dy


SHAPES = [(5,), (2, 3), (), (1, 1, 4)]


# ---------------------------------------------------------------- forward
@pytest.mark.parametrize("shape", SHAPES)
def test_layernorm_forward_matches_torch(impl, shape):
    x, g, b, _ = _data(shape, 8)
    y, cache = impl.layernorm_forward(x, g, b, 1e-5)
    assert y.shape == x.shape and y.dtype == x.dtype
    torch.testing.assert_close(y, F.layer_norm(x, (8,), g, b, 1e-5), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y, _ln_ref(x, g, b, 1e-5), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_forward_matches_formula(impl, shape):
    x, g, _, _ = _data(shape, 8)
    y, _ = impl.rmsnorm_forward(x, g, 1e-6)
    assert y.shape == x.shape and y.dtype == x.dtype
    torch.testing.assert_close(y, _rms_ref(x, g, 1e-6), atol=1e-12, rtol=1e-12)


def test_normalized_rows_have_zero_mean_unit_variance(impl):
    x, _, _, _ = _data((6,), 32)
    y, _ = impl.layernorm_forward(x, torch.ones(32, dtype=D64), torch.zeros(32, dtype=D64), 0.0)
    torch.testing.assert_close(y.mean(-1), torch.zeros(6, dtype=D64), atol=1e-12, rtol=0)
    torch.testing.assert_close(y.var(-1, unbiased=False), torch.ones(6, dtype=D64), atol=1e-12, rtol=0)


# ---------------------------------------------------------------- backward vs autograd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("eps", [1e-5, 1e-2])
def test_layernorm_backward_matches_autograd(impl, shape, eps):
    x, g, b, dy = _data(shape, 8)
    xr, gr, br = (t.clone().requires_grad_() for t in (x, g, b))
    F.layer_norm(xr, (8,), gr, br, eps).backward(dy)
    _, cache = impl.layernorm_forward(x, g, b, eps)
    dx, dg, db = impl.layernorm_backward(dy, cache)
    assert dx.shape == x.shape and dg.shape == (8,) and db.shape == (8,)
    torch.testing.assert_close(dx, xr.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(dg, gr.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(db, br.grad, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-2])
def test_rmsnorm_backward_matches_autograd(impl, shape, eps):
    x, g, _, dy = _data(shape, 8)
    xr, gr = (t.clone().requires_grad_() for t in (x, g))
    _rms_ref(xr, gr, eps).backward(dy)
    _, cache = impl.rmsnorm_forward(x, g, eps)
    dx, dg = impl.rmsnorm_backward(dy, cache)
    assert dx.shape == x.shape and dg.shape == (8,)
    torch.testing.assert_close(dx, xr.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(dg, gr.grad, atol=1e-11, rtol=1e-11)


def _as_autograd_function(fwd, bwd, n_params):
    """Wrap a hand-written forward/backward pair as a torch.autograd.Function for gradcheck."""
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, *params):
            y, cache = fwd(x, *params)
            ctx.cache = cache
            return y

        @staticmethod
        def backward(ctx, dy):
            grads = bwd(dy.contiguous(), ctx.cache)
            return tuple(grads)
    return Fn.apply


def test_layernorm_passes_gradcheck(impl):
    x, g, b, _ = _data((2, 3), 6)
    fn = _as_autograd_function(lambda x, g, b: impl.layernorm_forward(x, g, b, 1e-5),
                               impl.layernorm_backward, 2)
    args = tuple(t.clone().requires_grad_() for t in (x, g, b))
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-6, rtol=1e-5)


def test_rmsnorm_passes_gradcheck(impl):
    x, g, _, _ = _data((2, 3), 6)
    fn = _as_autograd_function(lambda x, g: impl.rmsnorm_forward(x, g, 1e-6),
                               impl.rmsnorm_backward, 1)
    args = tuple(t.clone().requires_grad_() for t in (x, g))
    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- properties that need no reference
def test_layernorm_dx_sums_to_zero_and_is_orthogonal_to_x(impl):
    # y is invariant to x -> x + c (so sum(dx) = 0) and to x -> s * x when eps = 0 (so dx . x = 0)
    x, g, b, dy = _data((4, 3), 16)
    _, cache = impl.layernorm_forward(x, g, b, 0.0)
    dx, _, _ = impl.layernorm_backward(dy, cache)
    torch.testing.assert_close(dx.sum(-1), torch.zeros(4, 3, dtype=D64), atol=1e-10, rtol=0)
    torch.testing.assert_close((dx * x).sum(-1), torch.zeros(4, 3, dtype=D64), atol=1e-9, rtol=0)


def test_rmsnorm_dx_is_orthogonal_to_x_but_does_not_sum_to_zero(impl):
    x, g, _, dy = _data((4, 3), 16)
    _, cache = impl.rmsnorm_forward(x, g, 0.0)
    dx, _ = impl.rmsnorm_backward(dy, cache)
    torch.testing.assert_close((dx * x).sum(-1), torch.zeros(4, 3, dtype=D64), atol=1e-9, rtol=0)
    assert dx.sum(-1).abs().max() > 1e-3            # RMSNorm is not shift invariant


def test_dg_and_db_are_sums_over_all_leading_axes(impl):
    x, g, b, dy = _data((3, 4), 5)
    y, cache = impl.layernorm_forward(x, g, b, 1e-5)
    _, dg, db = impl.layernorm_backward(dy, cache)
    xhat = (y - b) / g
    torch.testing.assert_close(db, dy.reshape(-1, 5).sum(0), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(dg, (dy * xhat).reshape(-1, 5).sum(0), atol=1e-11, rtol=1e-11)


def test_scale_and_shift_invariance_of_forward(impl):
    x, g, b, _ = _data((5,), 12)
    y1, _ = impl.layernorm_forward(x, g, b, 0.0)
    y2, _ = impl.layernorm_forward(3.0 * x + 7.0, g, b, 0.0)
    torch.testing.assert_close(y1, y2, atol=1e-10, rtol=1e-10)
    r1, _ = impl.rmsnorm_forward(x, g, 0.0)
    r2, _ = impl.rmsnorm_forward(-4.0 * x, g, 0.0)
    torch.testing.assert_close(r1, -r2, atol=1e-10, rtol=1e-10)      # scale invariant (sign flips y)


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_dx_is_row_local(impl, kind):
    x, g, b, dy = _data((3, 4), 6)
    if kind == "layernorm":
        _, cache = impl.layernorm_forward(x, g, b, 1e-5)
        backward = lambda d: impl.layernorm_backward(d, cache)[0]
    else:
        _, cache = impl.rmsnorm_forward(x, g, 1e-6)
        backward = lambda d: impl.rmsnorm_backward(d, cache)[0]
    dx1 = backward(dy)
    dy2 = dy.clone()
    dy2[1, 2] += torch.randn(6, dtype=D64)                 # change the incoming gradient of ONE row
    dx2 = backward(dy2)
    moved = (dx1 - dx2).abs().sum(-1)
    assert moved[1, 2] > 1e-6
    moved[1, 2] = 0.0
    assert moved.sum() == 0.0                              # every other row is untouched


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_directional_finite_differences(impl, kind):
    x, g, b, dy = _data((2, 3), 8)
    v = torch.randn_like(x)
    h = 1e-6
    if kind == "layernorm":
        fwd = lambda z: impl.layernorm_forward(z, g, b, 1e-5)
        dx = lambda cache: impl.layernorm_backward(dy, cache)[0]
    else:
        fwd = lambda z: impl.rmsnorm_forward(z, g, 1e-6)
        dx = lambda cache: impl.rmsnorm_backward(dy, cache)[0]
    loss = lambda z: (dy * fwd(z)[0]).sum()
    fd = (loss(x + h * v) - loss(x - h * v)) / (2 * h)     # directional derivative of L = sum(dy * y)
    analytic = (dx(fwd(x)[1]) * v).sum()
    assert fd.item() == pytest.approx(analytic.item(), rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------- contract details
def test_inputs_are_not_modified_and_cache_can_be_reused(impl):
    x, g, b, dy = _data((3,), 8)
    saved = [t.clone() for t in (x, g, b, dy)]
    _, cache = impl.layernorm_forward(x, g, b, 1e-5)
    first = impl.layernorm_backward(dy, cache)
    second = impl.layernorm_backward(dy, cache)          # in-place edits of the cache would show up here
    for a, b_ in zip(first, second):
        torch.testing.assert_close(a, b_, atol=0, rtol=0)
    for t, s in zip((x, g, b, dy), saved):
        assert torch.equal(t, s)
    _, cache = impl.rmsnorm_forward(x, g, 1e-6)
    first = impl.rmsnorm_backward(dy, cache)
    second = impl.rmsnorm_backward(dy, cache)
    for a, b_ in zip(first, second):
        torch.testing.assert_close(a, b_, atol=0, rtol=0)
    assert torch.equal(dy, saved[3])


def test_low_precision_uses_float32_statistics(impl):
    # bf16 values around 100 are spaced 0.5 apart: statistics computed in bf16 lose the variance
    d = 256
    x = (torch.randn(4, d) + 100).to(torch.bfloat16)
    g, b = torch.randn(d), torch.randn(d)
    y, cache = impl.layernorm_forward(x, g, b, 1e-5)
    assert y.dtype == torch.bfloat16
    ref = F.layer_norm(x.float(), (d,), g, b, 1e-5)
    assert (y.float() - ref).abs().max() < 5e-2           # bf16 output rounding only
    dy = torch.randn(4, d).to(torch.bfloat16)
    dx, dg, db = impl.layernorm_backward(dy, cache)
    assert dx.dtype == torch.bfloat16 and dg.dtype == torch.float32 and db.dtype == torch.float32
    xr = x.float().requires_grad_()
    gr, br = g.clone().requires_grad_(), b.clone().requires_grad_()
    F.layer_norm(xr, (d,), gr, br, 1e-5).backward(dy.float())
    assert (dg - gr.grad).abs().max() < 1e-3 * gr.grad.abs().max()
    assert (db - br.grad).abs().max() < 1e-3 * br.grad.abs().max()
    assert (dx.float() - xr.grad).abs().max() < 2e-2 * xr.grad.abs().max()
    gb, bb = g.to(torch.bfloat16), b.to(torch.bfloat16)   # bf16 parameters: dg and db come back in g.dtype
    _, cache_b = impl.layernorm_forward(x, gb, bb, 1e-5)
    _, dg_b, db_b = impl.layernorm_backward(dy, cache_b)
    assert dg_b.dtype == torch.bfloat16 and db_b.dtype == torch.bfloat16
    yr, cache = impl.rmsnorm_forward(x, g, 1e-6)
    assert yr.dtype == torch.bfloat16
    assert (yr.float() - _rms_ref(x.float(), g, 1e-6)).abs().max() < 5e-2 * yr.float().abs().max()


def test_degenerate_rows_stay_finite(impl):
    d = 8
    x = torch.zeros(2, d, dtype=D64)
    x[1] = 3.0                                             # a constant row: zero variance
    g, b, dy = torch.randn(d, dtype=D64), torch.randn(d, dtype=D64), torch.randn(2, d, dtype=D64)
    y, cache = impl.layernorm_forward(x, g, b, 1e-5)
    torch.testing.assert_close(y, b.expand(2, d), atol=1e-12, rtol=0)
    xr = x.clone().requires_grad_()
    F.layer_norm(xr, (d,), g, b, 1e-5).backward(dy)
    dx, _, _ = impl.layernorm_backward(dy, cache)
    assert torch.isfinite(dx).all()
    torch.testing.assert_close(dx, xr.grad, atol=1e-8, rtol=1e-10)
    y, cache = impl.rmsnorm_forward(x, g, 1e-6)
    assert torch.isfinite(y).all() and torch.equal(y[0], torch.zeros(d, dtype=D64))
    dx, dg = impl.rmsnorm_backward(dy, cache)
    assert torch.isfinite(dx).all() and torch.isfinite(dg).all()


def test_no_autograd_and_no_builtin_norms(impl):
    """The drill forbids autograd and the built-in norm layers: scan the module source
    (docstrings and comments are ignored because only AST names and attributes count)."""
    x, g, b = torch.randn(2, 4), torch.randn(4), torch.randn(4)
    impl.layernorm_forward(x, g, b)                        # stubs raise here
    banned = {"autograd", "backward", "layer_norm", "rms_norm", "LayerNorm", "RMSNorm"}
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(impl))):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            used.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in banned:
            used.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            used |= {a.name for a in node.names if a.name in banned}
    assert not used, f"forbidden names used: {sorted(used)}"
