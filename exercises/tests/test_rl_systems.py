import math

import pytest
import torch

MODULE = "rl_systems"


# ---------------------------------------------------------------- truncated_is_weights
def test_tis_caps_from_above_only(impl):
    lt = torch.tensor([[0.0, 1.0, -1.0, 3.0]])
    lr = torch.zeros(1, 4)
    w = impl.truncated_is_weights(lt, lr, 2.0, torch.ones(1, 4))
    expected = torch.tensor([[1.0, 2.0, math.exp(-1.0), 2.0]])   # e^1 = 2.72 -> 2, e^-1 untouched
    torch.testing.assert_close(w, expected)


def test_tis_ratio_uses_trainer_over_rollout(impl):
    lt = torch.tensor([[-1.0, -2.0]])
    lr = torch.tensor([[-1.5, -1.0]])
    w = impl.truncated_is_weights(lt, lr, 10.0, torch.ones(1, 2))
    torch.testing.assert_close(w, torch.tensor([[math.exp(0.5), math.exp(-1.0)]]))


def test_tis_mask_zeroes_padding_even_with_garbage(impl):
    ninf = float("-inf")
    lt = torch.tensor([[0.1, ninf, 0.2, float("nan")]])
    lr = torch.tensor([[0.0, ninf, 0.0, 0.0]])
    mask = torch.tensor([[True, False, True, False]])
    w = impl.truncated_is_weights(lt, lr, 5.0, mask)
    assert torch.isfinite(w).all()
    assert w[0, 1].item() == 0.0 and w[0, 3].item() == 0.0
    torch.testing.assert_close(w[0, [0, 2]], torch.exp(torch.tensor([0.1, 0.2])))
    # a {0, 1} float mask must work too
    w2 = impl.truncated_is_weights(lt, lr, 5.0, mask.float())
    torch.testing.assert_close(w, w2)


def test_tis_no_grad_and_no_overflow(impl):
    lt = torch.tensor([[100.0, 1.0]], requires_grad=True)
    lr = torch.zeros(1, 2, requires_grad=True)
    w = impl.truncated_is_weights(lt, lr, 5.0, torch.ones(1, 2))
    assert not w.requires_grad
    assert torch.isfinite(w).all()
    assert w[0, 0].item() == pytest.approx(5.0)
    assert w[0, 1].item() == pytest.approx(math.e)


def test_tis_identical_policies_give_ones(impl):
    x = torch.randn(3, 7)
    w = impl.truncated_is_weights(x, x.clone(), 3.0, torch.ones(3, 7))
    torch.testing.assert_close(w, torch.ones(3, 7))


# ---------------------------------------------------------------- agent_loss_mask
def test_agent_mask_hand_built_trajectory(impl):
    segs = [
        ("system", [1, 2]),
        ("user", [3]),
        ("assistant", [10, 11, 12]),      # reasoning + tool call
        ("tool", [20, 21, 22, 23]),       # tool output: context only
        ("assistant", [13, 14]),          # second tool call
        ("tool", []),                     # empty observation
        ("assistant", [15]),              # final answer
    ]
    ids, mask = impl.agent_loss_mask(segs)
    assert ids == [1, 2, 3, 10, 11, 12, 20, 21, 22, 23, 13, 14, 15]
    assert mask == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    assert all(type(m) is int for m in mask)          # real ints, not bools


def test_agent_mask_counts_only_policy_tokens(impl):
    segs = [("user", list(range(50))), ("assistant", list(range(7))), ("tool", list(range(300))), ("assistant", list(range(4)))]
    ids, mask = impl.agent_loss_mask(segs)
    assert len(ids) == len(mask) == 50 + 7 + 300 + 4
    assert sum(mask) == 7 + 4


def test_agent_mask_custom_train_roles(impl):
    segs = [("user", [1]), ("assistant", [2, 3]), ("tool", [4])]
    _, mask = impl.agent_loss_mask(segs, train_roles=("assistant", "tool"))
    assert mask == [0, 1, 1, 1]
    _, none = impl.agent_loss_mask(segs, train_roles=())
    assert none == [0, 0, 0, 0]


def test_agent_mask_unknown_role_raises(impl):
    with pytest.raises(ValueError):
        impl.agent_loss_mask([("user", [1]), ("critic", [2])])
    with pytest.raises(ValueError):                    # an empty segment still has to name a valid role
        impl.agent_loss_mask([("critic", [])])


# ---------------------------------------------------------------- sequence_ratio_drift
def test_drift_constant_gap_matches_closed_form(impl):
    d = torch.full((2, 16000), 0.01, dtype=torch.float64)
    lengths = torch.tensor([1000, 16000])
    slr, sr, geo = impl.sequence_ratio_drift(d, lengths)
    torch.testing.assert_close(slr, torch.tensor([10.0, 160.0], dtype=torch.float64))
    assert sr[0].item() == pytest.approx(math.exp(10), rel=1e-9)
    assert math.isfinite(sr[1].item()) and sr[1].item() == pytest.approx(math.exp(160), rel=1e-9)
    torch.testing.assert_close(geo, torch.full((2,), math.exp(0.01), dtype=torch.float64))


def test_drift_grows_with_length_but_geometric_mean_does_not(impl):
    d = torch.full((4, 4000), 0.01)
    lengths = torch.tensor([100, 500, 1000, 4000])
    slr, sr, geo = impl.sequence_ratio_drift(d, lengths)
    assert (slr[1:] > slr[:-1]).all()
    assert (sr[1:] > sr[:-1]).all()
    torch.testing.assert_close(geo, geo[:1].expand(4).clone())


def test_drift_float32_input_is_accumulated_in_float64(impl):
    d = torch.full((1, 4000), 0.05)                    # float32 input; e^200 overflows float32
    slr, sr, geo = impl.sequence_ratio_drift(d, torch.tensor([4000]))
    assert slr.dtype == sr.dtype == geo.dtype == torch.float64
    assert math.isfinite(sr[0].item())
    assert sr[0].item() == pytest.approx(math.exp(200), rel=1e-4)


def test_drift_ignores_padding_and_handles_empty(impl):
    d = torch.tensor([[0.1, 0.2, 99.0, 99.0], [5.0, 5.0, 5.0, 5.0], [0.5, float("nan"), float("inf"), 1.0]])
    lengths = torch.tensor([2, 0, 1])
    slr, sr, geo = impl.sequence_ratio_drift(d, lengths)
    torch.testing.assert_close(slr, torch.tensor([0.3, 0.0, 0.5], dtype=torch.float64))
    assert sr[1].item() == 1.0 and geo[1].item() == 1.0
    assert geo[0].item() == pytest.approx(math.exp(0.15))


def test_drift_random_signs_scale_with_sqrt_length(impl):
    g = torch.Generator().manual_seed(0)
    d = 0.01 * torch.randn(4000, 1600, generator=g, dtype=torch.float64)
    for T in (100, 1600):
        slr, _, _ = impl.sequence_ratio_drift(d, torch.full((4000,), T))
        assert slr.std().item() == pytest.approx(0.01 * math.sqrt(T), rel=0.06)


# ---------------------------------------------------------------- staleness_admits
def test_staleness_eta_zero_is_synchronous(impl):
    B = 8
    assert sum(impl.staleness_admits(n, B, 0, 0) for n in range(100)) == B
    assert not impl.staleness_admits(8, B, 0, 0)
    assert impl.staleness_admits(8, B, 1, 0)         # one training step later, the next batch opens


def test_staleness_bounds_outstanding_samples(impl):
    B = 8
    for eta in (0, 1, 4):
        for i in (0, 3):
            admitted = sum(impl.staleness_admits(n, B, i, eta) for n in range(1000))
            assert admitted == (i + eta + 1) * B
            assert admitted - i * B == (eta + 1) * B  # beyond the i*B already consumed by training


def test_staleness_returns_python_bool(impl):
    assert impl.staleness_admits(0, 4, 0, 0) is True
    assert impl.staleness_admits(40, 4, 0, 0) is False
