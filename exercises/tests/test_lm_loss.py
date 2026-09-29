import ast
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

MODULE = "lm_loss"


# ---------------------------------------------------------------- shift_for_lm
def test_shift_basic(impl):
    tokens = torch.tensor([[5, 6, 7, 8]])
    inputs, labels = impl.shift_for_lm(tokens, pad_id=0)
    assert inputs.tolist() == [[5, 6, 7, 8]]
    assert labels.tolist() == [[6, 7, 8, -100]]
    assert inputs.dtype == torch.long and labels.dtype == torch.long


def test_shift_padding_and_no_mutation(impl):
    tokens = torch.tensor([[5, 6, 7, 0, 0],
                           [1, 2, 3, 4, 5]])
    before = tokens.clone()
    inputs, labels = impl.shift_for_lm(tokens, pad_id=0)
    assert torch.equal(tokens, before), "the input tensor must not be modified"
    assert inputs.data_ptr() != tokens.data_ptr(), "inputs must be a copy"
    assert labels.tolist() == [[6, 7, -100, -100, -100],
                               [2, 3, 4, 5, -100]]


def test_shift_masks_by_value(impl):
    # a pad id in the middle of the row is dropped as a target too (masking is by value)
    tokens = torch.tensor([[9, 0, 9, 9]])
    _, labels = impl.shift_for_lm(tokens, pad_id=0)
    assert labels.tolist() == [[-100, 9, 9, -100]]


def test_shift_length_one(impl):
    _, labels = impl.shift_for_lm(torch.tensor([[3], [4]]), pad_id=0)
    assert labels.tolist() == [[-100], [-100]]


# ------------------------------------------------------------ lm_cross_entropy
def test_ce_matches_torch(impl):
    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[0, 2:] = -100
    labels[2, 0] = -100
    ref = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(impl.lm_cross_entropy(logits, labels), ref, atol=1e-6, rtol=1e-6)


def test_ce_custom_ignore_index(impl):
    logits = torch.randn(2, 4, 6)
    labels = torch.randint(0, 6, (2, 4))
    labels[1, 3] = 99
    ref = F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1), ignore_index=99)
    torch.testing.assert_close(impl.lm_cross_entropy(logits, labels, ignore_index=99), ref, atol=1e-6, rtol=1e-6)


def test_ce_is_hand_written(impl):
    """The drill forbids the built-in losses and log-softmax/logsumexp: scan the module source
    (docstrings and comments are not code, so they are ignored)."""
    impl.lm_cross_entropy(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long))   # stubs raise here
    banned = {"cross_entropy", "nll_loss", "log_softmax", "logsumexp"}
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(impl))):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            used.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in banned:
            used.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            used |= {a.name for a in node.names if a.name in banned}
    assert not used, f"forbidden built-ins used: {sorted(used)}"


def test_ce_uniform_logits_give_log_v(impl):
    V = 128
    loss = impl.lm_cross_entropy(torch.zeros(2, 5, V), torch.randint(0, V, (2, 5)))
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(math.log(V), abs=1e-6)


def test_ce_is_token_mean_not_sequence_mean(impl):
    # sequence 0 has 4 supervised tokens with NLL 1 each, sequence 1 has 1 token with NLL 3
    V = 4
    logits = torch.zeros(2, 4, V)
    # make the NLL of the label equal to a chosen value c: put logit gap g so that -log p = c
    def set_nll(b, t, c):
        # logits = [x, 0, 0, 0] with label 0 gives NLL = log(1 + 3 e^{-x}); solve for x
        x = -math.log((math.exp(c) - 1.0) / 3.0)
        logits[b, t, 0] = x
    for t in range(4):
        set_nll(0, t, 1.0)
    set_nll(1, 0, 3.0)
    labels = torch.zeros(2, 4, dtype=torch.long)
    labels[1, 1:] = -100
    got = impl.lm_cross_entropy(logits, labels).item()
    assert got == pytest.approx((4 * 1.0 + 3.0) / 5, abs=1e-5)       # token-mean = 1.4
    assert got != pytest.approx((1.0 + 3.0) / 2, abs=1e-3)           # sequence-mean would be 2.0


def test_ce_all_ignored_is_zero_with_grad(impl):
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), -100)
    loss = impl.lm_cross_entropy(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_ce_gradient_matches_torch(impl):
    torch.manual_seed(1)
    logits = torch.randn(2, 6, 9, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 9, (2, 6))
    labels[0, 4:] = -100
    impl.lm_cross_entropy(logits, labels).backward()
    mine = logits.grad.clone()
    logits.grad = None
    F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), ignore_index=-100).backward()
    torch.testing.assert_close(mine, logits.grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("scale, offset", [(1e4, 0.0), (1.0, 1e4), (1.0, -1e4)])
def test_ce_is_stable_for_huge_logits(impl, scale, offset):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 50) * scale + offset
    labels = torch.randint(0, 50, (2, 5))
    got = impl.lm_cross_entropy(logits, labels)
    assert torch.isfinite(got), "overflow or underflow in the softmax"
    ref = F.cross_entropy(logits.double().reshape(-1, 50), labels.reshape(-1))   # float64 reference
    torch.testing.assert_close(got.double(), ref, atol=1e-3, rtol=1e-4)


def test_ce_huge_logits_gradient_is_finite(impl):
    logits = (torch.randn(1, 3, 20) * 1e4).requires_grad_()
    labels = torch.randint(0, 20, (1, 3))
    impl.lm_cross_entropy(logits, labels).backward()
    assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------- metrics
def test_perplexity_values(impl):
    assert impl.perplexity(2.0) == pytest.approx(7.389056, abs=1e-5)
    assert impl.perplexity(math.log(10.0)) == pytest.approx(10.0, abs=1e-9)
    assert impl.perplexity(0.0) == pytest.approx(1.0)
    out = impl.perplexity(torch.tensor(2.0))
    assert isinstance(out, torch.Tensor) and out.item() == pytest.approx(7.389056, abs=1e-5)
    assert isinstance(impl.perplexity(1.0), float)


def test_perplexity_is_effective_branching_factor(impl):
    # true-token probabilities 1/2, 1/4, 4/5, 1/10: geometric mean of 1/p = 100 ** (1/4)
    p = torch.tensor([0.5, 0.25, 0.8, 0.1])
    mean_nll = float((-p.log()).mean())
    assert impl.perplexity(mean_nll) == pytest.approx(100 ** 0.25, rel=1e-6)


def test_bits_per_byte_worked_example(impl):
    # 1000 tokens at 2.0 nats/token, text is 4000 bytes (4.0 bytes/token)
    assert impl.bits_per_byte(1000 * 2.0, 4000) == pytest.approx(2.0 * 0.25 / math.log(2), rel=1e-9)
    assert impl.bits_per_byte(1000 * 2.0, 4000) == pytest.approx(0.7213475, abs=1e-6)


def test_bits_per_byte_is_tokenizer_independent(impl):
    # One text of 4000 bytes to which the model assigns 2000 nats in total. Tokenizer A cuts it into
    # 1000 tokens (2.0 nats/token), tokenizer B into 2000 tokens (1.0 nats/token).
    n_bytes = 4000
    per_token = {}
    for n_tokens, mean_nll in [(1000, 2.0), (2000, 1.0)]:
        total = mean_nll * n_tokens                  # what the model assigns to the whole text
        per_token[n_tokens] = impl.perplexity(mean_nll)
        assert impl.bits_per_byte(total, n_bytes) == pytest.approx(0.7213475, abs=1e-6)
    assert per_token[1000] == pytest.approx(math.exp(2.0)) and per_token[2000] == pytest.approx(math.e)
    assert per_token[1000] / per_token[2000] == pytest.approx(math.e)     # per-token PPL differs by e
    # bits per byte scales with total information per byte: doubling both leaves it unchanged
    assert impl.bits_per_byte(2 * 2000.0, 2 * n_bytes) == pytest.approx(impl.bits_per_byte(2000.0, n_bytes))


def test_bits_per_byte_rejects_empty_text(impl):
    with pytest.raises(ValueError):
        impl.bits_per_byte(10.0, 0)
    with pytest.raises(ValueError):
        impl.bits_per_byte(10.0, -5)


# ------------------------------------------------------------ end to end
def test_teacher_forced_loss_is_padding_invariant(impl):
    """Padding a sequence must not change the token-mean loss of the batch."""
    torch.manual_seed(0)
    V, d = 13, 8
    emb = torch.randn(V, d)
    head = torch.randn(d, V)

    def fake_model(ids):                              # (B, T) -> (B, T, V); position-wise, enough for this test
        return emb[ids] @ head

    a = torch.tensor([3, 4, 5, 6, 7])
    b = torch.tensor([8, 9, 10])
    padded = torch.zeros(2, 5, dtype=torch.long)
    padded[0] = a
    padded[1, :3] = b
    inputs, labels = impl.shift_for_lm(padded, pad_id=0)
    loss_batch = impl.lm_cross_entropy(fake_model(inputs), labels)

    def manual(seq):
        lp = fake_model(seq[None, :-1]).log_softmax(-1)[0]
        return -lp.gather(-1, seq[1:, None]).sum(), len(seq) - 1

    (sa, na), (sb, nb) = manual(a), manual(b)
    torch.testing.assert_close(loss_batch, (sa + sb) / (na + nb), atol=1e-5, rtol=1e-5)
