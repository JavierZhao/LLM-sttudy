import pytest
import torch
import torch.nn.functional as F

MODULE = "parallel"


def rand(*shape, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


# ------------------------------------------------------------------------- ring all-reduce

def make_chunks(sizes, n, seed=0):
    """chunks[r][c]: rank r's tensor for chunk c, with chunk c having sizes[c] elements."""
    return [[rand(sizes[c], seed=seed + 100 * r + c) for c in range(n)] for r in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_ring_allreduce_sums_every_chunk_on_every_rank(impl, n):
    sizes = [5 + 3 * c for c in range(n)]
    chunks = make_chunks(sizes, n)
    result, _ = impl.ring_allreduce(chunks)
    assert len(result) == n
    for c in range(n):
        want = sum(chunks[r][c] for r in range(n))
        for r in range(n):
            torch.testing.assert_close(result[r][c], want)


def test_ring_allreduce_does_not_modify_or_alias_the_input(impl):
    n = 4
    chunks = make_chunks([6] * n, n)
    before = [[c.clone() for c in rank] for rank in chunks]
    result, _ = impl.ring_allreduce(chunks)
    for r in range(n):
        for c in range(n):
            assert torch.equal(chunks[r][c], before[r][c])
            assert result[r][c].data_ptr() != chunks[r][c].data_ptr()


def test_ring_allreduce_bytes_equal_2_n_minus_1_over_n_of_the_tensor(impl):
    n, chunk = 8, 25
    chunks = make_chunks([chunk] * n, n)
    _, sent = impl.ring_allreduce(chunks)
    total_bytes = n * chunk * 8                          # float64 = 8 bytes
    expected = 2 * (n - 1) * chunk * 8                   # = 2 (n-1)/n * total_bytes
    assert expected == pytest.approx(2 * (n - 1) / n * total_bytes)
    assert sent == [expected] * n


def test_ring_allreduce_follows_the_stated_schedule_for_uneven_chunks(impl):
    # With unequal chunk sizes, the bytes a rank sends identify the schedule: rank r sends every
    # chunk except (r+1)%n in reduce-scatter and every chunk except (r+2)%n in all-gather.
    n = 5
    sizes = [2, 3, 5, 7, 11]
    chunks = make_chunks(sizes, n)
    _, sent = impl.ring_allreduce(chunks)
    for r in range(n):
        want = (2 * sum(sizes) - sizes[(r + 1) % n] - sizes[(r + 2) % n]) * 8
        assert sent[r] == want


def test_ring_allreduce_single_rank_sends_nothing(impl):
    chunks = [[rand(4)]]
    result, sent = impl.ring_allreduce(chunks)
    assert sent == [0]
    torch.testing.assert_close(result[0][0], chunks[0][0])


def test_ring_allreduce_counts_element_size(impl):
    n = 3
    chunks = [[rand(4, dtype=torch.float32, seed=r * 10 + c) for c in range(n)] for r in range(n)]
    _, sent = impl.ring_allreduce(chunks)
    assert sent == [2 * (n - 1) * 4 * 4] * n             # float32 = 4 bytes


# ---------------------------------------------------------------- column/row-parallel MLP

def dense_mlp(x, W1, W2):
    return F.gelu(x @ W1) @ W2


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_mlp_matches_the_dense_mlp(impl, tp):
    x, W1, W2 = rand(2, 5, 8), rand(8, 32, seed=1) / 3, rand(32, 8, seed=2) / 3
    y, partials = impl.column_row_parallel_mlp(x, W1, W2, tp)
    torch.testing.assert_close(y, dense_mlp(x, W1, W2))
    assert len(partials) == tp
    torch.testing.assert_close(sum(partials), y)
    assert all(p.shape == y.shape for p in partials)


def test_mlp_partials_are_partial_sums_not_the_answer(impl):
    x, W1, W2 = rand(3, 8), rand(8, 16, seed=1), rand(16, 8, seed=2)
    y, partials = impl.column_row_parallel_mlp(x, W1, W2, 2)
    assert not torch.allclose(partials[0], y)            # each rank alone is NOT the output


def test_mlp_rank_r_depends_only_on_its_own_shards(impl):
    # Column-parallel W1 + row-parallel W2: no communication is needed before the all-reduce.
    tp, d_ff = 4, 32
    s = d_ff // tp
    x, W1, W2 = rand(3, 8), rand(8, d_ff, seed=1), rand(d_ff, 8, seed=2)
    _, base = impl.column_row_parallel_mlp(x, W1, W2, tp)
    j = 2
    W1b, W2b = W1.clone(), W2.clone()
    W1b[:, j * s:(j + 1) * s] += 1.0
    W2b[j * s:(j + 1) * s, :] -= 0.5
    _, pert = impl.column_row_parallel_mlp(x, W1b, W2b, tp)
    for r in range(tp):
        if r == j:
            assert not torch.allclose(pert[r], base[r])
        else:
            assert torch.equal(pert[r], base[r])


def test_mlp_custom_activation(impl):
    x, W1, W2 = rand(2, 8), rand(8, 16, seed=1), rand(16, 8, seed=2)
    y, _ = impl.column_row_parallel_mlp(x, W1, W2, 2, act=torch.relu)
    torch.testing.assert_close(y, torch.relu(x @ W1) @ W2)


# ---------------------------------------------------------------- head-parallel attention

def dense_mha(x, Wq, Wk, Wv, Wo, n_heads):
    B, T, d = x.shape
    d_h = d // n_heads

    def split(t):
        return t.view(B, T, n_heads, d_h).transpose(1, 2)

    o = F.scaled_dot_product_attention(split(x @ Wq), split(x @ Wk), split(x @ Wv), is_causal=True)
    return o.transpose(1, 2).reshape(B, T, d) @ Wo


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_attention_matches_dense_multihead_attention(impl, tp):
    B, T, d, H = 2, 6, 16, 8
    x = rand(B, T, d)
    Ws = [rand(d, d, seed=s) / 4 for s in range(1, 5)]
    y, partials = impl.head_parallel_attention(x, *Ws, n_heads=H, tp=tp)
    torch.testing.assert_close(y, dense_mha(x, *Ws, H))
    assert len(partials) == tp
    torch.testing.assert_close(sum(partials), y)


def test_attention_rank_r_only_reads_its_own_heads(impl):
    B, T, d, H, tp = 1, 5, 16, 8, 4
    dh, hp = d // H, H // tp
    x = rand(B, T, d)
    Wq, Wk, Wv, Wo = [rand(d, d, seed=s) / 4 for s in range(1, 5)]
    _, base = impl.head_parallel_attention(x, Wq, Wk, Wv, Wo, H, tp)
    cols = slice(1 * hp * dh, 2 * hp * dh)               # rank 1's heads
    Wq2, Wo2 = Wq.clone(), Wo.clone()
    Wq2[:, cols] += 1.0
    Wo2[cols, :] *= 2.0
    _, pert = impl.head_parallel_attention(x, Wq2, Wk, Wv, Wo2, H, tp)
    for r in range(tp):
        if r == 1:
            assert not torch.allclose(pert[r], base[r])
        else:
            assert torch.equal(pert[r], base[r])


def test_attention_is_causal(impl):
    B, T, d, H = 1, 6, 8, 4
    x = rand(B, T, d)
    Ws = [rand(d, d, seed=s) / 3 for s in range(1, 5)]
    y0, _ = impl.head_parallel_attention(x, *Ws, n_heads=H, tp=2)
    x2 = x.clone()
    x2[:, -1] += 5.0                                      # change only the last token
    y1, _ = impl.head_parallel_attention(x2, *Ws, n_heads=H, tp=2)
    torch.testing.assert_close(y0[:, :-1], y1[:, :-1])    # earlier positions must not see it


# ----------------------------------------------------------------------- memory accounting

@pytest.mark.parametrize("stage, expected_gb", [(0, 120.0), (1, 31.40625), (2, 16.640625), (3, 1.875)])
def test_memory_matches_the_zero_paper_table_1(impl, stage, expected_gb):
    # Rajbhandari et al. 2020, Table 1 and Fig. 1: 7.5B parameters, N_d = 64, K = 12.
    out = impl.training_memory_bytes(7.5e9, stage, 64)
    assert out["total"] / 1e9 == pytest.approx(expected_gb)
    assert out["total"] == pytest.approx(out["params"] + out["grads"] + out["optim"])


def test_memory_breakdown_of_plain_data_parallelism(impl):
    out = impl.training_memory_bytes(7e9, 0, 8)
    assert out["params"] == pytest.approx(14e9)
    assert out["grads"] == pytest.approx(14e9)
    assert out["optim"] == pytest.approx(84e9)
    assert out["total"] == pytest.approx(16 * 7e9)


def test_memory_model_parallel_splits_first_then_zero_shards(impl):
    full = impl.training_memory_bytes(64e9, 1, 8, mp=1)["total"]
    split = impl.training_memory_bytes(64e9, 1, 8, mp=16)["total"]
    assert split == pytest.approx(full / 16)
    # 1T parameters, ZeRO-3, dp = 1024: 15.6 GB per GPU (Table 1).
    assert impl.training_memory_bytes(1e12, 3, 1024)["total"] / 1e9 == pytest.approx(15.625)


def test_memory_custom_byte_widths(impl):
    # DeepSeek-V3-style: bf16 weights, fp32 grads, fp32 master + two bf16 moments = 8 B of state.
    out = impl.training_memory_bytes(1e9, 2, 4, param_bytes=2, grad_bytes=4, optim_bytes=8)
    assert out["params"] == pytest.approx(2e9)
    assert out["grads"] == pytest.approx(1e9)
    assert out["optim"] == pytest.approx(2e9)


def test_memory_rejects_bad_stage(impl):
    with pytest.raises(ValueError):
        impl.training_memory_bytes(1e9, 4, 8)
    with pytest.raises(ValueError):
        impl.training_memory_bytes(1e9, -1, 8)


def test_zero_communication_volume(impl):
    psi, dp = 8e9, 64
    base = impl.zero_comm_bytes_per_step(psi, 0, dp)
    assert base == pytest.approx(2 * psi * 2 * (dp - 1) / dp)
    assert impl.zero_comm_bytes_per_step(psi, 1, dp) == pytest.approx(base)
    assert impl.zero_comm_bytes_per_step(psi, 2, dp) == pytest.approx(base)
    assert impl.zero_comm_bytes_per_step(psi, 3, dp) == pytest.approx(1.5 * base)   # ZeRO-3 = 1.5x DP
    assert impl.zero_comm_bytes_per_step(psi, 0, 1) == 0
    with pytest.raises(ValueError):
        impl.zero_comm_bytes_per_step(psi, 5, dp)


# --------------------------------------------------------------------------- pipeline

def test_bubble_formulas(impl):
    assert impl.pipeline_bubble(16, 64, of="ideal") == pytest.approx(15 / 64)
    assert impl.pipeline_bubble(16, 64, of="total") == pytest.approx(15 / 79)
    assert impl.pipeline_bubble(16, 64, v=4, of="ideal") == pytest.approx(15 / 256)
    assert impl.pipeline_bubble(16, 64, v=4, of="total") == pytest.approx(15 / (4 * 64 + 15))
    assert impl.pipeline_bubble(1, 8) == 0
    with pytest.raises(ValueError):
        impl.pipeline_bubble(4, 8, of="nonsense")


@pytest.mark.parametrize("p, m, tf, tb", [(4, 8, 1.0, 2.0), (4, 4, 1.0, 2.0), (8, 16, 1.0, 1.0), (3, 2, 1.0, 3.0), (1, 5, 1.0, 2.0)])
def test_simulated_1f1b_makespan_matches_the_closed_form(impl, p, m, tf, tb):
    makespan, _ = impl.simulate_1f1b(p, m, tf, tb)
    assert makespan == pytest.approx((m + p - 1) * (tf + tb))


def test_simulated_idle_fraction_matches_the_bubble_formula(impl):
    p, m = 8, 24
    makespan, _ = impl.simulate_1f1b(p, m, 1.0, 2.0)
    idle = 1 - m * 3.0 / makespan
    assert idle == pytest.approx((p - 1) / (m + p - 1))


def test_1f1b_holds_p_minus_i_microbatches_but_gpipe_holds_all_m(impl):
    p, m = 4, 8
    _, peak_1f1b = impl.simulate_1f1b(p, m, schedule="1f1b")
    _, peak_gpipe = impl.simulate_1f1b(p, m, schedule="gpipe")
    assert peak_1f1b == [4, 3, 2, 1]                     # stage i holds p - i microbatches
    assert peak_gpipe == [8, 8, 8, 8]                    # all m forwards run before any backward


def test_1f1b_and_gpipe_have_the_same_bubble(impl):
    a, _ = impl.simulate_1f1b(6, 12, 1.0, 2.0, schedule="1f1b")
    b, _ = impl.simulate_1f1b(6, 12, 1.0, 2.0, schedule="gpipe")
    assert a == pytest.approx(b)


def test_fewer_microbatches_than_stages_still_works(impl):
    makespan, peak = impl.simulate_1f1b(4, 2, 1.0, 2.0)
    assert makespan == pytest.approx((2 + 4 - 1) * 3.0)
    assert peak[0] == 2 and peak[-1] == 1


# ------------------------------------------------------------------------------- MFU

def test_mfu_reproduces_llama3_405b_and_palm(impl):
    # Llama 3 405B, Table 4: 430 TFLOP/s per GPU on an H100 (989 dense bf16) is 43% MFU.
    assert 430e12 / 989.4e12 == pytest.approx(0.4346, abs=1e-3)
    fpt = impl.model_flops_per_token(405e9)
    assert fpt == pytest.approx(6 * 405e9)
    tok_per_gpu = 430e12 / fpt
    assert impl.mfu(tok_per_gpu, fpt, 989.4e12) == pytest.approx(0.4346, abs=1e-3)
    # PaLM 540B: 238.3K tokens/s on 6144 TPU v4 chips (275 TFLOP/s each): 45.7% without attention,
    # 46.2% with it (12 * L * (H*Q) * T, full attention, L=118, H*Q=48*256, T=2048).
    tps = 238.3e3 / 6144
    assert impl.mfu(tps, impl.model_flops_per_token(540e9), 275e12) == pytest.approx(0.457, abs=0.001)
    with_attn = impl.model_flops_per_token(540e9, n_layers=118, d_model=48 * 256, seq_len=2048, causal=False)
    assert impl.mfu(tps, with_attn, 275e12) == pytest.approx(0.462, abs=0.001)


def test_causal_attention_term_is_half_of_the_full_term(impl):
    full = impl.model_flops_per_token(1e9, 24, 2048, 4096, causal=False) - 6e9
    causal = impl.model_flops_per_token(1e9, 24, 2048, 4096, causal=True) - 6e9
    assert full == pytest.approx(12 * 24 * 2048 * 4096)
    assert causal == pytest.approx(full / 2)
    assert impl.model_flops_per_token(1e9) == 6e9


def test_mfu_worked_example_from_the_page(impl):
    # Llama-3-8B, 64 H100s, 500K tokens/s in total, s = 8192.
    n_mm = 8_030_261_248 - 128256 * 4096 - (2 * 4096 * 32 + 4096)       # matmul weights only
    fpt = impl.model_flops_per_token(n_mm, n_layers=32, d_model=4096, seq_len=8192, causal=True)
    assert fpt / 1e9 == pytest.approx(51.47, abs=0.01)
    assert impl.mfu(500e3 / 64, fpt, 989.4e12) == pytest.approx(0.406, abs=0.001)
