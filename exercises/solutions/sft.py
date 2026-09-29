"""Reference solutions: SFT plumbing (page 19).

Chat-template loss masking, sequence packing with a document mask, loss normalization under
packing, and the two knowledge-distillation losses (forward and reverse KL).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ChatTemplate:
    """A ChatML-like template. One message is rendered as header[role] + content + end + sep."""
    header: Dict[str, str]
    end: str
    sep: str = ""
    bos: str = ""
    train_roles: Tuple[str, ...] = ("assistant",)


class PackedSequence(NamedTuple):
    input_ids: Tensor      # (max_len,) long
    loss_mask: Tensor      # (max_len,) bool
    position_ids: Tensor   # (max_len,) long, restarts at 0 for every document
    doc_ids: Tensor        # (max_len,) long, 0, 1, 2, ... per document; -1 on padding


def build_example(
    messages: Sequence[Dict[str, str]],
    tokenize: Callable[[str], List[int]],
    template: ChatTemplate,
    train_on: str = "all",
) -> Tuple[List[int], List[bool]]:
    """Render a conversation and mark which tokens are training targets."""
    if train_on not in ("all", "last"):
        raise ValueError(f"train_on must be 'all' or 'last', got {train_on!r}")
    ids: List[int] = []
    mask: List[bool] = []

    def emit(text: str, trained: bool) -> None:
        if not text:                                   # empty content / empty sep: nothing to tokenize
            return
        toks = tokenize(text)                          # segments are tokenized separately, so every
        ids.extend(toks)                               # boundary between masked and trained text is exact
        mask.extend([trained] * len(toks))

    emit(template.bos, False)
    last = len(messages) - 1
    for i, m in enumerate(messages):
        role = m["role"]
        trained = role in template.train_roles and (train_on == "all" or i == last)
        emit(template.header[role], False)             # the header is supplied at inference time: never a target
        emit(m["content"], trained)
        emit(template.end, trained)                    # the end-of-turn token IS a target: the model must learn to stop
        emit(template.sep, False)                      # text after the stop token is never generated
    return ids, mask


def pack(
    examples: Sequence[Tuple[Sequence[int], Sequence[bool]]],
    max_len: int,
    pad_id: int = 0,
) -> List[PackedSequence]:
    """Next-fit packing in the given order (no reordering, no splitting of examples)."""
    groups: List[List[Tuple[Sequence[int], Sequence[bool]]]] = []
    cur: List[Tuple[Sequence[int], Sequence[bool]]] = []
    used = 0
    for ids, mask in examples:
        n = len(ids)
        if n == 0:
            continue
        if n > max_len:
            raise ValueError(f"example of length {n} does not fit in max_len={max_len}")
        if used + n > max_len:                         # next-fit: close the pack, never look back at it
            groups.append(cur)
            cur, used = [], 0
        cur.append((ids, mask))
        used += n
    if cur:
        groups.append(cur)

    out: List[PackedSequence] = []
    for g in groups:
        input_ids = torch.full((max_len,), pad_id, dtype=torch.long)
        loss_mask = torch.zeros(max_len, dtype=torch.bool)
        position_ids = torch.zeros(max_len, dtype=torch.long)
        doc_ids = torch.full((max_len,), -1, dtype=torch.long)
        off = 0
        for d, (ids, mask) in enumerate(g):
            n = len(ids)
            input_ids[off:off + n] = torch.tensor(list(ids), dtype=torch.long)
            loss_mask[off:off + n] = torch.tensor(list(mask), dtype=torch.bool)
            position_ids[off:off + n] = torch.arange(n)          # positions restart inside each document
            doc_ids[off:off + n] = d
            off += n
        out.append(PackedSequence(input_ids, loss_mask, position_ids, doc_ids))
    return out


def document_causal_mask(doc_ids: Tensor) -> Tensor:
    """Block-diagonal causal mask: query i may attend key j iff j <= i and doc_ids match."""
    T = doc_ids.shape[-1]
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)             # (..., T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    return same_doc & causal                                              # padding (-1) only sees padding: no empty rows


def packed_lm_loss(
    logits: Tensor,
    input_ids: Tensor,
    loss_mask: Tensor,
    doc_ids: Tensor,
    reduction: str = "token",
) -> Tensor:
    """Next-token NLL on packed rows, averaged over tokens or over documents."""
    if reduction not in ("token", "sequence"):
        raise ValueError(f"reduction must be 'token' or 'sequence', got {reduction!r}")
    B, T, V = logits.shape
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, V), input_ids[:, 1:].reshape(-1),
                          reduction="none").view(B, T - 1)                # position t predicts token t + 1
    tgt_doc = doc_ids[:, 1:]
    # a target counts only if it is trained, is real (not padding), and is predicted from its own document
    valid = loss_mask[:, 1:] & (tgt_doc >= 0) & (doc_ids[:, :-1] == tgt_doc)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    if reduction == "token":
        return (nll * valid).sum() / valid.sum()
    n_docs = int(doc_ids.max().item()) + 1
    gid = torch.arange(B, device=logits.device).unsqueeze(1) * n_docs + tgt_doc.clamp(min=0)   # unique per (row, doc)
    g = gid[valid]
    sums = torch.zeros(B * n_docs, dtype=nll.dtype, device=nll.device).scatter_add(0, g, nll[valid])
    cnts = torch.zeros(B * n_docs, dtype=nll.dtype, device=nll.device).scatter_add(0, g, torch.ones_like(nll[valid]))
    present = cnts > 0                                                    # documents with at least one trained token
    return (sums[present] / cnts[present]).mean()


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    T: float = 1.0,
    kind: str = "forward",
    mask: Tensor = None,
) -> Tensor:
    """Per-position KL between temperature-softened distributions, times T^2, averaged over mask."""
    if kind not in ("forward", "reverse"):
        raise ValueError(f"kind must be 'forward' or 'reverse', got {kind!r}")
    log_s = F.log_softmax(student_logits / T, dim=-1)
    log_t = F.log_softmax(teacher_logits.detach() / T, dim=-1)            # the teacher is a constant target
    if kind == "forward":                                                 # KL(p_T || p_S): expectation under the teacher
        per_pos = (log_t.exp() * (log_t - log_s)).sum(-1)
    else:                                                                 # KL(p_S || p_T): expectation under the student
        per_pos = (log_s.exp() * (log_s - log_t)).sum(-1)
    if mask is None:
        mask = torch.ones_like(per_pos, dtype=torch.bool)
    w = mask.to(per_pos.dtype)
    return (T ** 2) * (per_pos * w).sum() / w.sum().clamp(min=1.0)
