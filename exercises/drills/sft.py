"""Drill (page 19): SFT plumbing.

Chat-template loss masking, sequence packing with a document mask, loss normalization under
packing, and the two knowledge-distillation losses. Run:
    pytest exercises/tests/test_sft.py
Conventions used by every function below:
  * A token is a *target* if the model is trained to predict it. ``loss_mask[i]`` refers to the
    token ``input_ids[i]`` itself (not shifted): position i - 1 predicts token i.
  * ``doc_ids`` labels which packed document a position belongs to (0, 1, 2, ...), and -1 on padding.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ChatTemplate:
    """A ChatML-like template. One message is rendered as header[role] + content + end + sep.

    header: role -> text that opens a turn, e.g. {"user": "<|im_start|>user\\n", ...}.
    end:    end-of-turn token text, e.g. "<|im_end|>". For a trained role it is a target.
    sep:    text after the end token, e.g. "\\n". Never a target (it is not generated at inference).
    bos:    text prepended once to the conversation. Never a target.
    train_roles: roles whose content and end token are targets.
    """
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
    """Render a conversation into token ids and a per-token training mask.

    Args:
        messages: [{"role": ..., "content": ...}, ...]; every role appears in template.header.
        tokenize: str -> list[int]. Call it on each rendered segment separately (bos, header,
            content, end, sep) and concatenate, so that segment boundaries are known exactly.
            Do not call it on the whole conversation string.
        template: see ChatTemplate.
        train_on: "all" trains every message whose role is in template.train_roles;
            "last" trains only the final message of the conversation (if its role is trainable).
    Returns:
        (input_ids, loss_mask), equal lengths. A position is True exactly when it belongs to the
        content or the end token of a trained message. Headers, sep text, bos, and every message
        of an untrained role (or an untrained turn) are False. Empty content adds no content
        tokens but its end token is still a target. A conversation with nothing trainable gives
        an all-False mask.
    Raises:
        ValueError: if train_on is neither "all" nor "last".
    """
    raise NotImplementedError


def pack(
    examples: Sequence[Tuple[Sequence[int], Sequence[bool]]],
    max_len: int,
    pad_id: int = 0,
) -> List[PackedSequence]:
    """Pack (input_ids, loss_mask) examples into fixed-length rows, next-fit, order preserved.

    Walk the examples in order. Append an example to the current row if it fits (used + len <=
    max_len); otherwise close the row and start a new one. Never reorder, split, or revisit a
    closed row. Empty examples are skipped.
    Each returned PackedSequence has length max_len: input_ids padded with pad_id, loss_mask
    False on padding, position_ids restarting at 0 for every document (0 on padding), doc_ids
    0, 1, 2, ... in order within the row (-1 on padding).
    Raises:
        ValueError: if a single example is longer than max_len.
    """
    raise NotImplementedError


def document_causal_mask(doc_ids: Tensor) -> Tensor:
    """Block-diagonal causal attention mask for packed rows.

    Args:
        doc_ids: (T,) or (B, T) integer document ids (padding has id -1).
    Returns:
        bool tensor (T, T) or (B, T, T); entry [..., i, j] is True iff query i may attend to key j,
        i.e. j <= i and doc_ids[..., i] == doc_ids[..., j]. Padding positions share the id -1, so
        they attend only to earlier padding (and to themselves), never to real tokens.
    """
    raise NotImplementedError


def packed_lm_loss(
    logits: Tensor,
    input_ids: Tensor,
    loss_mask: Tensor,
    doc_ids: Tensor,
    reduction: str = "token",
) -> Tensor:
    """Next-token cross-entropy on packed rows, averaged per token or per document.

    Args:
        logits: (B, T, V). Position t predicts input_ids[:, t + 1].
        input_ids: (B, T) long. loss_mask: (B, T) bool (unshifted, see module docstring).
        doc_ids: (B, T) long, -1 on padding.
        reduction: "token" or "sequence".
    A target token t + 1 counts only if loss_mask[t + 1] is True, doc_ids[t + 1] >= 0, and
    doc_ids[t] == doc_ids[t + 1] (a document's first token is never predicted from the previous
    document's last token).
      * "token": sum of NLL over all counted targets in the whole batch / number of counted targets.
      * "sequence": mean NLL inside each (row, document) that has at least one counted target,
        then the unweighted mean of those per-document means over the whole batch.
    Returns a scalar tensor (0.0 if nothing is counted).
    Raises:
        ValueError: if reduction is neither "token" nor "sequence".
    """
    raise NotImplementedError


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    T: float = 1.0,
    kind: str = "forward",
    mask: Tensor = None,
) -> Tensor:
    """Temperature knowledge-distillation loss over the vocabulary axis.

    Args:
        student_logits, teacher_logits: (..., V), finite. The teacher is a constant (no gradient).
        T: temperature. Both distributions are softmax(logits / T).
        kind: "forward" is KL(p_T || p_S) (expectation under the teacher);
              "reverse" is KL(p_S || p_T) (expectation under the student).
        mask: bool tensor of shape logits.shape[:-1]; True positions are averaged, others ignored.
            None means every position counts.
    Returns:
        T**2 * (mean over counted positions of the per-position KL), a scalar. The T**2 factor
        keeps gradient magnitudes comparable across temperatures. Returns 0.0 if no position counts.
    Raises:
        ValueError: if kind is neither "forward" nor "reverse".
    Use log_softmax, not softmax followed by log.
    """
    raise NotImplementedError
