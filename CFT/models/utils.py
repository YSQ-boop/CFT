from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_z3_leaf_modules(model: nn.Module) -> None:
    """Auto-detect and set DeepSpeed ZeRO3 leaf modules.

    ZeRO3 prefetches submodule parameters assuming a fixed module traversal order.
    This breaks for:
      - MoE: dynamic expert routing makes prefetch unpredictable.
        (https://github.com/microsoft/DeepSpeed/pull/4966)
      - Hybrid architectures (e.g., Qwen3.5): same decoder layer class but different
        child submodules per instance (self_attn vs linear_attn).

    Marking these as z3 leaves forces whole-module allgather instead of per-submodule
    prefetch, fixing the issue at the cost of slightly higher peak memory.
    """
    z3_leaf_classes = set()
    child_sigs: dict[type, frozenset[str]] = {}

    for m in model.modules():
        # MoE: dynamic expert routing
        if "SparseMoeBlock" in m.__class__.__name__:
            z3_leaf_classes.add(m.__class__)
            continue

        # Hybrid: same class, different child submodules across instances
        cls = m.__class__
        children = frozenset(name for name, _ in m.named_children())
        if not children:
            continue
        if cls in child_sigs:
            if child_sigs[cls] != children:
                z3_leaf_classes.add(cls)
        else:
            child_sigs[cls] = children

    if z3_leaf_classes:
        deepspeed.utils.set_z3_leaf_modules(model, list(z3_leaf_classes))
        for cls in z3_leaf_classes:
            print(f"Setting zero3 leaf: {cls.__name__}")


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl
    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward

    return reward


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/CFT/CFT/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def full_log_probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    return F.log_softmax(logits, dim=-1, dtype=torch.float32)


def topk_log_probs_from_logits(
    logits: torch.Tensor,
    topk: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > logits.size(-1):
        raise ValueError(f"topk ({topk}) must be <= vocab size ({logits.size(-1)})")

    if temperature != 1.0:
        logits = logits / temperature
    topk_logits, topk_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return topk_token_ids, topk_logits - log_z


def gather_log_probs_at_token_ids(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    gathered_logits = torch.gather(logits, dim=-1, index=token_ids)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return gathered_logits - log_z


def renormalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def add_tail_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log_prob = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log_prob], dim=-1)


def prepare_topk_log_probs_for_divergence(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    add_tail: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if add_tail:
        return add_tail_log_probs(student_topk_log_probs), add_tail_log_probs(teacher_topk_log_probs)
    return renormalize_log_probs(student_topk_log_probs), renormalize_log_probs(teacher_topk_log_probs)


def compute_directional_distribution_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    direction: str = "student_to_teacher",
) -> torch.Tensor:
    if direction == "student_to_teacher":
        input_log_probs = teacher_log_probs
        target_log_probs = student_log_probs
    elif direction == "teacher_to_student":
        input_log_probs = student_log_probs
        target_log_probs = teacher_log_probs
    else:
        raise ValueError(
            f"direction must be one of: student_to_teacher, teacher_to_student; got {direction}"
        )

    kl_loss = F.kl_div(input_log_probs, target_log_probs, reduction="none", log_target=True)
    log_ratio = kl_loss.sum(-1)
    return log_ratio.clamp(min=-10, max=10)


def compute_distribution_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if alpha == 0.0:
        return compute_directional_distribution_kl(
            student_log_probs,
            teacher_log_probs,
            direction="teacher_to_student",
        )
    elif alpha == 1.0:
        return compute_directional_distribution_kl(
            student_log_probs,
            teacher_log_probs,
            direction="student_to_teacher",
        )
    else:
        alpha_tensor = student_log_probs.new_tensor(alpha)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-alpha_tensor),
                    teacher_log_probs + torch.log(alpha_tensor),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        kl_loss = torch.lerp(kl_student, kl_teacher, alpha_tensor)

    log_ratio = kl_loss.sum(-1)
    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


def compute_directional_topk_distribution_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    direction: str = "student_to_teacher",
    add_tail: bool = True,
) -> torch.Tensor:
    student_distill_log_probs, teacher_distill_log_probs = prepare_topk_log_probs_for_divergence(
        student_topk_log_probs,
        teacher_topk_log_probs,
        add_tail=add_tail,
    )
    return compute_directional_distribution_kl(
        student_distill_log_probs,
        teacher_distill_log_probs,
        direction=direction,
    )


def compute_topk_distribution_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    alpha: float = 1.0,
    add_tail: bool = True,
) -> torch.Tensor:
    student_distill_log_probs, teacher_distill_log_probs = prepare_topk_log_probs_for_divergence(
        student_topk_log_probs,
        teacher_topk_log_probs,
        add_tail=add_tail,
    )
    log_ratio = compute_distribution_kl(student_distill_log_probs, teacher_distill_log_probs, alpha=alpha)
    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


def compute_entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_mean_with_fallback(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
    fallback: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)

    mask = mask.to(dtype=tensor.dtype)
    denom = mask.sum(dim=dim)
    mean = (tensor * mask).sum(dim=dim) / denom.clamp_min(1.0)
    if fallback is None:
        fallback = torch.zeros_like(mean)
    elif not isinstance(fallback, torch.Tensor):
        fallback = torch.full_like(mean, fallback)
    return torch.where(denom > 0, mean, fallback)


def select_advantage_conditioned_kl(
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    kl_s2t: torch.Tensor,
    kl_t2s: torch.Tensor,
    zero_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if advantages.shape != action_mask.shape:
        raise ValueError(f"advantages shape must match action_mask shape, got {advantages.shape} vs {action_mask.shape}")
    if kl_s2t.shape != action_mask.shape or kl_t2s.shape != action_mask.shape:
        raise ValueError(
            "Directional KL tensors must match action_mask shape, "
            f"got {kl_s2t.shape} and {kl_t2s.shape} vs {action_mask.shape}"
        )

    seq_advantages = masked_mean_with_fallback(advantages.float(), action_mask, dim=-1)
    direction = torch.zeros_like(seq_advantages, dtype=torch.int64)
    positive_mask = seq_advantages > zero_epsilon
    negative_mask = seq_advantages < -zero_epsilon
    direction[positive_mask] = 1
    direction[negative_mask] = -1

    selected_kl = torch.zeros_like(kl_s2t, dtype=torch.float32)
    selected_kl = torch.where(positive_mask.unsqueeze(-1), kl_s2t.float(), selected_kl)
    selected_kl = torch.where(negative_mask.unsqueeze(-1), kl_t2s.float(), selected_kl)

    action_mask_float = action_mask.to(dtype=selected_kl.dtype)
    selected_kl = selected_kl * action_mask_float

    return selected_kl, direction, seq_advantages


def apply_selected_kl_reweight(
    advantages: torch.Tensor,
    selected_kl: torch.Tensor,
    weight_mask: torch.Tensor,
    weight_clip: float,
    zero_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_clip < 0:
        raise ValueError(f"weight_clip must be non-negative, got {weight_clip}")

    if advantages.shape != selected_kl.shape:
        raise ValueError(
            f"advantages shape must match selected_kl shape, got {advantages.shape} vs {selected_kl.shape}"
        )
    if weight_mask.shape != selected_kl.shape:
        raise ValueError(
            f"weight_mask shape must match selected_kl shape, got {weight_mask.shape} vs {selected_kl.shape}"
        )

    weight_mask_bool = weight_mask.bool()
    selected_kl = selected_kl.float() * weight_mask.to(dtype=torch.float32)
    selected_kl_mean = masked_mean_with_fallback(selected_kl, weight_mask, dim=-1)
    normalized_kl = selected_kl / selected_kl_mean.unsqueeze(-1).clamp_min(zero_epsilon)
    weights = torch.where(weight_mask_bool, normalized_kl, torch.ones_like(normalized_kl))
    weights = torch.where((selected_kl_mean > zero_epsilon).unsqueeze(-1), weights, torch.ones_like(weights))

    clipped_weights = weights.clamp(min=1.0 - weight_clip, max=1.0 + weight_clip)
    reweighted_advantages = advantages * clipped_weights.to(dtype=advantages.dtype)
    weight_mean = masked_mean_with_fallback(clipped_weights, weight_mask, dim=-1, fallback=1.0)
    return reweighted_advantages, weight_mean


def apply_advantage_conditioned_kl_reweight(
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    kl_s2t: torch.Tensor,
    kl_t2s: torch.Tensor,
    weight_clip: float,
    zero_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_kl, direction, seq_advantages = select_advantage_conditioned_kl(
        advantages,
        action_mask,
        kl_s2t,
        kl_t2s,
        zero_epsilon=zero_epsilon,
    )
    reweighted_advantages, weight_mean = apply_selected_kl_reweight(
        advantages,
        selected_kl,
        action_mask,
        weight_clip,
        zero_epsilon=zero_epsilon,
    )
    return reweighted_advantages, selected_kl, direction, weight_mean, seq_advantages


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
