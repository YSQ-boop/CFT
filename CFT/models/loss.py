from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_selected_kl_reweight, masked_mean


def get_global_top_mask(values: torch.Tensor, response_mask: torch.Tensor, top_ratio: float) -> torch.Tensor:
    """Select the global top-ratio tokens among all valid response tokens in a batch."""
    if not 0.0 <= top_ratio <= 1.0:
        raise ValueError(f"top_ratio must be in [0, 1], got {top_ratio}")

    flat_values = values.flatten()
    flat_mask = response_mask.flatten().bool()
    selected_values = flat_values[flat_mask]
    if selected_values.numel() == 0 or top_ratio == 0.0:
        return torch.zeros_like(values, dtype=torch.long)

    top_k = max(1, int(len(selected_values) * top_ratio + 0.9999))  # ceil
    _, topk_idx = torch.topk(selected_values, k=top_k)

    response_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
    top_positions = response_positions[topk_idx]

    flat_out = torch.zeros_like(flat_values, dtype=torch.long)
    flat_out[top_positions] = 1
    return flat_out.view_as(values)


def get_global_entropy_top_mask(entropy, response_mask, top_ratio=0.2):
    """
    Select the top `top_ratio` high-entropy tokens among all response tokens in a batch.

    Args:
        entropy: [B, S] tensor of token entropies.
        response_mask: [B, S] tensor (1 = response token, 0 = non-response).
        top_ratio: fraction of response tokens to keep (e.g. 0.2 = top 20%).

    Returns:
        entropy_top_mask: [B, S] binary mask (1 = selected top entropy token)
    """
    return get_global_top_mask(entropy, response_mask, top_ratio=top_ratio)


def _safe_masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)

    masked = tensor * mask
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return masked.sum(dim=dim) / denom


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: float = None,
        use_adv_shaping: bool = False,
        enable_gspo_token: bool = False,
        vllm_is_correction_type: str = "tis",
        student_entropy_top_ratio: float = 1.0,
        kl_top_ratio: float = 0.1,
        entropy_diff_top_ratio: float = 0.0,
        teacher_entropy_top_ratio: float = 0.0,
        enable_adv_kl_reweight: bool = False,
        adv_kl_weight_clip: float = 0.2,
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.use_adv_shaping = use_adv_shaping
        self.enable_gspo_token = enable_gspo_token
        self.vllm_is_correction_type = vllm_is_correction_type
        self.student_entropy_top_ratio = student_entropy_top_ratio
        self.kl_top_ratio = kl_top_ratio
        self.entropy_diff_top_ratio = entropy_diff_top_ratio
        self.teacher_entropy_top_ratio = teacher_entropy_top_ratio
        self.enable_adv_kl_reweight = enable_adv_kl_reweight
        self.adv_kl_weight_clip = adv_kl_weight_clip

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if not 0.0 < self.student_entropy_top_ratio <= 1.0:
            raise ValueError(
                "student_entropy_top_ratio must be in (0, 1], "
                f"got {self.student_entropy_top_ratio}"
            )
        if not 0.0 <= self.kl_top_ratio <= 1.0:
            raise ValueError(f"kl_top_ratio must be in [0, 1], got {self.kl_top_ratio}")
        if not 0.0 <= self.entropy_diff_top_ratio <= 1.0:
            raise ValueError(f"entropy_diff_top_ratio must be in [0, 1], got {self.entropy_diff_top_ratio}")
        if not 0.0 <= self.teacher_entropy_top_ratio <= 1.0:
            raise ValueError(
                f"teacher_entropy_top_ratio must be in [0, 1], got {self.teacher_entropy_top_ratio}"
            )
        if self.adv_kl_weight_clip < 0.0:
            raise ValueError(f"adv_kl_weight_clip must be >= 0, got {self.adv_kl_weight_clip}")

        if self.use_adv_shaping:
            self.alpha = 0.4
            self.kappa = 2

    # def forward(
    #     self,
    #     log_probs: torch.Tensor,
    #     old_log_probs: torch.Tensor,
    #     advantages: torch.Tensor,
    #     action_mask: Optional[torch.Tensor] = None,
    #     rollout_log_probs: Optional[torch.Tensor] = None,
    #     entropy: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     if self.policy_loss_type == "ppo":
    #         log_ratio = log_probs - old_log_probs
    #         ratio = log_ratio.exp()
    #     elif self.policy_loss_type == "gspo":
    #         # GSPO: https://arxiv.org/pdf/2507.18071
    #         if self.enable_vllm_is_correction:
    #             log_ratio = log_probs - rollout_log_probs
    #         else:
    #             log_ratio = log_probs - old_log_probs
    #         ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
    #         ratio = ratio.exp().unsqueeze(-1) * action_mask
    #     else:
    #         raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

    #     if entropy is not None and self.use_adv_shaping:
    #             shaping_term = torch.min(
    #                 self.alpha * entropy.detach(),
    #                 advantages.abs() / self.kappa
    #             )
    #             advantages = advantages + shaping_term

    #     surr1 = ratio * advantages
    #     surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

    #     if self.dual_clip is None:
    #         # Standard PPO
    #         loss = -torch.min(surr1, surr2)
    #     else:
    #         # Standard PPO clipping
    #         clip1 = torch.min(surr1, surr2)
    #         # Dual-clip: additional lower bound for negative advantages
    #         clip2 = torch.max(clip1, self.dual_clip * advantages)
    #         # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
    #         loss = -torch.where(advantages < 0, clip2, clip1)

    #     # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    #     vllm_kl = None
    #     if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
    #         vllm_is = torch.exp(old_log_probs - rollout_log_probs).clamp(max=self.vllm_is_truncated_threshold).detach()
    #         loss = vllm_is * loss
    #         vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

    #     loss = (
    #         masked_mean(loss, action_mask, dim=None)
    #         if self.token_level_loss
    #         else masked_mean(loss, action_mask, dim=-1).mean()
    #     )
    #     clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
    #     ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
    #     return loss, clip_ratio, ppo_kl, vllm_kl

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        kl: Optional[torch.Tensor] = None,
        teacher_entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure mask exists and is float for math
        if action_mask is None:
            action_mask = torch.ones_like(log_probs, dtype=log_probs.dtype)
        else:
            action_mask = action_mask.to(dtype=log_probs.dtype)
        effective_action_mask = action_mask
        aux_info = {}

        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()

        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                if rollout_log_probs is None:
                    raise ValueError("rollout_log_probs must be provided when enable_vllm_is_correction is True.")
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs

            # sequence-level log-ratio (mean over valid tokens)
            denom = action_mask.sum(dim=-1).clamp_min(1.0)  # [B]
            seq_log_ratio = (log_ratio * action_mask).sum(dim=-1) / denom  # [B]
            seq_ratio = seq_log_ratio.exp()  # s_i(theta) in prob space, [B]

            if self.enable_gspo_token:
                # === GSPO-token ===
                # s_{i,t}(theta) = sg[s_i(theta)] * pi_theta(y_{i,t}|...) / sg[pi_theta(y_{i,t}|...)]
                # pi / sg(pi) implemented as exp(logp - detach(logp)) -> value=1, gradient flows through logp
                token_correction = torch.exp(log_probs - log_probs.detach())  # [B, T], value ~ 1
                ratio = seq_ratio.detach().unsqueeze(-1) * token_correction * action_mask  # [B, T]
            else:
                # === original GSPO (sequence-level ratio expanded to tokens) ===
                ratio = seq_ratio.unsqueeze(-1) * action_mask  # [B, T]

        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        if entropy is not None and self.use_adv_shaping:
            shaping_term = torch.min(
                self.alpha * entropy.detach(),
                advantages.abs() / self.kappa,
            )
            advantages = advantages + shaping_term

        top_masks = []
        if self.student_entropy_top_ratio < 1.0:
            if entropy is None:
                raise ValueError("entropy must be provided when student_entropy_top_ratio < 1.0")
            if entropy.shape != action_mask.shape:
                raise ValueError(
                    f"entropy shape must match action_mask shape, got {entropy.shape} vs {action_mask.shape}"
                )
            entropy_top_mask = get_global_entropy_top_mask(
                entropy,
                action_mask,
                top_ratio=self.student_entropy_top_ratio,
            )
            top_masks.append(entropy_top_mask.to(dtype=action_mask.dtype))

        if self.kl_top_ratio > 0.0:
            if kl is None:
                raise ValueError("kl must be provided when kl_top_ratio > 0")
            if kl.shape != action_mask.shape:
                raise ValueError(f"kl shape must match action_mask shape, got {kl.shape} vs {action_mask.shape}")
            kl_top_mask = get_global_top_mask(kl, action_mask, top_ratio=self.kl_top_ratio)
            top_masks.append(kl_top_mask.to(dtype=action_mask.dtype))

        if self.entropy_diff_top_ratio > 0.0:
            if entropy is None:
                raise ValueError("entropy must be provided when entropy_diff_top_ratio > 0")
            if teacher_entropy is None:
                raise ValueError("teacher_entropy must be provided when entropy_diff_top_ratio > 0")
            if entropy.shape != action_mask.shape:
                raise ValueError(
                    f"entropy shape must match action_mask shape, got {entropy.shape} vs {action_mask.shape}"
                )
            if teacher_entropy.shape != action_mask.shape:
                raise ValueError(
                    "teacher_entropy shape must match action_mask shape, "
                    f"got {teacher_entropy.shape} vs {action_mask.shape}"
                )
            entropy_diff = entropy - teacher_entropy
            entropy_diff_top_mask = get_global_top_mask(
                entropy_diff,
                action_mask,
                top_ratio=self.entropy_diff_top_ratio,
            )
            top_masks.append(entropy_diff_top_mask.to(dtype=action_mask.dtype))

        if self.teacher_entropy_top_ratio > 0.0:
            if teacher_entropy is None:
                raise ValueError("teacher_entropy must be provided when teacher_entropy_top_ratio > 0")
            if teacher_entropy.shape != action_mask.shape:
                raise ValueError(
                    "teacher_entropy shape must match action_mask shape, "
                    f"got {teacher_entropy.shape} vs {action_mask.shape}"
                )
            teacher_entropy_top_mask = get_global_top_mask(
                teacher_entropy,
                action_mask,
                top_ratio=self.teacher_entropy_top_ratio,
            )
            top_masks.append(teacher_entropy_top_mask.to(dtype=action_mask.dtype))

        if top_masks:
            merged_top_mask = torch.zeros_like(action_mask)
            for top_mask in top_masks:
                merged_top_mask = torch.maximum(merged_top_mask, top_mask)
            effective_action_mask = action_mask * merged_top_mask

        if self.enable_adv_kl_reweight:
            if kl is None:
                raise ValueError("kl must be provided when enable_adv_kl_reweight is True")
            if kl.shape != action_mask.shape:
                raise ValueError(f"kl shape must match action_mask shape, got {kl.shape} vs {action_mask.shape}")
            advantages, adv_kl_weight_mean = apply_selected_kl_reweight(
                advantages,
                kl,
                effective_action_mask,
                self.adv_kl_weight_clip,
            )
            aux_info["adv_kl_weight_mean"] = adv_kl_weight_mean

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = _safe_masked_mean(rollout_log_probs - old_log_probs, effective_action_mask, dim=None)

        loss = (
            _safe_masked_mean(loss, effective_action_mask, dim=None)
            if self.token_level_loss
            else _safe_masked_mean(loss, effective_action_mask, dim=-1).mean()
        )
        clip_ratio = _safe_masked_mean(torch.lt(surr2, surr1).float(), effective_action_mask, dim=None)
        ppo_kl = _safe_masked_mean(-log_ratio.detach(), effective_action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl, aux_info


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
