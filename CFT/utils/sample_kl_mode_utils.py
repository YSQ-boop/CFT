VALID_SAMPLE_KL_MODES = frozenset({"approx", "topk", "full"})
TEACHER_PROMPT_SAMPLE_KL_MODES = frozenset({"topk", "full"})


def ensure_valid_sample_kl_mode(sample_kl_mode: str) -> str:
    if sample_kl_mode not in VALID_SAMPLE_KL_MODES:
        raise ValueError(f"Invalid sample_kl_mode: {sample_kl_mode}")
    return sample_kl_mode


def sample_kl_uses_teacher_prompt(sample_kl_mode: str) -> bool:
    return ensure_valid_sample_kl_mode(sample_kl_mode) in TEACHER_PROMPT_SAMPLE_KL_MODES


def sample_kl_requires_reference_model(sample_kl_mode: str) -> bool:
    return ensure_valid_sample_kl_mode(sample_kl_mode) in TEACHER_PROMPT_SAMPLE_KL_MODES


def need_policy_base_action_log_probs(
    sample_kl_mode: str,
    *,
    has_reference_model: bool,
    init_kl_coef: float,
    use_kl_loss: bool,
    use_kl_reward: bool,
) -> bool:
    return (
        has_reference_model
        and ensure_valid_sample_kl_mode(sample_kl_mode) == "approx"
        and (init_kl_coef > 0 or use_kl_loss or use_kl_reward)
    )


def validate_sample_kl_mode_args(args) -> None:
    sample_kl_mode = ensure_valid_sample_kl_mode(args.sample_kl_mode)
    if sample_kl_mode == "topk" and (args.sample_kl_topk is None or args.sample_kl_topk <= 0):
        raise ValueError("--sample_kl_topk must be a positive integer when --sample_kl_mode=topk")
    if sample_kl_mode in TEACHER_PROMPT_SAMPLE_KL_MODES and args.use_kl_loss:
        raise ValueError(f"--sample_kl_mode={sample_kl_mode} does not support --use_kl_loss")
    if getattr(args, "enable_adv_kl_reweight", False) and sample_kl_mode not in TEACHER_PROMPT_SAMPLE_KL_MODES:
        raise ValueError("--enable_adv_kl_reweight requires --sample_kl_mode to be one of: topk, full")
    if getattr(args, "policy_loss_entropy_diff_top_ratio", 0.0) > 0 and sample_kl_mode not in TEACHER_PROMPT_SAMPLE_KL_MODES:
        raise ValueError(
            "--policy_loss_entropy_diff_top_ratio requires --sample_kl_mode to be one of: topk, full"
        )
    if (
        getattr(args, "policy_loss_teacher_entropy_top_ratio", 0.0) > 0
        and sample_kl_mode not in TEACHER_PROMPT_SAMPLE_KL_MODES
    ):
        raise ValueError(
            "--policy_loss_teacher_entropy_top_ratio requires --sample_kl_mode to be one of: topk, full"
        )
