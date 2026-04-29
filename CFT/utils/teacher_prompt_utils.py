import torch


def build_teacher_user_content(chat: str, label: str) -> str:
    return (
        chat
        + "\nHere is a reference solution:\n"
        + label
        + "After understanding the reference solution, please try to solve this problem\n"
        + "using your own approach below:"
    )


def build_teacher_reference_token_layout(teacher_prompt_ids, response_token_ids, max_length: int):
    prompt_budget = max_length - len(response_token_ids)
    if prompt_budget <= 0:
        raise ValueError("Teacher prompt reconstruction requires room for at least one prompt token")

    teacher_prompt_ids = list(teacher_prompt_ids)[-prompt_budget:]
    if not teacher_prompt_ids:
        raise ValueError("Teacher prompt is empty after truncation")

    response_token_ids = list(response_token_ids)
    sequence = teacher_prompt_ids + response_token_ids
    attention_mask = [1] * len(sequence)
    full_action_mask = [0] * len(sequence)
    full_action_mask[len(teacher_prompt_ids) :] = [1] * len(response_token_ids)
    return sequence, attention_mask, full_action_mask[1:]


def build_response_only_mask(full_action_mask: torch.Tensor) -> torch.Tensor:
    if full_action_mask.ndim != 2:
        raise ValueError(f"full_action_mask must be rank 2, got shape {tuple(full_action_mask.shape)}")

    response_lengths = full_action_mask.long().sum(dim=-1)
    max_response_length = int(response_lengths.max().item())
    if max_response_length <= 0:
        raise ValueError("Response-only extraction requires at least one response token")

    response_mask = full_action_mask.new_zeros((full_action_mask.size(0), max_response_length))
    for row_idx, response_length in enumerate(response_lengths.tolist()):
        if response_length > 0:
            response_mask[row_idx, :response_length] = 1
    return response_mask


def extract_response_only_tensor(
    tensor: torch.Tensor,
    full_action_mask: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"tensor must be rank >= 2, got shape {tuple(tensor.shape)}")
    if tensor.shape[:2] != full_action_mask.shape:
        raise ValueError(
            f"tensor batch/time dims {tuple(tensor.shape[:2])} must match action mask {tuple(full_action_mask.shape)}"
        )

    if response_mask is None:
        response_mask = build_response_only_mask(full_action_mask)

    if response_mask.ndim != 2 or response_mask.shape[0] != full_action_mask.shape[0]:
        raise ValueError(
            f"response_mask must be rank 2 with batch {full_action_mask.shape[0]}, got {tuple(response_mask.shape)}"
        )

    response_lengths = response_mask.long().sum(dim=-1).tolist()
    response_shape = (tensor.size(0), response_mask.size(1), *tensor.shape[2:])
    response_tensor = tensor.new_zeros(response_shape)

    for row_idx, response_length in enumerate(response_lengths):
        if response_length <= 0:
            continue
        row_values = tensor[row_idx][full_action_mask[row_idx].to(device=tensor.device).bool()]
        if row_values.shape[0] != response_length:
            raise ValueError(
                f"Response length mismatch at row {row_idx}: "
                f"mask expects {response_length}, tensor extraction got {row_values.shape[0]}"
            )
        response_tensor[row_idx, :response_length] = row_values

    return response_tensor


def scatter_response_tensor_to_full_action_mask(
    response_tensor: torch.Tensor,
    full_action_mask: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if response_tensor.ndim < 2:
        raise ValueError(f"response_tensor must be rank >= 2, got shape {tuple(response_tensor.shape)}")
    if full_action_mask.ndim != 2 or full_action_mask.shape[0] != response_tensor.shape[0]:
        raise ValueError(
            f"full_action_mask must be rank 2 with batch {response_tensor.shape[0]}, got {tuple(full_action_mask.shape)}"
        )

    if response_mask is None:
        response_mask = build_response_only_mask(full_action_mask)

    if response_mask.shape[:2] != response_tensor.shape[:2]:
        raise ValueError(
            f"response_tensor batch/time dims {tuple(response_tensor.shape[:2])} "
            f"must match response_mask {tuple(response_mask.shape)}"
        )

    full_shape = (response_tensor.size(0), full_action_mask.size(1), *response_tensor.shape[2:])
    full_tensor = response_tensor.new_zeros(full_shape)
    response_lengths = response_mask.long().sum(dim=-1).tolist()

    for row_idx, response_length in enumerate(response_lengths):
        if response_length <= 0:
            continue
        target_positions = full_action_mask[row_idx].to(device=response_tensor.device).bool()
        if int(target_positions.sum().item()) != response_length:
            raise ValueError(
                f"Response length mismatch at row {row_idx}: "
                f"target action mask has {int(target_positions.sum().item())} positions, "
                f"response tensor has {response_length}"
            )
        full_tensor[row_idx][target_positions] = response_tensor[row_idx, :response_length]

    return full_tensor
