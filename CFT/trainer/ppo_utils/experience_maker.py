
import heapq
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, List, Optional, Tuple, Union

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from CFT.utils.sample_kl_mode_utils import (
    ensure_valid_sample_kl_mode,
    need_policy_base_action_log_probs as should_fetch_policy_base_action_log_probs,
    sample_kl_requires_reference_model,
    sample_kl_uses_teacher_prompt,
)
from CFT.models.utils import (
    apply_advantage_conditioned_kl_reweight,
    compute_approx_kl,
    compute_directional_distribution_kl,
    compute_directional_topk_distribution_kl,
    compute_distribution_kl,
    compute_entropy_from_log_probs,
    compute_reward,
    compute_topk_distribution_kl,
    masked_mean,
    prepare_topk_log_probs_for_divergence,
)
from CFT.trainer.ppo_utils.length_penalty import apply_length_penalties
from CFT.trainer.ray.launcher import RayActorGroup
from CFT.trainer.ray.vllm_engine import batch_vllm_engine_call
from CFT.utils.logging_utils import init_logger
from CFT.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from CFT.utils.teacher_prompt_utils import (
    build_response_only_mask,
    build_teacher_reference_token_layout,
    extract_response_only_tensor,
    scatter_response_tensor_to_full_action_mask,
)
from CFT.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


def _flatten_batched_refs(refs, duplicate_factor: int):
    return sum(ray.get(refs)[::duplicate_factor], [])


def _split_prompt_batch(batch):
    if len(batch) not in {3, 4}:
        raise ValueError(f"Unexpected prompt batch size: {len(batch)}")
    _, prompts, labels = batch[:3]
    teacher_prompts = batch[3] if len(batch) == 4 else None
    return prompts, labels, teacher_prompts


def _extract_response_token_ids(sequence: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    shifted_sequence = sequence[1 : action_mask.shape[-1] + 1]
    return shifted_sequence[action_mask.bool()]


def _build_teacher_reference_input(
    teacher_prompt: str,
    response_token_ids: torch.Tensor,
    sequence_dtype,
    tokenizer,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_prompt_ids = tokenizer(teacher_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    teacher_sequence, teacher_attention_mask, teacher_action_mask = build_teacher_reference_token_layout(
        teacher_prompt_ids.tolist(),
        response_token_ids.tolist(),
        max_length,
    )
    teacher_sequence = torch.tensor(teacher_sequence, dtype=sequence_dtype)
    teacher_attention_mask = torch.tensor(teacher_attention_mask, dtype=torch.long)
    teacher_action_mask = torch.tensor(teacher_action_mask, dtype=torch.long)
    return teacher_sequence, teacher_attention_mask, teacher_action_mask


def _build_teacher_reference_batches(samples_list: List["Experience"], tokenizer, max_length: int, pad_token_id: int):
    teacher_sequences_list = []
    teacher_attention_mask_list = []
    teacher_action_mask_list = []

    for samples in samples_list:
        if len(samples.teacher_prompts) != len(samples.sequences):
            raise ValueError("Teacher prompts must align with batch sequences")

        batch_sequences = []
        batch_attention_masks = []
        batch_action_masks = []
        for teacher_prompt, sequence, action_mask in zip(samples.teacher_prompts, samples.sequences, samples.action_mask):
            response_token_ids = _extract_response_token_ids(sequence, action_mask)
            teacher_sequence, teacher_attention_mask, teacher_action_mask = _build_teacher_reference_input(
                teacher_prompt,
                response_token_ids,
                sequence.dtype,
                tokenizer,
                max_length,
            )
            if response_token_ids.numel() != action_mask.bool().sum().item():
                raise ValueError("Teacher/reference response token count must match the student action mask")
            batch_sequences.append(teacher_sequence.unsqueeze(0))
            batch_attention_masks.append(teacher_attention_mask.unsqueeze(0))
            batch_action_masks.append(teacher_action_mask.unsqueeze(0))

        teacher_sequences_list.append(zero_pad_sequences(batch_sequences, side="right", value=pad_token_id))
        teacher_attention_mask_list.append(zero_pad_sequences(batch_attention_masks, side="right", value=0))
        teacher_action_mask_list.append(zero_pad_sequences(batch_action_masks, side="right", value=0))

    return teacher_sequences_list, teacher_attention_mask_list, teacher_action_mask_list


def _normalize_response_distribution_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    # Packed gather can leave an extra singleton axis before the distribution axis:
    # [B, R, 1, K] or [B, R, 1, V]. KL expects [B, R, K/V].
    while tensor.ndim >= 4 and tensor.shape[-2] == 1:
        tensor = tensor.squeeze(-2)
    return tensor


def _normalize_kl_response_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    while tensor.ndim > 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _clear_advantage_conditioned_kl_tensors(samples: "Experience") -> None:
    for attr_name in ("_adv_kl_s2t", "_adv_kl_t2s"):
        if hasattr(samples, attr_name):
            delattr(samples, attr_name)


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    teacher_entropy: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None
    teacher_entropy: torch.Tensor = None

    prompts: list[str] = None
    teacher_prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        teacher_entropy=None,
        prompts=None,
        teacher_prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.teacher_entropy = teacher_entropy
        self.prompts = prompts or []
        self.teacher_prompts = teacher_prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)

def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
    prompts, labels = [], []
    teacher_prompts = None
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            batch_prompts, batch_labels, batch_teacher_prompts = _split_prompt_batch(next(dataloader_iter))
            remaining = num_prompts - len(prompts)
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
            if batch_teacher_prompts is not None:
                if teacher_prompts is None:
                    teacher_prompts = []
                teacher_prompts.extend(batch_teacher_prompts[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, teacher_prompts, exhausted


class SamplesGenerator:
    """Stateless sample generator: pulls prompts and dispatches to rollout workers."""

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        vllm_engines: List,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._eval_dataloader_iter,
            num_prompts=len(self.eval_dataloader),
            dynamic_filtering=False,
            **generate_kwargs,
        )

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        self._eval_dataloader_iter = None

        return experiences

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            dynamic_filtering=self.args.dynamic_filtering,
            **generate_kwargs,
        )

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        filter_pass_rate = None
        if self.args.dynamic_filtering and prompts_consumed:
            filter_pass_rate = self.args.rollout_batch_size / prompts_consumed * 100

        if exhausted:
            self._dataloader_iter = None
            logger.info("Prompt dataloader is exhausted.")

        return experiences, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], int, bool]:
        """Generate a batch of Experiences with optional reward filtering."""
        prompts_consumed = 0
        prompts, labels, teacher_prompts, exhausted = _collect_prompt_batch(dataloader_iter, num_prompts)
        # Stop early if the prompt source is fully consumed.
        if exhausted:
            return [], prompts_consumed, exhausted

        pending_refs, pending_teacher_prompts = self._dispatch_prompts_to_vllm(
            prompts,
            labels,
            teacher_prompts,
            **generate_kwargs,
        )
        prompts_consumed += len(prompts)

        accepted_experiences: List[Experience] = []
        pbar = tqdm(range(num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                teacher_prompt = pending_teacher_prompts.pop(ref.hex(), None)
                # Build Experience objects for each vLLM response returned from this worker.
                experiences = [
                    self._process_response_into_experience(response, teacher_prompt=teacher_prompt, **generate_kwargs)
                    for response in ray.get(ref)
                ]

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range
                    if not (min_r < avg_reward < max_r):
                        logger.info(
                            f"Filtered out: avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}), scores={[f'{s:.2f}' for s in scores]}"
                        )
                        experiences = []

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()

                # If rejected, request a new prompt to keep filling the batch.
                else:
                    # Pull another prompt when the current one fails filtering.
                    new_prompts, new_labels, new_teacher_prompts, exhausted = _collect_prompt_batch(dataloader_iter, 1)
                    prompts_consumed += len(new_prompts)
                    # Cancel outstanding work if the dataloader is drained.
                    if exhausted:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return [], prompts_consumed, True
                    # Otherwise dispatch the new prompt to keep filling the queue.
                    else:
                        new_refs, new_teacher_prompt_map = self._dispatch_prompts_to_vllm(
                            new_prompts,
                            new_labels,
                            new_teacher_prompts,
                            **generate_kwargs,
                        )
                        pending_refs.extend(new_refs)
                        pending_teacher_prompts.update(new_teacher_prompt_map)

        return accepted_experiences, prompts_consumed, exhausted

    def _dispatch_prompts_to_vllm(
        self,
        prompts: List[str],
        labels: List[str],
        teacher_prompts: Optional[List[str]] = None,
        **generate_kwargs,
    ) -> tuple[List, dict[str, str]]:
        """Send prompts to rollout executors and return Ray object refs."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            logprobs=1 if self.args.enable_vllm_is_correction else None,
        )
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Snapshot current pending rollout counts to balance upcoming work.
        pending_counts = ray.get([engine.get_num_unfinished_requests.remote() for engine in self.vllm_engines])
        engine_heap = [(count, idx) for idx, count in enumerate(pending_counts)]
        heapq.heapify(engine_heap)

        # Pre-compute engine assignment to keep loads even.
        engine_indices = []
        for _ in prompts:
            current_load, engine_idx = heapq.heappop(engine_heap)
            engine_indices.append(engine_idx)
            heapq.heappush(engine_heap, (current_load + self.args.n_samples_per_prompt, engine_idx))

        refs = []
        teacher_prompt_map = {}
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            # Spread work across engines/workers in load-aware order.
            llm_engine = self.vllm_engines[engine_indices[idx]]
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=self.args.n_samples_per_prompt,
            )
            refs.append(ref)
            if teacher_prompts is not None:
                teacher_prompt_map[ref.hex()] = teacher_prompts[idx]

        return refs, teacher_prompt_map

    def _process_response_into_experience(self, response, teacher_prompt=None, **generate_kwargs) -> Experience:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Base rollout fields from the output.
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)
        score_val = response.get("scores", None)

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_observation))
        # Mark the action span within the concatenated tokens.
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Truncate everything to the configured context window.
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        # Align rollout logprobs with the truncated action span.
        if response["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
        ones_indices = torch.where(action_mask)[0]
        response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
        total_length = attention_mask.float().sum()
        is_clipped = total_length >= truncate_length

        # Check if response was truncated (hit max_tokens limit, finish_reason == "length")
        is_truncated = response.get("truncated", False)

        info = {
            "response_length": torch.tensor([response_length]),
            "total_length": torch.tensor([total_length]),
            "response_clip_ratio": torch.tensor([is_clipped]),
            "truncated": torch.tensor([is_truncated]),
        }
        if reward_val is not None:
            info["reward"] = torch.tensor([reward_val])
        if score_val is not None:
            info["score"] = torch.tensor([score_val])

        # Convert extra logs to tensors for downstream consumers.
        extra_logs = response.get("extra_logs", {})
        for key, value in extra_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.flatten()[0].item()
            info[key] = torch.tensor([value])

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[response["prompt"]],
            teacher_prompts=[teacher_prompt] if teacher_prompt is not None else [],
            labels=[response["label"]],
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            scores=torch.tensor([score_val]) if score_val is not None else None,
            info=info,
        )


class RemoteExperienceMaker:
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.advantage_estimator
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        device = "cpu"
        enable_adv_kl_reweight = bool(getattr(args, "enable_adv_kl_reweight", False))

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sample_kl_mode = ensure_valid_sample_kl_mode(getattr(args, "sample_kl_mode", "approx"))
        need_teacher_entropy = (
            args.policy_loss_entropy_diff_top_ratio > 0.0
            or args.policy_loss_teacher_entropy_top_ratio > 0.0
        )
        teacher_prompt_enabled = sample_kl_uses_teacher_prompt(sample_kl_mode)
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        response_mask_list = [build_response_only_mask(action_mask) for action_mask in action_mask_list]
        teacher_reference_sequences_list = sequences_list
        teacher_reference_attention_mask_list = attention_mask_list
        teacher_reference_action_mask_list = action_mask_list

        if teacher_prompt_enabled:
            truncate_length = args.max_len or (args.prompt_max_len + args.generate_max_len)
            (
                teacher_reference_sequences_list,
                teacher_reference_attention_mask_list,
                teacher_reference_action_mask_list,
            ) = _build_teacher_reference_batches(
                samples_list,
                self.tokenizer,
                truncate_length,
                self.tokenizer.pad_token_id,
            )

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        use_reward_model = samples_list[0].rewards is None
        if use_reward_model:
            if self.reward_model_group is None:
                raise ValueError("reward_model_group is required when rewards are not precomputed")
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )
        else:
            r_refs = None

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and r_refs is not None:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        reference_model_group = self.initial_model_group
        if sample_kl_requires_reference_model(sample_kl_mode) and reference_model_group is None:
            raise ValueError(f"sample_kl_mode={sample_kl_mode} requires a reference model")
        if args.use_kl_reward and reference_model_group is None:
            raise ValueError("use_kl_reward requires a reference model")

        need_policy_base_action_log_probs = should_fetch_policy_base_action_log_probs(
            sample_kl_mode,
            has_reference_model=reference_model_group is not None,
            init_kl_coef=args.init_kl_coef,
            use_kl_loss=args.use_kl_loss,
            use_kl_reward=args.use_kl_reward,
        )
        action_log_probs_list = None
        base_action_log_probs_list = [None] * len(samples_list)
        actor_distribution_list = None
        teacher_distribution_list = None

        if sample_kl_mode == "approx":
            action_log_probs_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(action_log_probs_ref)
                ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

            if need_policy_base_action_log_probs:
                base_action_log_probs_ref = reference_model_group.async_run_method_batch(
                    method_name="forward",
                    sequences=sequences_list,
                    action_mask=action_mask_list,
                    attention_mask=attention_mask_list,
                )

                if args.colocate_all_models or args.colocate_actor_ref:
                    ray.get(base_action_log_probs_ref)
                    ray.get(reference_model_group.async_run_method(method_name="empty_cache"))

                base_action_log_probs_list = _flatten_batched_refs(base_action_log_probs_ref, duplicate_factor)

            action_log_probs_list = _flatten_batched_refs(action_log_probs_ref, duplicate_factor)
        elif sample_kl_mode == "full":
            actor_distribution_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward_with_full_log_probs",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(actor_distribution_ref)
                ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

            teacher_distribution_ref = reference_model_group.async_run_method_batch(
                method_name="forward_with_full_log_probs",
                sequences=teacher_reference_sequences_list,
                action_mask=teacher_reference_action_mask_list,
                attention_mask=teacher_reference_attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(teacher_distribution_ref)
                ray.get(reference_model_group.async_run_method(method_name="empty_cache"))

            actor_distribution_list = _flatten_batched_refs(actor_distribution_ref, duplicate_factor)
            teacher_distribution_list = _flatten_batched_refs(teacher_distribution_ref, duplicate_factor)
            action_log_probs_list = [result["action_log_probs"] for result in actor_distribution_list]
            for actor_distribution, teacher_distribution, student_action_mask, teacher_action_mask, response_mask in zip(
                actor_distribution_list,
                teacher_distribution_list,
                action_mask_list,
                teacher_reference_action_mask_list,
                response_mask_list,
            ):
                actor_distribution["response_log_probs"] = extract_response_only_tensor(
                    actor_distribution["response_log_probs"],
                    student_action_mask,
                    response_mask,
                )
                teacher_distribution["response_log_probs"] = extract_response_only_tensor(
                    teacher_distribution["response_log_probs"],
                    teacher_action_mask,
                    response_mask,
                )
                actor_distribution["response_log_probs"] = _normalize_response_distribution_tensor(
                    actor_distribution["response_log_probs"]
                )
                teacher_distribution["response_log_probs"] = _normalize_response_distribution_tensor(
                    teacher_distribution["response_log_probs"]
                )
        else:
            teacher_distribution_ref = reference_model_group.async_run_method_batch(
                method_name="forward_with_topk_log_probs",
                sequences=teacher_reference_sequences_list,
                action_mask=teacher_reference_action_mask_list,
                attention_mask=teacher_reference_attention_mask_list,
                topk=[args.sample_kl_topk] * len(samples_list),
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(teacher_distribution_ref)
                ray.get(reference_model_group.async_run_method(method_name="empty_cache"))

            teacher_distribution_list = _flatten_batched_refs(teacher_distribution_ref, duplicate_factor)
            teacher_token_ids_list = []
            for teacher_distribution, student_action_mask, teacher_action_mask, response_mask in zip(
                teacher_distribution_list,
                action_mask_list,
                teacher_reference_action_mask_list,
                response_mask_list,
            ):
                teacher_distribution["response_topk_token_ids"] = extract_response_only_tensor(
                    teacher_distribution["response_topk_token_ids"],
                    teacher_action_mask,
                    response_mask,
                )
                teacher_distribution["response_topk_log_probs"] = extract_response_only_tensor(
                    teacher_distribution["response_topk_log_probs"],
                    teacher_action_mask,
                    response_mask,
                )
                teacher_distribution["response_topk_log_probs"] = _normalize_response_distribution_tensor(
                    teacher_distribution["response_topk_log_probs"]
                )
                teacher_token_ids_list.append(
                    scatter_response_tensor_to_full_action_mask(
                        teacher_distribution["response_topk_token_ids"],
                        student_action_mask,
                        response_mask,
                    )
                )

            actor_distribution_ref = self.actor_model_group.async_run_method_batch(
                method_name="forward_with_log_probs_at_token_ids",
                sequences=sequences_list,
                action_mask=action_mask_list,
                token_ids=teacher_token_ids_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(actor_distribution_ref)
                ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

            actor_distribution_list = _flatten_batched_refs(actor_distribution_ref, duplicate_factor)
            action_log_probs_list = [result["action_log_probs"] for result in actor_distribution_list]
            for actor_distribution, student_action_mask, response_mask in zip(
                actor_distribution_list,
                action_mask_list,
                response_mask_list,
            ):
                actor_distribution["response_log_probs"] = extract_response_only_tensor(
                    actor_distribution["response_log_probs"],
                    student_action_mask,
                    response_mask,
                )
                actor_distribution["response_log_probs"] = _normalize_response_distribution_tensor(
                    actor_distribution["response_log_probs"]
                )

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and r_refs is not None:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_list = [None] * len(samples_list)

        if self.critic_model_group is not None:
            value_list = _flatten_batched_refs(value_ref, duplicate_factor)

        # Process rewards based on source
        if use_reward_model:
            # Reward Model
            rewards_list = _flatten_batched_refs(r_refs, duplicate_factor)
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            logprobs_diff_mean = None
            kl_reward_mean = None
            teacher_entropy = None

            if sample_kl_mode == "approx":
                if base_action_log_probs is not None:
                    kl = compute_approx_kl(
                        action_log_probs,
                        base_action_log_probs,
                        kl_estimator=self.strategy.args.kl_estimator,
                    )
                    logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
                    logprobs_diff_mean = masked_mean(logprobs_diff, samples.action_mask, dim=-1)
                    kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
                else:
                    kl = torch.zeros_like(action_log_probs, dtype=torch.float32, device=device)
                    kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            elif sample_kl_mode == "full":
                response_mask = response_mask_list[i]
                if enable_adv_kl_reweight:
                    kl_response_s2t = compute_directional_distribution_kl(
                        actor_distribution_list[i]["response_log_probs"],
                        teacher_distribution_list[i]["response_log_probs"],
                        direction="student_to_teacher",
                    )
                    kl_response_t2s = compute_directional_distribution_kl(
                        actor_distribution_list[i]["response_log_probs"],
                        teacher_distribution_list[i]["response_log_probs"],
                        direction="teacher_to_student",
                    )
                    kl_response_s2t = _normalize_kl_response_tensor(kl_response_s2t)
                    kl_response_t2s = _normalize_kl_response_tensor(kl_response_t2s)
                    samples._adv_kl_s2t = scatter_response_tensor_to_full_action_mask(
                        kl_response_s2t,
                        samples.action_mask,
                        response_mask,
                    )
                    samples._adv_kl_t2s = scatter_response_tensor_to_full_action_mask(
                        kl_response_t2s,
                        samples.action_mask,
                        response_mask,
                    )
                kl_response = compute_distribution_kl(
                    actor_distribution_list[i]["response_log_probs"],
                    teacher_distribution_list[i]["response_log_probs"],
                    alpha=args.sample_kl_alpha,
                )
                kl_response = _normalize_kl_response_tensor(kl_response)
                kl = scatter_response_tensor_to_full_action_mask(kl_response, samples.action_mask, response_mask)
                kl_mean = masked_mean(kl_response, response_mask, dim=-1)
                if need_teacher_entropy:
                    teacher_entropy_response = compute_entropy_from_log_probs(
                        teacher_distribution_list[i]["response_log_probs"]
                    )
                    teacher_entropy = scatter_response_tensor_to_full_action_mask(
                        teacher_entropy_response,
                        samples.action_mask,
                        response_mask,
                    )
            elif sample_kl_mode == "topk":
                response_mask = response_mask_list[i]
                teacher_distill_log_probs = None
                if enable_adv_kl_reweight:
                    kl_response_s2t = compute_directional_topk_distribution_kl(
                        actor_distribution_list[i]["response_log_probs"],
                        teacher_distribution_list[i]["response_topk_log_probs"],
                        direction="student_to_teacher",
                        add_tail=args.sample_kl_topk_add_tail,
                    )
                    kl_response_t2s = compute_directional_topk_distribution_kl(
                        actor_distribution_list[i]["response_log_probs"],
                        teacher_distribution_list[i]["response_topk_log_probs"],
                        direction="teacher_to_student",
                        add_tail=args.sample_kl_topk_add_tail,
                    )
                    kl_response_s2t = _normalize_kl_response_tensor(kl_response_s2t)
                    kl_response_t2s = _normalize_kl_response_tensor(kl_response_t2s)
                    samples._adv_kl_s2t = scatter_response_tensor_to_full_action_mask(
                        kl_response_s2t,
                        samples.action_mask,
                        response_mask,
                    )
                    samples._adv_kl_t2s = scatter_response_tensor_to_full_action_mask(
                        kl_response_t2s,
                        samples.action_mask,
                        response_mask,
                    )
                if need_teacher_entropy:
                    _, teacher_distill_log_probs = prepare_topk_log_probs_for_divergence(
                        actor_distribution_list[i]["response_log_probs"],
                        teacher_distribution_list[i]["response_topk_log_probs"],
                        add_tail=args.sample_kl_topk_add_tail,
                    )
                kl_response = compute_topk_distribution_kl(
                    actor_distribution_list[i]["response_log_probs"],
                    teacher_distribution_list[i]["response_topk_log_probs"],
                    alpha=args.sample_kl_alpha,
                    add_tail=args.sample_kl_topk_add_tail,
                )
                kl_response = _normalize_kl_response_tensor(kl_response)
                kl = scatter_response_tensor_to_full_action_mask(kl_response, samples.action_mask, response_mask)
                kl_mean = masked_mean(kl_response, response_mask, dim=-1)
                if need_teacher_entropy:
                    teacher_entropy_response = compute_entropy_from_log_probs(teacher_distill_log_probs)
                    teacher_entropy = scatter_response_tensor_to_full_action_mask(
                        teacher_entropy_response,
                        samples.action_mask,
                        response_mask,
                    )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=torch.float32, device=device)
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if need_teacher_entropy and teacher_entropy is None:
                raise ValueError("teacher entropy requires sample_kl_mode to provide teacher distributions")

            if args.use_kl_reward:
                kl_reward_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.teacher_entropy = teacher_entropy
            samples.info["kl"] = kl_mean
            samples.info["sample_kl_mode"] = [sample_kl_mode] * len(samples.sequences)
            if sample_kl_mode != "approx":
                samples.info["distill_kl"] = kl_mean
                if not enable_adv_kl_reweight:
                    samples.info["distill_alpha"] = torch.full_like(kl_mean, args.sample_kl_alpha, dtype=torch.float32)
            if logprobs_diff_mean is not None:
                samples.info["logprobs_diff"] = logprobs_diff_mean
            # use_kl_reward
            if args.use_kl_reward:
                samples.info["org_reward"] = samples.rewards
                # update rewards
                samples.all_reward = deepcopy(samples.rewards)
                samples.all_reward -= kl_reward_mean * self.kl_ctl.value
                # samples.all_reward = shaped_reward
                samples.rewards = samples.all_reward
                samples.info["reward"] = samples.all_reward
                samples.info["kl_reward"] = kl_reward_mean


        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # Apply length penalties (DAPO overlong / ProRL stop properly) - BEFORE dynamic indices processing
        apply_length_penalties(experiences, args)

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearrange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        if getattr(args, "enable_adv_kl_reweight", False):
            for exp in experiences:
                kl_s2t = getattr(exp, "_adv_kl_s2t", None)
                kl_t2s = getattr(exp, "_adv_kl_t2s", None)
                if kl_s2t is None or kl_t2s is None:
                    raise ValueError(
                        "Directional KL tensors must be precomputed for advantage-conditioned KL reweighting"
                    )

                reweighted_advantages, selected_kl, adv_kl_direction, adv_kl_weight_mean, seq_advantages = (
                    apply_advantage_conditioned_kl_reweight(
                        exp.advantages,
                        exp.action_mask,
                        kl_s2t,
                        kl_t2s,
                        args.adv_kl_weight_clip,
                    )
                )
                exp.advantages = reweighted_advantages
                target_kl_dtype = exp.kl.dtype if exp.kl is not None else selected_kl.dtype
                exp.kl = selected_kl.to(dtype=target_kl_dtype)
                exp.info["kl"] = masked_mean(exp.kl, exp.action_mask, dim=-1)
                exp.info["distill_kl"] = exp.info["kl"]
                exp.info["adv_kl_direction"] = adv_kl_direction
                exp.info["adv_kl_weight_mean"] = adv_kl_weight_mean
                exp.info["adv_kl_seq_adv"] = seq_advantages
                _clear_advantage_conditioned_kl_tensors(exp)

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
