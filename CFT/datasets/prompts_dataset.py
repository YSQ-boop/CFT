from torch.utils.data import Dataset
from tqdm import tqdm

from CFT.utils.sample_kl_mode_utils import sample_kl_uses_teacher_prompt
from CFT.utils.teacher_prompt_utils import build_teacher_user_content


def preprocess_data(data, input_template=None, sys_key=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "system", "content": data[sys_key]}, {"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # prompt = data[sys_key]+data[input_key]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


def preprocess_teacher_prompt(data, sys_key, input_key, label_key, apply_chat_template) -> str:
    chat = data[input_key]
    if not isinstance(chat, str):
        raise ValueError("sample_kl_mode=full/topk only supports string chat inputs")

    teacher_chat = [
        {"role": "system", "content": data[sys_key]},
        {
            "role": "user",
            "content": build_teacher_user_content(chat, data[label_key]),
        },
    ]
    return apply_chat_template(teacher_chat, tokenize=False, add_generation_prompt=True)


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        sys_key = getattr(self.strategy.args, "sys_key", None)
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        sample_kl_mode = getattr(self.strategy.args, "sample_kl_mode", "approx")
        self.teacher_prompt_enabled = sample_kl_uses_teacher_prompt(sample_kl_mode)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        self.teacher_prompts = [] if self.teacher_prompt_enabled else None
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, sys_key, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))
            if self.teacher_prompt_enabled:
                teacher_prompt = preprocess_teacher_prompt(data, sys_key, input_key, label_key, apply_chat_template)
                self.teacher_prompts.append(teacher_prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if self.teacher_prompts is not None:
            return self.datasources[idx], self.prompts[idx], self.labels[idx], self.teacher_prompts[idx]
        return self.datasources[idx], self.prompts[idx], self.labels[idx]

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        teacher_prompts = [] if self.teacher_prompts is not None else None
        for item in item_list:
            datasource, prompt, label = item[:3]
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            if teacher_prompts is not None:
                teacher_prompts.append(item[3])

        if teacher_prompts is not None:
            return datasources, prompts, labels, teacher_prompts
        return datasources, prompts, labels
