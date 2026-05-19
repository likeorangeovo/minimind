import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _normalize_messages(sample: dict):
    if "conversations" in sample:
        messages = sample["conversations"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("conversations must be a non-empty list")
        normalized = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("each conversation item must be a dict")
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("conversations contain no valid content")
        return normalized

    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("each message must be a dict")
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("messages contain no valid content")
        return normalized

    if "instruction" in sample and "output" in sample:
        instruction = str(sample["instruction"]).strip()
        output = str(sample["output"]).strip()
        system = str(sample.get("system", "")).strip()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": output})
        return messages

    if "prompt" in sample and "response" in sample:
        prompt = str(sample["prompt"]).strip()
        response = str(sample["response"]).strip()
        system = str(sample.get("system", "")).strip()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        return messages

    if "question" in sample and "answer" in sample:
        question = str(sample["question"]).strip()
        answer = str(sample["answer"]).strip()
        messages = [{"role": "user", "content": question}]
        messages.append({"role": "assistant", "content": answer})
        return messages

    raise KeyError(
        "Unsupported SFT sample format. Expected one of: conversations, "
        "messages, instruction/output, prompt/response, question/answer."
    )


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.samples = []

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"SFT dataset not found: {data_path.resolve()}")

        with data_path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    sample = json.loads(raw)
                    messages = _normalize_messages(sample)
                    encoded = self._encode_messages(messages)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse SFT sample at line {line_no}: {exc}"
                    ) from exc
                if encoded is not None:
                    self.samples.append(encoded)

        if not self.samples:
            raise ValueError(f"No usable SFT samples found in {data_path.resolve()}")

        print(f"[sft_dataset] loaded {len(self.samples)} samples from {data_path.name}")

    def _encode_messages(self, messages):
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        if len(full_ids) < 2:
            return None

        labels = [-100] * len(full_ids)
        for idx, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            prefix_text = self.tokenizer.apply_chat_template(
                messages[:idx], tokenize=False, add_generation_prompt=True
            )
            target_text = self.tokenizer.apply_chat_template(
                messages[: idx + 1], tokenize=False, add_generation_prompt=False
            )

            prefix_ids = self.tokenizer(
                prefix_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            start = min(len(prefix_ids), len(full_ids))
            end = min(len(target_ids), len(full_ids))
            for pos in range(start, end):
                labels[pos] = full_ids[pos]

        if all(value == -100 for value in labels):
            return None

        attention_mask = [1] * len(full_ids)
        pad_len = self.max_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
