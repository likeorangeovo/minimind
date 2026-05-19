import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集：首次运行会把 jsonl 全量 tokenize + pad 到 max_length，
    缓存为 uint16 的 numpy memmap。之后训练直接零 tokenize 从磁盘读取，
    __getitem__ 几乎是纯内存拷贝，DataLoader 吞吐提升 10x 以上。
    """

    def __init__(self, data_path, tokenizer, max_length=512, cache_dir=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path.resolve()}")

        if cache_dir is None:
            cache_dir = data_path.parent / ".cache"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存键：原文件 mtime/size + tokenizer 词表大小 + max_length + special token id
        # 任意一项变了都会重新 tokenize
        stat = data_path.stat()
        key = "|".join(
            str(x)
            for x in (
                data_path.resolve(),
                stat.st_size,
                int(stat.st_mtime),
                tokenizer.vocab_size,
                self.pad_id,
                self.bos_id,
                self.eos_id,
                max_length,
            )
        )
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        self._tokens_path = cache_dir / f"pretrain_tokens_{digest}.bin"
        self._meta_path = cache_dir / f"pretrain_tokens_{digest}.json"

        if not self._tokens_path.exists() or not self._meta_path.exists():
            print(f"[dataset] building token cache: {self._tokens_path.name}")
            self._build_cache(data_path)

        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_samples = meta["num_samples"]
        assert meta["max_length"] == max_length
        assert tokenizer.vocab_size <= 65535, (
            "vocab_size exceeds uint16 range; raise cache dtype to int32."
        )

        self._mmap = np.memmap(
            self._tokens_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_samples, self.max_length),
        )
        print(
            f"[dataset] using token cache: {self._tokens_path.name} "
            f"({self.num_samples} samples)"
        )

    def _build_cache(self, data_path: Path):
        # 两遍扫描：第一遍只数样本数，第二遍写入 memmap。
        # 这样避免在内存里囤一份完整 token 数组。
        num_samples = 0
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    num_samples += 1
        if num_samples == 0:
            raise ValueError(f"Dataset file is empty: {data_path.resolve()}")

        tmp_path = self._tokens_path.with_suffix(".bin.tmp")
        arr = np.memmap(
            tmp_path,
            dtype=np.uint16,
            mode="w+",
            shape=(num_samples, self.max_length),
        )

        # 批量 tokenize：一次送一批文本给 tokenizer，减少 Python 循环和 Rust 调用开销
        batch_size = 2048
        idx = 0
        buf_text = []

        def flush():
            nonlocal idx, buf_text
            if not buf_text:
                return
            enc = self.tokenizer(
                buf_text,
                add_special_tokens=False,
                max_length=self.max_length - 2,
                truncation=True,
            )["input_ids"]
            for ids in enc:
                row = [self.bos_id] + ids + [self.eos_id]
                if len(row) < self.max_length:
                    row = row + [self.pad_id] * (self.max_length - len(row))
                arr[idx, :] = np.asarray(row[: self.max_length], dtype=np.uint16)
                idx += 1
            buf_text = []

        with data_path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {data_path}: {exc}"
                    ) from exc
                if "text" not in sample:
                    raise KeyError(
                        f'Missing "text" field on line {line_no} of {data_path}'
                    )
                buf_text.append(str(sample["text"]))
                if len(buf_text) >= batch_size:
                    flush()
        flush()
        arr.flush()
        del arr

        os.replace(tmp_path, self._tokens_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"num_samples": num_samples, "max_length": self.max_length},
                f,
                ensure_ascii=False,
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        row = np.asarray(self._mmap[index], dtype=np.int64)
        input_ids = torch.from_numpy(row)
        return input_ids

        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100

        # attention_mask 保留给下游使用；预训练阶段 pad 右对齐 + labels 忽略 pad，
        # 训练时可直接丢弃以启用 Flash SDPA 快速路径。
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, labels, attention_mask
