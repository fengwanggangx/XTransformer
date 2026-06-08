import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from tokenizer import tokenizer_load


class TDataSet(IterableDataset):
    """Streaming dataset for pretrain, single-turn SFT, and multi-turn SFT."""

    def __init__(self, cfg, path, mode=None, max_seq_len=None, ignore_index=None):
        self.cfg = cfg
        self.path = Path(path)
        self.mode = cfg.train.mode if mode is None else mode
        self.tokenizer = None
        self.max_seq_len = cfg.model.max_seq_len if max_seq_len is None else max_seq_len
        self.ignore_index = cfg.train.ignore_index if ignore_index is None else ignore_index

        self.PAD = cfg.tokens.padding_idx
        self.BOS = cfg.tokens.bos_idx
        self.EOS = cfg.tokens.eos_idx
        self.SYSTEM = cfg.tokens.system_idx
        self.USER = cfg.tokens.user_idx
        self.ASSISTANT = cfg.tokens.assistant_idx
        self.SEP = cfg.tokens.sep_idx

    def __iter__(self):
        self.ensure_tokenizer()
        yield from self.load_data()

    def collate_fn(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        token_mask = []

        for item in batch:
            ids = item["input_ids"]
            label = item["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.PAD] * pad_len)
            labels.append(label + [self.ignore_index] * pad_len)
            token_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_mask": torch.tensor(token_mask, dtype=torch.long),
        }

    def load_data(self):
        for record in self.get_a_line():
            if self.mode == "pretrain":
                samples = self.build_pretrain_samples(record)
                for sample in samples:
                    yield sample
            elif self.mode == "sft_single":
                sample = self.build_single_turn_samples(record)
                if sample is not None:
                    yield sample
            elif self.mode == "sft_multi":
                sample = self.build_multi_turn_samples(record)
                if sample is not None:
                    yield sample
            else:
                raise ValueError(f"unknown dataset mode: {self.mode}")

    def get_a_line(self):
        files = self.get_path_files(self.path)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        record_index = 0

        for f in files:
            suffix = f.suffix.lower()
            if suffix == ".jsonl":
                records = self.read_jsonl(f)
            elif suffix == ".txt":
                records = self.read_txt(f)
            else:
                continue

            for record in records:
                if record_index % num_workers == worker_id:
                    yield record
                record_index += 1

    def get_path_files(self, path):
        if path.is_file():
            return [path]
        if not path.exists():
            raise FileNotFoundError(f"dataset path does not exist: {path}")

        files = []
        for pattern in ("*.jsonl", "*.txt"):
            files.extend(path.rglob(pattern))
        return sorted(files)

    def read_jsonl(self, path):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid jsonl at {path}:{line_no}") from exc

    def read_txt(self, path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield {"text": text}

    def build_pretrain_samples(self, record):
        text = self.record_text(record)
        if not text:
            return []

        ids = [self.BOS] + self.encode(text) + [self.EOS]
        loss_mask = [True] * len(ids)
        block_size = self.max_seq_len + 1

        samples = []
        for start in range(0, len(ids), block_size):
            chunk_ids = ids[start:start + block_size]
            chunk_mask = loss_mask[start:start + block_size]
            sample = self.finalize(chunk_ids, chunk_mask)
            if sample is not None:
                samples.append(sample)
        return samples

    def build_single_turn_samples(self, record):
        system = self.first_text(record, "system")
        instruction = self.first_text(record, "instruction", "question", "prompt", "user")
        extra_input = self.first_text(record, "input")
        response = self.first_text(record, "response", "output", "answer", "assistant")

        if extra_input:
            instruction = f"{instruction}\n{extra_input}" if instruction else extra_input
        if not instruction or not response:
            return None

        ids = [self.BOS]
        loss_mask = [False]

        if system:
            self.append_message(ids, loss_mask, self.SYSTEM, system, train=False)
        self.append_message(ids, loss_mask, self.USER, instruction, train=False)
        self.append_message(ids, loss_mask, self.ASSISTANT, response, train=True)
        ids.append(self.EOS)
        loss_mask.append(True)

        return self.finalize_sft(ids, loss_mask)

    def build_multi_turn_samples(self, record):
        messages = self.record_messages(record)
        if not messages:
            return None

        ids = [self.BOS]
        loss_mask = [False]

        system = self.first_text(record, "system")
        if system:
            self.append_message(ids, loss_mask, self.SYSTEM, system, train=False)

        has_assistant = False
        for message in messages:
            role = self.normalize_role(message.get("role") or message.get("from"))
            content = self.message_content(message)
            if not role or not content:
                continue

            if role == "system":
                role_id = self.SYSTEM
                train = False
            elif role == "user":
                role_id = self.USER
                train = False
            elif role == "assistant":
                role_id = self.ASSISTANT
                train = True
                has_assistant = True
            else:
                continue

            self.append_message(ids, loss_mask, role_id, content, train=train)

        if not has_assistant:
            return None

        ids.append(self.EOS)
        loss_mask.append(True)
        return self.finalize_sft(ids, loss_mask)

    def append_message(self, ids, loss_mask, role_id, content, train):
        ids.append(role_id)
        loss_mask.append(False)

        content_ids = self.encode(content)
        ids.extend(content_ids)
        loss_mask.extend([train] * len(content_ids))

        ids.append(self.SEP)
        loss_mask.append(train)

    def finalize_sft(self, ids, loss_mask):
        ids, loss_mask = self.truncate_sft(ids, loss_mask)
        return self.finalize(ids, loss_mask)

    def truncate_sft(self, ids, loss_mask):
        limit = self.max_seq_len + 1
        if len(ids) <= limit:
            return ids, loss_mask

        assistant_pos = self.last_trainable_assistant_pos(ids, loss_mask)
        if assistant_pos is None:
            return ids[:limit], loss_mask[:limit]

        user_pos = self.previous_role_pos(ids, assistant_pos, self.USER)
        if user_pos is None:
            return self.trim_from_role(ids, loss_mask, assistant_pos, limit)

        space = limit - 1
        if len(ids) - user_pos <= space:
            start = max(1, len(ids) - space)
            start = self.nearest_role_start(ids, start, user_pos)
            return self.slice_with_bos(ids, loss_mask, start, limit)

        return self.truncate_last_turn(ids, loss_mask, user_pos, assistant_pos, limit)

    def trim_from_role(self, ids, loss_mask, role_pos, limit):
        space = limit - 1
        if len(ids) - role_pos >= space:
            start = role_pos
        else:
            start = max(1, len(ids) - space)
            start = self.nearest_role_start(ids, start, role_pos)
        return self.slice_with_bos(ids, loss_mask, start, limit)

    def truncate_last_turn(self, ids, loss_mask, user_pos, assistant_pos, limit):
        space = limit - 1
        if space < 2:
            return ids[:limit], loss_mask[:limit]

        user_ids = ids[user_pos:assistant_pos]
        user_mask = loss_mask[user_pos:assistant_pos]
        assistant_ids = ids[assistant_pos:]
        assistant_mask = loss_mask[assistant_pos:]

        min_user_budget = min(len(user_ids), max(1, space // 4))
        if len(user_ids) > 1 and space >= 3:
            min_user_budget = max(2, min_user_budget)

        assistant_budget = min(len(assistant_ids), space - min_user_budget)
        user_budget = space - assistant_budget

        kept_user_ids, kept_user_mask = self.keep_user_context(
            user_ids,
            user_mask,
            user_budget,
        )
        kept_assistant_ids = assistant_ids[:assistant_budget]
        kept_assistant_mask = assistant_mask[:assistant_budget]

        return (
            [self.BOS] + kept_user_ids + kept_assistant_ids,
            [False] + kept_user_mask + kept_assistant_mask,
        )

    def keep_user_context(self, user_ids, user_mask, budget):
        if budget <= 0:
            return [], []
        if len(user_ids) <= budget:
            return user_ids, user_mask
        if budget == 1:
            return [user_ids[0]], [user_mask[0]]

        body_ids = user_ids[1:]
        body_mask = user_mask[1:]
        if len(body_ids) > 1 and body_ids[-1] == self.SEP:
            body_ids = body_ids[:-1]
            body_mask = body_mask[:-1]

        return (
            [user_ids[0]] + body_ids[-(budget - 1):],
            [user_mask[0]] + body_mask[-(budget - 1):],
        )

    def slice_with_bos(self, ids, loss_mask, start, limit):
        trimmed_ids = [self.BOS] + ids[start:start + limit - 1]
        trimmed_mask = [False] + loss_mask[start:start + limit - 1]
        return trimmed_ids, trimmed_mask

    def nearest_role_start(self, ids, start, assistant_pos):
        role_ids = {self.SYSTEM, self.USER, self.ASSISTANT}
        for pos in range(start, assistant_pos + 1):
            if ids[pos] in role_ids:
                return pos
        return assistant_pos

    def previous_role_pos(self, ids, before_pos, role_id):
        for pos in range(before_pos - 1, 0, -1):
            if ids[pos] == role_id:
                return pos
        return None

    def last_trainable_assistant_pos(self, ids, loss_mask):
        for pos in range(len(ids) - 1, -1, -1):
            if ids[pos] != self.ASSISTANT:
                continue
            if any(loss_mask[pos + 1:]):
                return pos
        return None

    def finalize(self, ids, loss_mask):
        if len(ids) > self.max_seq_len + 1:
            ids = ids[:self.max_seq_len + 1]
            loss_mask = loss_mask[:self.max_seq_len + 1]

        if len(ids) < 2:
            return None

        input_ids = ids[:-1]
        labels = ids[1:]
        target_mask = loss_mask[1:]
        labels = [
            token_id if keep else self.ignore_index
            for token_id, keep in zip(labels, target_mask)
        ]

        if all(label == self.ignore_index for label in labels):
            return None

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def encode(self, text):
        self.ensure_tokenizer()
        return list(self.tokenizer.encode(str(text), out_type=int))

    def ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = tokenizer_load(self.cfg)

    def record_text(self, record):
        if isinstance(record, str):
            return record.strip()
        return self.first_text(record, "text", "content")

    def first_text(self, record, *keys):
        if not isinstance(record, dict):
            return ""
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def record_messages(self, record):
        if not isinstance(record, dict):
            return []
        messages = record.get("messages") or record.get("conversations")
        return messages if isinstance(messages, list) else []

    def message_content(self, message):
        if not isinstance(message, dict):
            return ""
        value = message.get("content")
        if value is None:
            value = message.get("value")
        return value.strip() if isinstance(value, str) else ""

    def normalize_role(self, role):
        if not isinstance(role, str):
            return ""
        role = role.lower().strip()
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "bot": "assistant",
            "system": "system",
        }
        return role_map.get(role, "")
