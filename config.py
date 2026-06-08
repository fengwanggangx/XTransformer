from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml


DEFAULT_CONFIG_PATH = "configs/default.yaml"
TRAIN_MODES = ("pretrain", "sft_single", "sft_multi")
TRAIN_LANGUAGES = ("all", "chinese", "english")


@dataclass
class RuntimeConfig:
    seed: int
    debug: bool
    device: str


@dataclass
class ModelConfig:
    d_model: int
    vocab_size: int
    max_seq_len: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float


@dataclass
class TrainConfig:
    mode: str
    language: str
    data_path: str
    batch_size: int
    num_workers: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    grad_accum_steps: int
    warmup_steps: int
    min_lr_ratio: float
    max_grad_norm: float
    ignore_index: int
    log_steps: int
    save_steps: int
    eval_data_path: str
    eval_steps: int
    eval_batches: int


@dataclass
class InferenceConfig:
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    stop_at_sep: bool
    stop_token_ids: tuple[int, ...]


@dataclass
class TokenConfig:
    pad_token: str
    unk_token: str
    bos_token: str
    eos_token: str
    system_token: str
    user_token: str
    assistant_token: str
    sep_token: str
    padding_idx: int
    unk_idx: int
    bos_idx: int
    eos_idx: int
    system_idx: int
    user_idx: int
    assistant_idx: int
    sep_idx: int

    @property
    def special_tokens(self):
        return (
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.sep_token,
        )

    @property
    def special_token_ids(self):
        return {
            self.system_token: self.system_idx,
            self.user_token: self.user_idx,
            self.assistant_token: self.assistant_idx,
            self.sep_token: self.sep_idx,
        }

    @property
    def all_token_ids(self):
        return {
            self.pad_token: self.padding_idx,
            self.unk_token: self.unk_idx,
            self.bos_token: self.bos_idx,
            self.eos_token: self.eos_idx,
            **self.special_token_ids,
        }


@dataclass
class PathConfig:
    data_dir: str
    model_dir: str
    tokenizer_corpus_dir: str
    tokenizer_output_dir: str
    tokenizer_model_prefix: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str
    tokenizer_corpus_path: str
    pretrain_dir: str
    pretrain_chinese_dir: str
    pretrain_english_dir: str
    sft_dir: str
    sft_single_turn_dir: str
    sft_single_turn_chinese_dir: str
    sft_single_turn_english_dir: str
    sft_multi_turn_dir: str
    sft_multi_turn_chinese_dir: str
    sft_multi_turn_english_dir: str
    eval_dir: str
    checkpoint_dir: str
    pretrain_checkpoint_path: str
    sft_single_checkpoint_path: str
    sft_multi_checkpoint_path: str


@dataclass
class AppConfig:
    project_dir: Path
    config_path: Path
    runtime: RuntimeConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig
    tokens: TokenConfig
    paths: PathConfig
    raw: dict

    @property
    def device(self):
        if self.runtime.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.runtime.device)

    @property
    def train_modes(self):
        return TRAIN_MODES

    @property
    def train_languages(self):
        return TRAIN_LANGUAGES

    @property
    def train_data_dirs(self):
        return {
            "pretrain": {
                "all": self.paths.pretrain_dir,
                "chinese": self.paths.pretrain_chinese_dir,
                "english": self.paths.pretrain_english_dir,
            },
            "sft_single": {
                "all": self.paths.sft_single_turn_dir,
                "chinese": self.paths.sft_single_turn_chinese_dir,
                "english": self.paths.sft_single_turn_english_dir,
            },
            "sft_multi": {
                "all": self.paths.sft_multi_turn_dir,
                "chinese": self.paths.sft_multi_turn_chinese_dir,
                "english": self.paths.sft_multi_turn_english_dir,
            },
        }

    @property
    def checkpoint_paths(self):
        return {
            "pretrain": self.paths.pretrain_checkpoint_path,
            "sft_single": self.paths.sft_single_checkpoint_path,
            "sft_multi": self.paths.sft_multi_checkpoint_path,
        }

    def get_train_data_path(self, mode=None, language=None, data_path=None):
        if data_path:
            return data_path
        if self.train.data_path:
            return self.train.data_path

        mode = mode or self.train.mode
        language = language or self.train.language
        if mode not in self.train_data_dirs:
            raise ValueError(f"unsupported train mode: {mode}")
        if language not in self.train_data_dirs[mode]:
            raise ValueError(f"unsupported train language: {language}")
        return self.train_data_dirs[mode][language]

    def get_checkpoint_path(self, mode=None, checkpoint=None):
        if checkpoint:
            return checkpoint

        mode = mode or self.train.mode
        if mode not in self.checkpoint_paths:
            raise ValueError(f"unsupported train mode: {mode}")
        return self.checkpoint_paths[mode]

    def to_dict(self):
        return {
            "runtime": asdict(self.runtime),
            "model": asdict(self.model),
            "train": asdict(self.train),
            "inference": {
                **asdict(self.inference),
                "stop_token_ids": list(self.inference.stop_token_ids),
            },
            "tokens": asdict(self.tokens),
            "paths": asdict(self.paths),
        }


def resolve_path(project_dir, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(project_dir / path)


def section(raw, name):
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section `{name}` must be a mapping")
    return value


def build_paths(project_dir, raw_paths):
    def path_value(name, default):
        return resolve_path(project_dir, raw_paths.get(name, default))

    data_dir = path_value("data_dir", "data")
    model_dir = path_value("model_dir", "model")
    tokenizer_corpus_dir = path_value("tokenizer_corpus_dir", f"{data_dir}/tokenizer")
    tokenizer_output_dir = path_value("tokenizer_output_dir", f"{model_dir}/tokenizer")
    tokenizer_model_prefix = path_value(
        "tokenizer_model_prefix",
        f"{tokenizer_output_dir}/transformer",
    )
    checkpoint_dir = path_value("checkpoint_dir", f"{model_dir}/checkpoints")
    pretrain_dir = path_value("pretrain_dir", f"{data_dir}/pretrain")
    sft_dir = path_value("sft_dir", f"{data_dir}/sft")

    return PathConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        tokenizer_corpus_dir=tokenizer_corpus_dir,
        tokenizer_output_dir=tokenizer_output_dir,
        tokenizer_model_prefix=tokenizer_model_prefix,
        tokenizer_model_path=path_value("tokenizer_model_path", f"{tokenizer_model_prefix}.model"),
        tokenizer_vocab_path=path_value("tokenizer_vocab_path", f"{tokenizer_model_prefix}.vocab"),
        tokenizer_corpus_path=path_value("tokenizer_corpus_path", f"{tokenizer_corpus_dir}/corpus.txt"),
        pretrain_dir=pretrain_dir,
        pretrain_chinese_dir=path_value("pretrain_chinese_dir", f"{pretrain_dir}/chinese"),
        pretrain_english_dir=path_value("pretrain_english_dir", f"{pretrain_dir}/english"),
        sft_dir=sft_dir,
        sft_single_turn_dir=path_value("sft_single_turn_dir", f"{sft_dir}/single_turn"),
        sft_single_turn_chinese_dir=path_value(
            "sft_single_turn_chinese_dir",
            f"{sft_dir}/single_turn/chinese",
        ),
        sft_single_turn_english_dir=path_value(
            "sft_single_turn_english_dir",
            f"{sft_dir}/single_turn/english",
        ),
        sft_multi_turn_dir=path_value("sft_multi_turn_dir", f"{sft_dir}/multi_turn"),
        sft_multi_turn_chinese_dir=path_value(
            "sft_multi_turn_chinese_dir",
            f"{sft_dir}/multi_turn/chinese",
        ),
        sft_multi_turn_english_dir=path_value(
            "sft_multi_turn_english_dir",
            f"{sft_dir}/multi_turn/english",
        ),
        eval_dir=path_value("eval_dir", f"{data_dir}/eval"),
        checkpoint_dir=checkpoint_dir,
        pretrain_checkpoint_path=path_value("pretrain_checkpoint_path", f"{checkpoint_dir}/pretrain.pt"),
        sft_single_checkpoint_path=path_value(
            "sft_single_checkpoint_path",
            f"{checkpoint_dir}/sft_single.pt",
        ),
        sft_multi_checkpoint_path=path_value(
            "sft_multi_checkpoint_path",
            f"{checkpoint_dir}/sft_multi.pt",
        ),
    )


def build_tokens(raw_tokens):
    return TokenConfig(
        pad_token=raw_tokens["pad_token"],
        unk_token=raw_tokens["unk_token"],
        bos_token=raw_tokens["bos_token"],
        eos_token=raw_tokens["eos_token"],
        system_token=raw_tokens["system_token"],
        user_token=raw_tokens["user_token"],
        assistant_token=raw_tokens["assistant_token"],
        sep_token=raw_tokens["sep_token"],
        padding_idx=int(raw_tokens["padding_idx"]),
        unk_idx=int(raw_tokens["unk_idx"]),
        bos_idx=int(raw_tokens["bos_idx"]),
        eos_idx=int(raw_tokens["eos_idx"]),
        system_idx=int(raw_tokens["system_idx"]),
        user_idx=int(raw_tokens["user_idx"]),
        assistant_idx=int(raw_tokens["assistant_idx"]),
        sep_idx=int(raw_tokens["sep_idx"]),
    )


def checkpoint_model_config(cfg):
    return {
        "d_model": cfg.model.d_model,
        "vocab_size": cfg.model.vocab_size,
        "max_seq_len": cfg.model.max_seq_len,
        "n_heads": cfg.model.n_heads,
        "n_layers": cfg.model.n_layers,
        "d_ff": cfg.model.d_ff,
        "dropout": cfg.model.dropout,
        "padding_idx": cfg.tokens.padding_idx,
        "ignore_index": cfg.train.ignore_index,
    }


def load_config(path=None):
    project_dir = Path(__file__).resolve().parent
    path = DEFAULT_CONFIG_PATH if path is None else path
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_dir / config_path

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    runtime = section(raw, "runtime")
    model = section(raw, "model")
    train = section(raw, "train")
    inference = section(raw, "inference")
    tokens = build_tokens(section(raw, "tokens"))
    paths = build_paths(project_dir, section(raw, "paths"))

    return AppConfig(
        project_dir=project_dir,
        config_path=config_path,
        runtime=RuntimeConfig(
            seed=int(runtime.get("seed", 42)),
            debug=bool(runtime.get("debug", False)),
            device=str(runtime.get("device", "auto")),
        ),
        model=ModelConfig(
            d_model=int(model["d_model"]),
            vocab_size=int(model["vocab_size"]),
            max_seq_len=int(model["max_seq_len"]),
            n_heads=int(model["n_heads"]),
            n_layers=int(model["n_layers"]),
            d_ff=int(model["d_ff"]),
            dropout=float(model["dropout"]),
        ),
        train=TrainConfig(
            mode=str(train.get("mode", "pretrain")),
            language=str(train.get("language", "all")),
            data_path=str(train.get("data_path", "")),
            batch_size=int(train["batch_size"]),
            num_workers=int(train["num_workers"]),
            max_steps=int(train["max_steps"]),
            learning_rate=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
            grad_accum_steps=int(train["grad_accum_steps"]),
            warmup_steps=int(train["warmup_steps"]),
            min_lr_ratio=float(train["min_lr_ratio"]),
            max_grad_norm=float(train["max_grad_norm"]),
            ignore_index=int(train["ignore_index"]),
            log_steps=int(train["log_steps"]),
            save_steps=int(train["save_steps"]),
            eval_data_path=str(train.get("eval_data_path", "")),
            eval_steps=int(train["eval_steps"]),
            eval_batches=int(train["eval_batches"]),
        ),
        inference=InferenceConfig(
            temperature=float(inference["temperature"]),
            top_k=int(inference["top_k"]),
            top_p=float(inference["top_p"]),
            max_new_tokens=int(inference["max_new_tokens"]),
            stop_at_sep=bool(inference.get("stop_at_sep", True)),
            stop_token_ids=tuple(int(token_id) for token_id in inference["stop_token_ids"]),
        ),
        tokens=tokens,
        paths=paths,
        raw=raw,
    )
