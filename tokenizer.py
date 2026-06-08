from pathlib import Path

import sentencepiece as spm


def validate_special_tokens(tokenizer, cfg):
    actual_vocab_size = tokenizer.GetPieceSize()
    if actual_vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"tokenizer vocab size mismatch: expected {cfg.model.vocab_size}, got {actual_vocab_size}"
        )

    for token, expected_id in cfg.tokens.all_token_ids.items():
        actual_id = tokenizer.piece_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"token {token} id mismatch: expected {expected_id}, got {actual_id}"
            )


def content_token_ids_to_mask(cfg):
    return frozenset(
        token_id
        for token, token_id in cfg.tokens.all_token_ids.items()
        if token != cfg.tokens.unk_token
    )


def encode_content(cfg, tokenizer, text):
    blocked_token_ids = content_token_ids_to_mask(cfg)
    return [
        cfg.tokens.unk_idx if token_id in blocked_token_ids else token_id
        for token_id in tokenizer.encode(str(text), out_type=int)
    ]


def tokenizer_load(cfg, model_path=None):
    model_path = cfg.paths.tokenizer_model_path if model_path is None else model_path
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(model_path)
    validate_special_tokens(tokenizer, cfg)
    return tokenizer


def entry(cfg):
    Path(cfg.paths.tokenizer_output_dir).mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=cfg.paths.tokenizer_corpus_path,
        model_prefix=cfg.paths.tokenizer_model_prefix,
        vocab_size=cfg.model.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=cfg.tokens.padding_idx,
        unk_id=cfg.tokens.unk_idx,
        bos_id=cfg.tokens.bos_idx,
        eos_id=cfg.tokens.eos_idx,
        pad_piece=cfg.tokens.pad_token,
        unk_piece=cfg.tokens.unk_token,
        bos_piece=cfg.tokens.bos_token,
        eos_piece=cfg.tokens.eos_token,
        control_symbols=list(cfg.tokens.special_tokens),
    )
    tokenizer_load(cfg, cfg.paths.tokenizer_model_path)


if __name__ == "__main__":
    from config import load_config

    entry(load_config())
