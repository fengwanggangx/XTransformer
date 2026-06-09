import re
import warnings

from config import checkpoint_model_config


ALLOWED_MISSING_PATTERNS = (
    re.compile(r"decoder\.layers\.\d+\.self_attn\.rope\.inv_freq"),
)

ALLOWED_UNEXPECTED_KEYS = {
    "input_embed.1.pe",
}

CHECKPOINT_MODEL_CONFIG_KEYS = (
    "d_embed",
    "vocab_size",
    "max_seq_len",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "padding_idx",
    "ignore_index",
)


def is_allowed_missing_key(key):
    return any(pattern.fullmatch(key) for pattern in ALLOWED_MISSING_PATTERNS)


def validate_checkpoint_config(checkpoint, cfg, source="checkpoint"):
    saved_config = checkpoint.get("model_config")
    if not saved_config:
        warnings.warn(
            f"{source} has no model_config; skipped config compatibility check",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    missing_fields = [
        key for key in CHECKPOINT_MODEL_CONFIG_KEYS
        if key not in saved_config
    ]
    if missing_fields:
        warnings.warn(
            f"{source} model_config missing fields: {', '.join(missing_fields)}",
            RuntimeWarning,
            stacklevel=2,
        )

    mismatches = []
    current_config = checkpoint_model_config(cfg)
    for key in CHECKPOINT_MODEL_CONFIG_KEYS:
        if key not in saved_config:
            continue
        current_value = current_config[key]
        saved_value = saved_config[key]
        if saved_value != current_value:
            mismatches.append(f"{key}: checkpoint={saved_value}, current={current_value}")

    if mismatches:
        raise RuntimeError(
            "checkpoint model_config mismatch: "
            f"{source}\n  " + "\n  ".join(mismatches)
        )


def load_model_state(model, state_dict, source="checkpoint"):
    result = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    bad_missing = [key for key in missing_keys if not is_allowed_missing_key(key)]
    bad_unexpected = [
        key for key in unexpected_keys
        if key not in ALLOWED_UNEXPECTED_KEYS
    ]

    if bad_missing or bad_unexpected:
        parts = [f"checkpoint structure mismatch: {source}"]
        if bad_missing:
            parts.append("missing keys:\n  " + "\n  ".join(bad_missing))
        if bad_unexpected:
            parts.append("unexpected keys:\n  " + "\n  ".join(bad_unexpected))
        raise RuntimeError("\n".join(parts))

    allowed_missing = [
        key for key in missing_keys
        if is_allowed_missing_key(key)
    ]
    allowed_unexpected = [
        key for key in unexpected_keys
        if key in ALLOWED_UNEXPECTED_KEYS
    ]
    if allowed_missing or allowed_unexpected:
        print(
            f"loaded {source} with allowed compatibility differences: "
            f"missing={allowed_missing}, unexpected={allowed_unexpected}",
            flush=True,
        )

    return result
