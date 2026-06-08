from pathlib import Path
import math

import torch
import torch.nn.functional as F

from checkpoint import load_model_state, validate_checkpoint_config
from config import load_config
from model import make_model, zero_padding_embedding
from reproducibility import seed_everything
from tokenizer import tokenizer_load


_SESSION_CACHE = {}


def normalize_int(name, value, minimum):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def normalize_float(name, value, minimum=None, maximum=None):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def load_model(cfg, checkpoint_path=None):
    checkpoint_path = cfg.get_checkpoint_path(checkpoint=checkpoint_path)
    model = make_model(cfg)
    path = Path(checkpoint_path)
    if path.exists():
        checkpoint = torch.load(path, map_location=cfg.device, weights_only=True)
        validate_checkpoint_config(checkpoint, cfg, source=str(path))
        load_model_state(model, checkpoint["model_state_dict"], source=str(path))
        zero_padding_embedding(model)
    else:
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    model.eval()
    return model


def append_message(cfg, ids, tokenizer, role_id, content):
    ids.append(role_id)
    ids.extend(tokenizer.encode(str(content), out_type=int))
    ids.append(cfg.tokens.sep_idx)


def build_prompt(cfg, tokenizer, prompt, system=""):
    ids = [cfg.tokens.bos_idx]
    if system:
        append_message(cfg, ids, tokenizer, cfg.tokens.system_idx, system)
    append_message(cfg, ids, tokenizer, cfg.tokens.user_idx, prompt)
    ids.append(cfg.tokens.assistant_idx)
    return ids


def filter_logits(cfg, logits, top_k=None, top_p=None):
    top_k = cfg.inference.top_k if top_k is None else top_k
    top_p = cfg.inference.top_p if top_p is None else top_p
    top_k = normalize_int("top_k", top_k, 0)
    top_p = normalize_float("top_p", top_p, minimum=0.0, maximum=1.0)
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < values[..., -1, None], torch.full_like(logits, float("-inf")), logits)

    if top_p and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))

    return logits


def normalize_stop_token_ids(cfg, stop_token_ids=None, stop_at_sep=True):
    stop_token_ids = (
        cfg.inference.stop_token_ids
        if stop_token_ids is None
        else stop_token_ids
    )
    stop_token_ids = {normalize_int("stop_token_id", token_id, 0) for token_id in stop_token_ids}
    if not stop_at_sep:
        stop_token_ids.discard(cfg.tokens.sep_idx)
    return frozenset(stop_token_ids)


class InferenceSession:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        checkpoint_path = cfg.get_checkpoint_path(checkpoint=checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer_load(cfg)
        self.model = load_model(cfg, checkpoint_path)

    def build_cache_from_context(self, input_ids, max_seq_len):
        context = input_ids[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=self.cfg.device)
        logits, past_key_values = self.model(x, use_cache=True)
        return logits[0, -1], past_key_values

    @torch.no_grad()
    def generate(
        self,
        prompt,
        system="",
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        stop_token_ids=None,
        stop_at_sep=None,
    ):
        cfg = self.cfg
        max_new_tokens = cfg.inference.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = cfg.inference.temperature if temperature is None else temperature
        top_k = cfg.inference.top_k if top_k is None else top_k
        top_p = cfg.inference.top_p if top_p is None else top_p
        stop_at_sep = cfg.inference.stop_at_sep if stop_at_sep is None else stop_at_sep
        max_new_tokens = normalize_int("max_new_tokens", max_new_tokens, 1)
        temperature = normalize_float("temperature", temperature, minimum=0.0)
        top_k = normalize_int("top_k", top_k, 0)
        top_p = normalize_float("top_p", top_p, minimum=0.0, maximum=1.0)
        max_seq_len = normalize_int("max_seq_len", cfg.model.max_seq_len, 1)
        stop_token_ids = normalize_stop_token_ids(
            cfg,
            stop_token_ids=stop_token_ids,
            stop_at_sep=stop_at_sep,
        )
        if seed is not None:
            seed_everything(seed)
        input_ids = build_prompt(cfg, self.tokenizer, prompt, system=system)
        generated = []
        next_logits, past_key_values = self.build_cache_from_context(input_ids, max_seq_len)

        for token_index in range(max_new_tokens):
            if temperature and temperature > 0:
                sample_logits = filter_logits(cfg, next_logits / temperature, top_k=top_k, top_p=top_p)
                probs = F.softmax(sample_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(next_logits).item())

            if next_id in stop_token_ids:
                break

            input_ids.append(next_id)
            generated.append(next_id)
            if token_index == max_new_tokens - 1:
                break

            past_len = past_key_values[0][0].size(-2)
            if past_len >= max_seq_len:
                next_logits, past_key_values = self.build_cache_from_context(input_ids, max_seq_len)
            else:
                x = torch.tensor([[next_id]], dtype=torch.long, device=cfg.device)
                logits, past_key_values = self.model(
                    x,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_logits = logits[0, -1]

        return self.tokenizer.decode(generated)

def checkpoint_cache_key(checkpoint_path):
    path = Path(checkpoint_path)
    return str(path), path.stat().st_mtime_ns if path.exists() else None


def get_inference_session(cfg, checkpoint_path=None):
    checkpoint_path = cfg.get_checkpoint_path(checkpoint=checkpoint_path)
    checkpoint_path, checkpoint_mtime = checkpoint_cache_key(checkpoint_path)
    cache_key = (str(cfg.config_path), checkpoint_path, checkpoint_mtime)
    session = _SESSION_CACHE.get(cache_key)
    if session is None:
        if len(_SESSION_CACHE) >= 4:
            _SESSION_CACHE.pop(next(iter(_SESSION_CACHE)))
        session = InferenceSession(cfg, checkpoint_path)
        _SESSION_CACHE[cache_key] = session
    return session


def clear_inference_session_cache():
    _SESSION_CACHE.clear()


def generate(
    cfg,
    prompt,
    system="",
    checkpoint_path=None,
    max_new_tokens=None,
    temperature=None,
    top_k=None,
    top_p=None,
    seed=None,
    stop_token_ids=None,
    stop_at_sep=None,
):
    session = get_inference_session(cfg, checkpoint_path)
    return session.generate(
        prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        stop_token_ids=stop_token_ids,
        stop_at_sep=stop_at_sep,
    )


def entry():
    cfg = load_config()
    session = get_inference_session(cfg, cfg.get_checkpoint_path())
    while True:
        prompt = input("user> ").strip()
        if not prompt:
            return
        print(session.generate(prompt))


if __name__ == "__main__":
    entry()
