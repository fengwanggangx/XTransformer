import random

import torch


def normalize_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {seed!r}") from exc
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return seed


def seed_everything(seed):
    seed = normalize_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def build_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(normalize_seed(seed))
    return generator
