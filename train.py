from pathlib import Path
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import checkpoint_model_config, load_config
from checkpoint import load_model_state, validate_checkpoint_config
from data import TDataSet
from model import make_model, zero_padding_embedding
from reproducibility import build_torch_generator, seed_everything, seed_worker


MODEL_CONFIG_KEYS = (
    "d_model",
    "vocab_size",
    "max_seq_len",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "padding_idx",
    "ignore_index",
)


def build_model_config(cfg):
    return checkpoint_model_config(cfg)


def normalize_int(name, value, minimum):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def build_lr_scheduler(
    cfg,
    optimizer,
    max_steps,
    warmup_steps=None,
    min_lr_ratio=None,
):
    warmup_steps = cfg.train.warmup_steps if warmup_steps is None else warmup_steps
    min_lr_ratio = cfg.train.min_lr_ratio if min_lr_ratio is None else min_lr_ratio
    warmup_steps = max(int(warmup_steps), 0)
    max_steps = max(int(max_steps), 1)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(max_steps - warmup_steps, 1)
        decay_progress = min(max(current_step - warmup_steps, 0) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def advance_scheduler_to_step(scheduler, step):
    step = max(int(step), 0)
    scheduler.last_epoch = step
    learning_rates = [
        base_lr * lr_lambda(step)
        for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    for param_group, lr in zip(scheduler.optimizer.param_groups, learning_rates):
        param_group["lr"] = lr
    scheduler._last_lr = learning_rates


@torch.no_grad()
def evaluate(cfg, model, dataloader, max_batches=None, skip_batches=0):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0
    max_batches = cfg.train.eval_batches if max_batches is None else max_batches
    max_batches = max(int(max_batches), 1)
    skip_batches = max(int(skip_batches), 0)

    for batch_index, batch in enumerate(dataloader):
        if batch_index < skip_batches:
            continue
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=cfg.train.ignore_index,
        )
        total_loss += loss.item()
        total_batches += 1
        if total_batches >= max_batches:
            break

    if was_training:
        model.train()
    if total_batches == 0:
        return None
    return total_loss / total_batches


def save_checkpoint(cfg, path, model, optimizer, scaler, step, mode, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "mode": mode,
        "config": cfg.to_dict(),
        "model_config": build_model_config(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(cfg, path, model, optimizer=None, scaler=None, scheduler=None, map_location=None, mode=None):
    map_location = cfg.device if map_location is None else map_location
    path = Path(path)
    if not path.exists():
        return 0, None

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    validate_checkpoint_config(checkpoint, cfg, source=str(path))
    load_model_state(model, checkpoint["model_state_dict"], source=str(path))
    zero_padding_embedding(model)

    checkpoint_mode = checkpoint.get("mode")
    if mode is not None and checkpoint_mode is not None and checkpoint_mode != mode:
        print(
            f"loaded model weights from {path}, reset optimizer state because "
            f"checkpoint mode is {checkpoint_mode}, current mode is {mode}",
            flush=True,
        )
        return 0, checkpoint_mode

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step = int(checkpoint.get("step", 0))
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif scheduler is not None and step > 0:
        advance_scheduler_to_step(scheduler, step)
        print(
            f"checkpoint has no scheduler_state_dict; advanced scheduler to step {step}",
            flush=True,
        )
    return step, checkpoint_mode


def train(
    cfg,
    data_path,
    mode,
    checkpoint_path=None,
    load_checkpoint_path=None,
    save_checkpoint_path=None,
    resume=True,
    max_steps=None,
    batch_size=None,
    num_workers=None,
    grad_accum_steps=None,
    eval_data_path=None,
    eval_steps=None,
    eval_batches=None,
    seed=None,
):
    max_steps = cfg.train.max_steps if max_steps is None else max_steps
    batch_size = cfg.train.batch_size if batch_size is None else batch_size
    num_workers = cfg.train.num_workers if num_workers is None else num_workers
    grad_accum_steps = cfg.train.grad_accum_steps if grad_accum_steps is None else grad_accum_steps
    eval_data_path = cfg.train.eval_data_path if eval_data_path is None else eval_data_path
    eval_steps = cfg.train.eval_steps if eval_steps is None else eval_steps
    eval_batches = cfg.train.eval_batches if eval_batches is None else eval_batches
    seed = cfg.runtime.seed if seed is None else seed
    max_steps = normalize_int("max_steps", max_steps, 1)
    batch_size = normalize_int("batch_size", batch_size, 1)
    num_workers = normalize_int("num_workers", num_workers, 0)
    grad_accum_steps = normalize_int("grad_accum_steps", grad_accum_steps, 1)
    eval_steps = normalize_int("eval_steps", eval_steps, 0)
    eval_batches = normalize_int("eval_batches", eval_batches, 1)
    log_steps = normalize_int("log_steps", cfg.train.log_steps, 1)
    save_steps = normalize_int("save_steps", cfg.train.save_steps, 1)
    seed = seed_everything(seed)
    train_generator = build_torch_generator(seed)
    eval_generator = build_torch_generator(seed + 1)
    worker_init_fn = seed_worker if num_workers > 0 else None

    explicit_save_checkpoint = save_checkpoint_path is not None
    load_checkpoint_path = cfg.get_checkpoint_path(
        mode=mode,
        checkpoint=load_checkpoint_path or checkpoint_path,
    )
    save_checkpoint_path = cfg.get_checkpoint_path(
        mode=mode,
        checkpoint=save_checkpoint_path or checkpoint_path,
    )
    dataset = TDataSet(cfg, data_path, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=cfg.device.type == "cuda",
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    eval_dataloader = None
    if eval_data_path:
        eval_dataset = TDataSet(cfg, eval_data_path, mode=mode)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=eval_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=cfg.device.type == "cuda",
            worker_init_fn=worker_init_fn,
            generator=eval_generator,
        )

    model = make_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,  weight_decay=cfg.train.weight_decay)
    scheduler = build_lr_scheduler(cfg, optimizer, max_steps=max_steps)
    use_amp = cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    checkpoint_mode = None
    if resume:
        step, checkpoint_mode = load_checkpoint(
            cfg,
            load_checkpoint_path,
            model,
            optimizer,
            scaler,
            scheduler,
            mode=mode,
        )
        if checkpoint_mode is not None and checkpoint_mode != mode and not explicit_save_checkpoint:
            save_checkpoint_path = cfg.get_checkpoint_path(mode=mode)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    started_at = time.time()
    running_loss = 0.0
    running_count = 0
    micro_step = 0

    def optimizer_step(grad_scale=1.0):
        if grad_scale != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(grad_scale)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = not use_amp or scaler.get_scale() >= scale_before_step
        if optimizer_stepped:
            scheduler.step()
            zero_padding_embedding(model)
        optimizer.zero_grad(set_to_none=True)
        return optimizer_stepped

    def after_optimizer_step():
        nonlocal running_loss, running_count
        if step % log_steps == 0:
            elapsed = max(time.time() - started_at, 1e-6)
            avg_loss = running_loss / max(running_count, 1)
            lr = optimizer.param_groups[0]["lr"]
            print(f"step={step} loss={avg_loss:.4f} lr={lr:.6g} speed={step / elapsed:.2f} steps/s", flush=True)
            running_loss = 0.0
            running_count = 0

        if eval_dataloader is not None and eval_steps > 0 and step % eval_steps == 0:
            eval_loss = evaluate(
                cfg,
                model,
                eval_dataloader,
                max_batches=eval_batches,
            )
            if eval_loss is not None:
                print(f"step={step} eval_loss={eval_loss:.4f}", flush=True)

        if step % save_steps == 0:
            save_checkpoint(cfg, save_checkpoint_path, model, optimizer, scaler, step, mode, scheduler)
            print(f"saved checkpoint: {save_checkpoint_path}", flush=True)

    while step < max_steps:
        has_batch = False
        for batch in dataloader:
            has_batch = True
            micro_step += 1
            input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)

            with torch.amp.autocast(device_type=cfg.device.type, enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=cfg.train.ignore_index,
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps
            running_count += 1

            if micro_step % grad_accum_steps == 0:
                if not optimizer_step():
                    continue

                step += 1
                after_optimizer_step()

                if step >= max_steps:
                    break
        if not has_batch:
            raise ValueError(f"no valid samples loaded from {data_path}")
        residual_micro_steps = micro_step % grad_accum_steps
        if residual_micro_steps and step < max_steps:
            grad_scale = grad_accum_steps / residual_micro_steps
            if optimizer_step(grad_scale=grad_scale):
                step += 1
                after_optimizer_step()
            micro_step = 0

    save_checkpoint(cfg, save_checkpoint_path, model, optimizer, scaler, step, mode, scheduler)
    print(f"train complete: step={step}, checkpoint={save_checkpoint_path}", flush=True)
    return model


def entry():
    cfg = load_config()
    data_path = cfg.get_train_data_path(
        mode=cfg.train.mode,
        language=cfg.train.language,
    )
    checkpoint_path = cfg.get_checkpoint_path(mode=cfg.train.mode)
    train(cfg, data_path, mode=cfg.train.mode, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    entry()
