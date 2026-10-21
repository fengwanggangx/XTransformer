from itertools import islice
import math
import os
from pathlib import Path
import tempfile
import time
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import checkpoint_model_config, load_config
from checkpoint import load_model_state, tokenizer_fingerprint, validate_checkpoint_config
from data import TDataSet
from model import make_model, zero_padding_embedding
from reproducibility import (
    build_torch_generator,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


MODEL_CONFIG_KEYS = (
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
    total_tokens = 0
    total_batches = 0
    max_batches = cfg.train.eval_batches if max_batches is None else max_batches
    max_batches = max(int(max_batches), 1)
    skip_batches = max(int(skip_batches), 0)

    try:
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
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.ne(cfg.train.ignore_index).sum().item()
            total_batches += 1
            if total_batches >= max_batches:
                break
    finally:
        model.train(was_training)

    return total_loss / total_tokens if total_tokens else None


def save_checkpoint(cfg, path, model, optimizer, scaler, step, mode, scheduler=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "mode": mode,
        "config": cfg.to_dict(),
        "model_config": build_model_config(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_sha256": tokenizer_fingerprint(cfg),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if training_state is not None:
        checkpoint["training_state"] = training_state

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def inspect_checkpoint(path):
    path = Path(path)
    info = {
        "exists": path.exists(),
        "step": 0,
        "mode": None,
        "error": "",
    }
    if not info["exists"]:
        return info

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["step"] = int(checkpoint.get("step", 0))
    info["mode"] = checkpoint.get("mode")
    return info


def load_checkpoint(cfg, path, model, optimizer=None, scaler=None, scheduler=None, map_location=None, mode=None, required=True, training_state=None):
    map_location = "cpu" if map_location is None else map_location
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
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
    if training_state is not None:
        saved_training_state = checkpoint.get("training_state")
        if saved_training_state is not None:
            training_state.update(saved_training_state)
        elif step > 0:
            warnings.warn(
                f"{path} has no training_state; data will restart from the beginning "
                "and random state cannot be restored",
                RuntimeWarning,
                stacklevel=2,
            )
    return step, checkpoint_mode


def build_data_config(dataset, batch_size, num_workers, grad_accum_steps, seed):
    files = []
    for path in dataset.get_path_files(dataset.path):
        stat = path.stat()
        files.append({
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "files": files,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "grad_accum_steps": grad_accum_steps,
        "seed": seed,
    }


def train(
    cfg,
    data_path,
    mode,
    checkpoint_path=None,
    load_checkpoint_path=None,
    save_checkpoint_path=None,
    resume=True,
    max_steps=None,
    train_steps=None,
    batch_size=None,
    num_workers=None,
    grad_accum_steps=None,
    eval_data_path=None,
    eval_steps=None,
    eval_batches=None,
    seed=None,
    reset_data_state=False,
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
    train_steps = None if train_steps is None else normalize_int("train_steps", train_steps, 1)
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
    explicit_load_checkpoint = load_checkpoint_path is not None
    if explicit_load_checkpoint and not resume:
        raise ValueError("--load-checkpoint cannot be combined with --no-resume")
    load_checkpoint_path = cfg.get_checkpoint_path(
        mode=mode,
        checkpoint=load_checkpoint_path or checkpoint_path,
    )
    save_checkpoint_path = cfg.get_checkpoint_path(
        mode=mode,
        checkpoint=save_checkpoint_path or checkpoint_path,
    )
    if explicit_load_checkpoint and not Path(load_checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {load_checkpoint_path}")
    if train_steps is not None:
        start_step_for_target = 0
        if resume:
            checkpoint_info = inspect_checkpoint(load_checkpoint_path)
            checkpoint_mode = checkpoint_info.get("mode")
            if (
                checkpoint_info.get("exists")
                and not checkpoint_info.get("error")
                and (checkpoint_mode is None or checkpoint_mode == mode)
            ):
                start_step_for_target = int(checkpoint_info.get("step", 0))
        max_steps = normalize_int("max_steps", start_step_for_target + train_steps, 1)

    dataset = TDataSet(cfg, data_path, mode=mode)
    data_config = build_data_config(dataset, batch_size, num_workers, grad_accum_steps, seed)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_lr_scheduler(cfg, optimizer, max_steps=max_steps)
    use_amp = cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    checkpoint_mode = None
    resumed_training_state = {}
    if resume:
        step, checkpoint_mode = load_checkpoint(
            cfg,
            load_checkpoint_path,
            model,
            optimizer,
            scaler,
            scheduler,
            mode=mode,
            required=explicit_load_checkpoint,
            training_state=resumed_training_state,
        )
        if checkpoint_mode is not None and checkpoint_mode != mode and not explicit_save_checkpoint:
            save_checkpoint_path = cfg.get_checkpoint_path(mode=mode)

    if reset_data_state:
        resumed_training_state.clear()
    if resumed_training_state and resumed_training_state["data_config"] != data_config:
        raise ValueError(
            "training data or batching configuration changed; use --reset-data-state "
            "to explicitly restart data traversal while retaining the checkpoint weights and optimizer"
        )
    epoch = int(resumed_training_state.get("epoch", 0))
    batches_in_epoch = int(resumed_training_state.get("batches_in_epoch", 0))
    epoch_generator_state = resumed_training_state.get("epoch_generator_state")
    resume_rng_state = resumed_training_state.get("rng_state")
    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
        eval_generator.set_state(resumed_training_state["eval_generator_state"].cpu())

    start_step = step
    planned_train_steps = max(max_steps - start_step, 0)
    print(
        f"training progress: start_step={start_step} "
        f"planned_train_steps={planned_train_steps} "
        f"target_total_steps={max_steps}",
        flush=True,
    )
    if batches_in_epoch:
        print(f"resuming data: epoch={epoch} replay_batches={batches_in_epoch}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    started_at = time.time()
    running_loss = 0.0
    running_tokens = 0

    def optimizer_step():
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

    def save_training_checkpoint():
        training_state = {
            "data_config": data_config,
            "epoch": epoch,
            "batches_in_epoch": batches_in_epoch,
            "epoch_generator_state": (
                epoch_generator_state
                if epoch_generator_state is not None
                else train_generator.get_state()
            ),
            "eval_generator_state": eval_generator.get_state(),
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(
            cfg, save_checkpoint_path, model, optimizer, scaler, step, mode,
            scheduler, training_state=training_state,
        )

    def after_optimizer_step():
        nonlocal running_loss, running_tokens
        if step % log_steps == 0:
            elapsed = max(time.time() - started_at, 1e-6)
            run_steps = max(step - start_step, 0)
            avg_loss = running_loss / max(running_tokens, 1)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} loss={avg_loss:.4f} lr={lr:.6g} "
                f"speed={run_steps / elapsed:.2f} steps/s "
                f"trained_total_steps={step} run_steps={run_steps} "
                f"target_total_steps={max_steps}",
                flush=True,
            )
            running_loss = 0.0
            running_tokens = 0

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
            save_training_checkpoint()
            print(f"saved checkpoint: {save_checkpoint_path}", flush=True)

    while step < max_steps:
        if epoch_generator_state is None:
            epoch_generator_state = train_generator.get_state()
        else:
            train_generator.set_state(epoch_generator_state.cpu())
        iterator = iter(dataloader)
        # Replay only the current epoch to recover the consumed DataLoader order,
        # including worker sharding and partial batches; prefetch is not a cursor.
        for _ in range(batches_in_epoch):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("checkpoint data cursor exceeds the current epoch") from exc
        if resume_rng_state is not None:
            restore_rng_state(resume_rng_state)
            resume_rng_state = None

        has_batch = batches_in_epoch > 0
        while step < max_steps:
            # Only token/label tensors are buffered, not model activations. Knowing
            # the window's token count before backward also keeps AMP scaling sane.
            batches = list(islice(iterator, grad_accum_steps))
            if not batches:
                break
            has_batch = True
            batches_in_epoch += len(batches)
            window_tokens = sum(
                batch["labels"].ne(cfg.train.ignore_index).sum().item()
                for batch in batches
            )
            if window_tokens == 0:
                raise ValueError("gradient accumulation window has no trainable tokens")
            for batch in batches:
                input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
                labels = batch["labels"].to(cfg.device, non_blocking=True)

                with torch.amp.autocast(device_type=cfg.device.type, enabled=use_amp):
                    logits = model(input_ids)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=cfg.train.ignore_index,
                        reduction="sum",
                    )
                    loss = loss_sum / window_tokens

                scaler.scale(loss).backward()
                running_loss += loss_sum.item()
            running_tokens += window_tokens

            if optimizer_step():
                step += 1
                after_optimizer_step()
        if not has_batch:
            raise ValueError(f"no valid samples loaded from {data_path}")
        if step >= max_steps:
            break
        epoch += 1
        batches_in_epoch = 0
        epoch_generator_state = None

    save_training_checkpoint()
    print(
        f"train complete: step={step}, trained_total_steps={step}, "
        f"run_steps={max(step - start_step, 0)}, checkpoint={save_checkpoint_path}",
        flush=True,
    )
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
