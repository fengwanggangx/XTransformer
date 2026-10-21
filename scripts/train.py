"""Start or resume one training stage using the project's existing trainer."""

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import DEFAULT_CONFIG_PATH, TRAIN_LANGUAGES, TRAIN_MODES, load_config
from data import TDataSet
from tokenizer import tokenizer_load


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=TRAIN_MODES, default="pretrain")
    parser.add_argument("--language", choices=TRAIN_LANGUAGES, default="chinese")
    parser.add_argument("--data-path", help="training file or directory, relative to the project root")
    parser.add_argument("--eval-data-path", help="validation file or directory")
    parser.add_argument("--no-eval", action="store_true", help="disable validation")
    parser.add_argument("--checkpoint", help="this stage's output/resume checkpoint")
    parser.add_argument("--load-checkpoint", help="override the checkpoint to load")
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--train-steps", type=positive_int, help="additional optimizer steps (default: 1000)")
    steps.add_argument("--max-steps", type=positive_int, help="total target steps, including saved steps")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--grad-accum-steps", type=positive_int, default=4)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--eval-steps", type=positive_int, default=100)
    parser.add_argument("--eval-batches", type=positive_int, default=10)
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--reset-data-state", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="explicitly allow CPU training")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print the command without training")
    parser.add_argument(
        "--exit-on-complete", action="store_true",
        help="exit after training instead of keeping the process and model loaded",
    )
    return parser


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def training_data_path(cfg, args):
    if args.data_path or cfg.train.data_path:
        return project_path(args.data_path or cfg.train.data_path)

    default = project_path(cfg.get_train_data_path(args.mode, args.language))
    if args.mode == "pretrain" and args.language != "all" and not default.exists():
        filenames = {"chinese": "wiki_zh.txt", "english": "wiki_en_balanced.txt"}
        flat_file = Path(cfg.paths.pretrain_dir) / filenames[args.language]
        if flat_file.is_file():
            return flat_file
    return default


def validation_data_path(cfg, args):
    if args.no_eval:
        return None
    if args.eval_data_path is not None:
        return project_path(args.eval_data_path)
    if cfg.train.eval_data_path:
        return project_path(cfg.train.eval_data_path)
    if args.mode != "pretrain":
        return None

    root = Path(cfg.paths.eval_dir) / "pretrain"
    filenames = {"chinese": "wiki_zh_eval.txt", "english": "wiki_en_eval.txt"}
    path = root if args.language == "all" else root / filenames[args.language]
    return path if path.exists() else None


def checkpoint_paths(cfg, args):
    output = project_path(args.checkpoint or cfg.get_checkpoint_path(args.mode))
    if args.load_checkpoint:
        source = project_path(args.load_checkpoint)
    elif output.exists():
        source = output
    elif args.mode == "pretrain":
        source = None
    else:
        previous = {"sft_single": "pretrain", "sft_multi": "sft_single"}[args.mode]
        source = project_path(cfg.get_checkpoint_path(previous))

    if source is not None and not source.is_file():
        raise ValueError(
            f"checkpoint does not exist: {source}; train the preceding stage first "
            "or specify --load-checkpoint"
        )
    return source, output


def build_command(cfg, args):
    data_path = training_data_path(cfg, args)
    eval_path = validation_data_path(cfg, args)
    source, output = checkpoint_paths(cfg, args)
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
        "--config", str(cfg.config_path), "train", "--yes",
        "--mode", args.mode, "--language", args.language,
        "--data-path", str(data_path), "--checkpoint", str(output),
        "--save-checkpoint", str(output),
        "--batch-size", str(args.batch_size),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--num-workers", str(args.num_workers),
        "--eval-data-path", str(eval_path) if eval_path is not None else "",
        "--eval-steps", str(args.eval_steps), "--eval-batches", str(args.eval_batches),
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    else:
        command.extend(["--train-steps", str(args.train_steps or 1000)])
    if source is not None:
        command.extend(["--load-checkpoint", str(source)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.reset_data_state:
        command.append("--reset-data-state")
    if not args.exit_on_complete:
        command.append("--keep-alive")
    return command, data_path, eval_path, source, output


def check_inputs(cfg, args, data_path, eval_path):
    if cfg.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("configuration selects CUDA, but CUDA is unavailable in this Python environment")
    if cfg.device.type != "cuda" and not args.allow_cpu:
        raise ValueError(
            f"GPU training is unavailable with {sys.executable} (device={cfg.device}). "
            "Use scripts/train.cmd or activate python_31208; use --allow-cpu for intentional CPU training."
        )

    tokenizer = tokenizer_load(cfg)
    for label, path in (("training", data_path), ("validation", eval_path)):
        if path is None:
            continue
        dataset = TDataSet(cfg, path, mode=args.mode)
        dataset.tokenizer = tokenizer
        if next(iter(dataset), None) is None:
            raise ValueError(f"no valid {label} samples found in {path}")
    return tokenizer.GetPieceSize()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(project_path(args.config))
        command, data_path, eval_path, source, output = build_command(cfg, args)
        vocab_size = check_inputs(cfg, args, data_path, eval_path)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Training setup failed: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"Python: {sys.executable}", flush=True)
    print(f"Device: {cfg.device}; tokenizer vocabulary: {vocab_size}", flush=True)
    print(f"Stage: {args.mode}; language: {args.language}", flush=True)
    print(f"Training data: {data_path}", flush=True)
    print(f"Validation data: {eval_path or 'disabled'}", flush=True)
    print(f"Load checkpoint: {source or 'new pretraining model'}", flush=True)
    print(f"Save checkpoint: {output}", flush=True)
    print("Command: " + subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        print("Checks passed. Dry run completed.", flush=True)
        return 0
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except KeyboardInterrupt:
        print("Training interrupted. Resume from the latest saved checkpoint on the next run.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
