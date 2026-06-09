
import argparse

from config import DEFAULT_CONFIG_PATH, load_config
import inference
import tokenizer
import train


def print_run_settings(title, rows):
    print("", flush=True)
    print(f"=== {title} settings ===", flush=True)
    for name, value in rows:
        print(f"{name}: {value}", flush=True)
    print("", flush=True)


def confirm_execution(yes=False):
    if yes:
        return True

    answer = input("Proceed? [Y/N]: ").strip().lower()
    return answer in ("y", "yes")


def print_tokenizer_settings(cfg, args):
    print_run_settings(
        "Tokenizer",
        (
            ("config", cfg.config_path),
            ("device", cfg.device),
            ("corpus_path", cfg.paths.tokenizer_corpus_path),
            ("model_prefix", cfg.paths.tokenizer_model_prefix),
            ("model_path", cfg.paths.tokenizer_model_path),
            ("vocab_path", cfg.paths.tokenizer_vocab_path),
            ("vocab_size", cfg.model.vocab_size),
            ("model_type", "bpe"),
            ("character_coverage", 0.9995),
            ("input_sentence_size", args.input_sentence_size),
            ("shuffle_input_sentence", args.shuffle_input_sentence),
            ("train_extremely_large_corpus", not args.no_large_corpus),
            ("special_tokens", ", ".join(cfg.tokens.special_tokens)),
        ),
    )


def print_train_settings(
    cfg,
    args,
    data_path,
    checkpoint_path,
    load_checkpoint_path,
    save_checkpoint_path,
):
    effective_batch = args.batch_size * args.grad_accum_steps
    print_run_settings(
        "Training",
        (
            ("config", cfg.config_path),
            ("mode", args.mode),
            ("language", args.language),
            ("device", cfg.device),
            ("data_path", data_path),
            ("resume", not args.no_resume),
            ("load_checkpoint", load_checkpoint_path),
            ("save_checkpoint", save_checkpoint_path),
            ("default_checkpoint", checkpoint_path),
            ("seed", args.seed),
            ("d_embed", cfg.model.d_embed),
            ("vocab_size", cfg.model.vocab_size),
            ("max_seq_len", cfg.model.max_seq_len),
            ("n_heads", cfg.model.n_heads),
            ("n_layers", cfg.model.n_layers),
            ("d_ff", cfg.model.d_ff),
            ("dropout", cfg.model.dropout),
            ("batch_size", args.batch_size),
            ("grad_accum_steps", args.grad_accum_steps),
            ("effective_batch_samples", effective_batch),
            ("max_steps", args.max_steps),
            ("learning_rate", cfg.train.learning_rate),
            ("weight_decay", cfg.train.weight_decay),
            ("warmup_steps", cfg.train.warmup_steps),
            ("min_lr_ratio", cfg.train.min_lr_ratio),
            ("max_grad_norm", cfg.train.max_grad_norm),
            ("num_workers", args.num_workers),
            ("log_steps", cfg.train.log_steps),
            ("save_steps", cfg.train.save_steps),
            ("eval_data_path", args.eval_data_path or ""),
            ("eval_steps", args.eval_steps),
            ("eval_batches", args.eval_batches),
        ),
    )


def build_parser(cfg):
    parser = argparse.ArgumentParser(description="Transformer training and inference entrypoint.")
    parser.add_argument("--config", default=str(cfg.config_path))
    subparsers = parser.add_subparsers(dest="command")

    tokenizer_parser = subparsers.add_parser("tokenizer", help="train sentencepiece tokenizer")
    tokenizer_parser.add_argument("-y", "--yes", action="store_true", help="run without confirmation")
    tokenizer_parser.add_argument("--no-progress", action="store_true", help="hide tokenizer progress output")
    tokenizer_parser.add_argument(
        "--check",
        nargs="*",
        default=None,
        help="check the trained tokenizer with optional text",
    )
    tokenizer_parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=tokenizer.DEFAULT_INPUT_SENTENCE_SIZE,
        help="number of sentences to sample for tokenizer training",
    )
    tokenizer_parser.add_argument(
        "--shuffle-input-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle sampled sentences before tokenizer training",
    )
    tokenizer_parser.add_argument(
        "--no-large-corpus",
        action="store_true",
        help="disable SentencePiece large corpus mode",
    )

    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("--data-path", default=None)
    train_parser.add_argument("--mode", choices=cfg.train_modes, default=cfg.train.mode)
    train_parser.add_argument("--language", choices=cfg.train_languages, default=cfg.train.language)
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--load-checkpoint", default=None)
    train_parser.add_argument("--save-checkpoint", default=None)
    train_parser.add_argument("--max-steps", type=int, default=cfg.train.max_steps)
    train_parser.add_argument("--batch-size", type=int, default=cfg.train.batch_size)
    train_parser.add_argument("--num-workers", type=int, default=cfg.train.num_workers)
    train_parser.add_argument("--grad-accum-steps", type=int, default=cfg.train.grad_accum_steps)
    train_parser.add_argument("--eval-data-path", default=cfg.train.eval_data_path)
    train_parser.add_argument("--eval-steps", type=int, default=cfg.train.eval_steps)
    train_parser.add_argument("--eval-batches", type=int, default=cfg.train.eval_batches)
    train_parser.add_argument("--seed", type=int, default=cfg.runtime.seed)
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.add_argument("-y", "--yes", action="store_true", help="run without confirmation")

    infer_parser = subparsers.add_parser("infer", help="run inference")
    infer_parser.add_argument("--prompt", default="")
    infer_parser.add_argument("--system", default="")
    infer_parser.add_argument("--checkpoint", default=cfg.get_checkpoint_path())
    infer_parser.add_argument("--max-new-tokens", type=int, default=cfg.inference.max_new_tokens)
    infer_parser.add_argument("--temperature", type=float, default=cfg.inference.temperature)
    infer_parser.add_argument("--top-k", type=int, default=cfg.inference.top_k)
    infer_parser.add_argument("--top-p", type=float, default=cfg.inference.top_p)
    infer_parser.add_argument("--seed", type=int, default=None)
    infer_parser.add_argument("--no-stop-at-sep", dest="stop_at_sep", action="store_false", default=None)

    return parser


def entry():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap.parse_known_args()
    cfg = load_config(bootstrap_args.config)

    parser = build_parser(cfg)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "tokenizer":
        if args.check is not None:
            check_text = " ".join(args.check) if args.check else tokenizer.DEFAULT_CHECK_TEXT
            tokenizer.check(cfg, text=check_text)
            return
        print_tokenizer_settings(cfg, args)
        if not confirm_execution(args.yes):
            print("Aborted.", flush=True)
            return
        tokenizer.entry(
            cfg,
            show_progress=not args.no_progress,
            input_sentence_size=args.input_sentence_size,
            shuffle_input_sentence=args.shuffle_input_sentence,
            train_extremely_large_corpus=not args.no_large_corpus,
        )
    elif args.command == "train":
        data_path = cfg.get_train_data_path(
            mode=args.mode,
            language=args.language,
            data_path=args.data_path,
        )
        checkpoint_path = cfg.get_checkpoint_path(
            mode=args.mode,
            checkpoint=args.checkpoint,
        )
        load_checkpoint_path = cfg.get_checkpoint_path(
            mode=args.mode,
            checkpoint=args.load_checkpoint or checkpoint_path,
        )
        save_checkpoint_path = cfg.get_checkpoint_path(
            mode=args.mode,
            checkpoint=args.save_checkpoint or checkpoint_path,
        )
        print_train_settings(
            cfg,
            args,
            data_path,
            checkpoint_path,
            load_checkpoint_path,
            save_checkpoint_path,
        )
        if not confirm_execution(args.yes):
            print("Aborted.", flush=True)
            return
        train.train(
            cfg,
            data_path=data_path,
            mode=args.mode,
            checkpoint_path=checkpoint_path,
            load_checkpoint_path=args.load_checkpoint,
            save_checkpoint_path=args.save_checkpoint,
            resume=not args.no_resume,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            grad_accum_steps=args.grad_accum_steps,
            eval_data_path=args.eval_data_path,
            eval_steps=args.eval_steps,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )
    elif args.command == "infer":
        prompt = args.prompt or input("user> ").strip()
        print(
            inference.generate(
                cfg,
                prompt,
                system=args.system,
                checkpoint_path=args.checkpoint,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                stop_at_sep=args.stop_at_sep,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    entry()
