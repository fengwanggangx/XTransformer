
import argparse

from config import DEFAULT_CONFIG_PATH, load_config
import inference
import tokenizer
import train


def build_parser(cfg):
    parser = argparse.ArgumentParser(description="Transformer training and inference entrypoint.")
    parser.add_argument("--config", default=str(cfg.config_path))
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tokenizer", help="train sentencepiece tokenizer")

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
        tokenizer.entry(cfg)
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
