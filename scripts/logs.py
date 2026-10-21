"""Append selected training output to a project-local daily UTF-8 log."""

import builtins
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def log_print(*values, **kwargs):
    builtins.print(*values, **kwargs)
    text = kwargs.get("sep", " ").join(str(value) for value in values)
    # Keep validation metrics, but leave frequent training loss lines on screen.
    if text.startswith("step=") and " eval_loss=" not in text:
        return
    now = datetime.now()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"{now:%Y%m%d}.log").open("a", encoding="utf-8") as stream:
        stream.write(f"[{now:%Y-%m-%d %H:%M:%S}] {text}\n")
