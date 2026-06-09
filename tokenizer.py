from pathlib import Path
import ctypes
import os
import subprocess
import sys
import threading
import time

import sentencepiece as spm


DEFAULT_INPUT_SENTENCE_SIZE = 5_000_000
DEFAULT_CHECK_TEXT = "\u4f60\u597d, Transformer tokenizer check."
PROGRESS_MONITOR_CODE = r"""
import ctypes
import os
import sys
import time


class IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


def format_bytes(value):
    value = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def read_bytes(handle):
    counters = IoCounters()
    ok = ctypes.windll.kernel32.GetProcessIoCounters(handle, ctypes.byref(counters))
    if not ok:
        return None
    return int(counters.ReadTransferCount)


def main():
    pid = int(sys.argv[1])
    corpus_path = sys.argv[2]
    label = sys.argv[3]
    interval = float(sys.argv[4])
    total_bytes = os.path.getsize(corpus_path) if os.path.exists(corpus_path) else None
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        print(f"{label}: elapsed=0s", flush=True)
        return

    started_at = time.monotonic()
    start_read_bytes = read_bytes(handle)
    while True:
        time.sleep(interval)
        elapsed = max(time.monotonic() - started_at, 1e-6)
        current_read_bytes = read_bytes(handle)
        if current_read_bytes is None or start_read_bytes is None or total_bytes is None:
            print(f"{label}: elapsed={format_duration(elapsed)}", flush=True)
            continue

        read_size = max(current_read_bytes - start_read_bytes, 0)
        shown_size = min(read_size, total_bytes)
        percent = shown_size / max(total_bytes, 1) * 100
        speed = read_size / elapsed
        eta = (
            format_duration((total_bytes - shown_size) / speed)
            if speed > 0 and shown_size < total_bytes
            else "0s"
        )
        status = (
            f"{label}: read={format_bytes(shown_size)}/{format_bytes(total_bytes)} "
            f"({percent:.1f}%) speed={format_bytes(speed)}/s "
            f"elapsed={format_duration(elapsed)} eta={eta}"
        )
        if read_size > total_bytes:
            status += " corpus_read_complete"
        print(status, flush=True)


if __name__ == "__main__":
    main()
"""


class ProcessReadProgress:
    def __init__(self, path, label="progress", interval=5.0):
        self.path = Path(path)
        self.label = label
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.started_at = None
        self.start_read_bytes = None
        self.total_bytes = self.path.stat().st_size if self.path.exists() else None
        self.process = None

    def start(self):
        if os.name == "nt" and self.start_subprocess_monitor():
            return

        self.started_at = time.monotonic()
        self.start_read_bytes = process_read_bytes()
        self.report()
        self.thread.start()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            return

        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.report(final=True)

    def start_subprocess_monitor(self):
        args = (
            sys.executable,
            "-u",
            "-c",
            PROGRESS_MONITOR_CODE,
            str(os.getpid()),
            str(self.path),
            self.label,
            str(self.interval),
        )
        try:
            self.process = subprocess.Popen(args)
        except OSError:
            self.process = None
            return False
        return True

    def run(self):
        while not self.stop_event.wait(self.interval):
            self.report()

    def report(self, final=False):
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        current_read_bytes = process_read_bytes()
        prefix = f"{self.label}:"

        if self.total_bytes is None or current_read_bytes is None or self.start_read_bytes is None:
            status = f"{prefix} elapsed={format_duration(elapsed)}"
            if final:
                status += " done"
            print(status, flush=True)
            return

        read_bytes = max(current_read_bytes - self.start_read_bytes, 0)
        shown_bytes = min(read_bytes, self.total_bytes)
        percent = shown_bytes / max(self.total_bytes, 1) * 100
        speed = read_bytes / elapsed
        eta = format_duration((self.total_bytes - shown_bytes) / speed) if speed > 0 and shown_bytes < self.total_bytes else "0s"

        status = (
            f"{prefix} read={format_bytes(shown_bytes)}/{format_bytes(self.total_bytes)} "
            f"({percent:.1f}%) speed={format_bytes(speed)}/s "
            f"elapsed={format_duration(elapsed)} eta={eta}"
        )
        if read_bytes > self.total_bytes:
            status += " corpus_read_complete"
        if final:
            status += " done"
        print(status, flush=True)


def process_read_bytes():
    if os.name == "nt":
        return windows_process_read_bytes()
    return procfs_process_read_bytes()


def windows_process_read_bytes():
    class IoCounters(ctypes.Structure):
        _fields_ = (
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        )

    counters = IoCounters()
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.kernel32.GetProcessIoCounters(handle, ctypes.byref(counters))
    if not ok:
        return None
    return int(counters.ReadTransferCount)


def procfs_process_read_bytes():
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("rchar:"):
                    return int(line.split(":", 1)[1].strip())
    except OSError:
        return None
    return None


def format_bytes(value):
    value = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


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


def check(cfg, text=DEFAULT_CHECK_TEXT, max_items=80):
    model_path = Path(cfg.paths.tokenizer_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"tokenizer model does not exist: {model_path}")

    sp = tokenizer_load(cfg)
    text = DEFAULT_CHECK_TEXT if text is None else str(text)
    pieces = sp.encode(text, out_type=str)
    raw_ids = sp.encode(text, out_type=int)
    content_ids = encode_content(cfg, sp, text)
    decoded = sp.decode(content_ids)

    print("", flush=True)
    print("=== Tokenizer check ===", flush=True)
    print(f"model_path: {cfg.paths.tokenizer_model_path}", flush=True)
    print(f"vocab_size: {sp.GetPieceSize()}", flush=True)
    print(f"text: {text}", flush=True)
    print(f"piece_count: {len(pieces)}", flush=True)
    print(f"pieces: {pieces[:max_items]}", flush=True)
    print(f"raw_ids: {raw_ids[:max_items]}", flush=True)
    print(f"content_ids: {content_ids[:max_items]}", flush=True)
    print(f"decoded: {decoded}", flush=True)
    print("special_token_ids:", flush=True)
    for token, expected_id in cfg.tokens.all_token_ids.items():
        print(f"  {token}: {sp.piece_to_id(token)}", flush=True)
    print("OK: tokenizer loaded, special tokens validated, encode/decode succeeded", flush=True)


def entry(
    cfg,
    show_progress=True,
    input_sentence_size=DEFAULT_INPUT_SENTENCE_SIZE,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
):
    Path(cfg.paths.tokenizer_output_dir).mkdir(parents=True, exist_ok=True)
    input_sentence_size = int(input_sentence_size)

    progress = None
    if show_progress:
        progress = ProcessReadProgress(cfg.paths.tokenizer_corpus_path, label="tokenizer progress")
        progress.start()

    try:
        spm.SentencePieceTrainer.train(
            input=cfg.paths.tokenizer_corpus_path,
            model_prefix=cfg.paths.tokenizer_model_prefix,
            vocab_size=cfg.model.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=bool(shuffle_input_sentence),
            train_extremely_large_corpus=bool(train_extremely_large_corpus),
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
    finally:
        if progress is not None:
            progress.stop()
    tokenizer_load(cfg, cfg.paths.tokenizer_model_path)


if __name__ == "__main__":
    from config import load_config

    entry(load_config())
