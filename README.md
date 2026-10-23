# Transformer

Decoder-only Transformer language model with SentencePiece tokenizer, streaming datasets, training, checkpoint, and inference entrypoints.

## Environment

Install dependencies in the training environment:

```bash
pip install -r requirements.txt
```

## Configuration

Project settings are loaded from YAML. The default file is:

```text
configs/default.yaml
```

Use `--config` before the subcommand to select a different file:

```bash
python main.py --config configs/default.yaml train --mode pretrain
```

## Tokenizer

Tokenizer training reads:

```text
data/tokenizer/corpus.txt
```

Run:

```bash
python main.py tokenizer
```

Special token ids are fixed in `configs/default.yaml`:

| Token | Id |
| --- | ---: |
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |
| `<SYSTEM>` | 4 |
| `<USER>` | 5 |
| `<ASSISTANT>` | 6 |
| `<SEP>` | 7 |

## Data Format

`data.py` supports `.txt` and `.jsonl` files. Directory paths are selected by `--mode` and `--language`, or overridden by `--data-path`.

### Pretrain

Default directories:

```text
data/pretrain
data/pretrain/chinese
data/pretrain/english
```

`.txt` format: one sample per non-empty line.

```text
This is one training document.
This is another training document.
```

`.jsonl` format: one JSON object per line. Supported text fields are `text` and `content`.

```jsonl
{"text": "This is one training document."}
{"content": "This is another training document."}
```

Pretrain samples are trained on all next-token targets, including EOS. Long documents are split into blocks with one overlapping token so that block boundaries do not lose prediction targets.

### SFT Single Turn

Default directories:

```text
data/sft/single_turn
data/sft/single_turn/chinese
data/sft/single_turn/english
```

`.jsonl` format:

```jsonl
{"system": "You are a helpful assistant.", "instruction": "Explain RoPE.", "input": "", "response": "RoPE applies rotary position information to query and key."}
```

Supported aliases:

| Meaning | Fields |
| --- | --- |
| system prompt | `system` |
| user instruction | `instruction`, `question`, `prompt`, `user` |
| extra user input | `input` |
| assistant response | `response`, `output`, `answer`, `assistant` |

Only assistant response tokens are trained. System and user tokens are used as context and masked out from loss.

### SFT Multi Turn

Default directories:

```text
data/sft/multi_turn
data/sft/multi_turn/chinese
data/sft/multi_turn/english
```

`.jsonl` format:

```jsonl
{"system": "You are a helpful assistant.", "messages": [{"role": "user", "content": "Explain RoPE."}, {"role": "assistant", "content": "RoPE rotates query and key features by position."}]}
```

The conversation list can be named `messages` or `conversations`.

Each message supports:

| Meaning | Fields |
| --- | --- |
| role | `role`, `from` |
| content | `content`, `value` |

Supported role aliases:

| Input role | Normalized role |
| --- | --- |
| `human`, `user` | `user` |
| `gpt`, `assistant`, `bot` | `assistant` |
| `system` | `system` |

Only assistant messages are trained. System and user messages are context only. Trailing user/system messages after the last assistant reply are discarded before adding EOS. Long SFT samples retain the user/assistant role markers and the separator between them; samples too short to fit these markers and an assistant target are skipped.

## Batching

`TDataSet.collate_fn` returns:

```python
{
    "input_ids": LongTensor[B, T],
    "labels": LongTensor[B, T],
    "token_mask": LongTensor[B, T],
}
```

`token_mask` marks real tokens as `1` and padding as `0`. It is not the Transformer attention mask. The model builds its own causal padding mask when `model(input_ids)` is called.

## Training

For the prepared `python_31208` environment and Wiki datasets, start or resume Chinese pretraining with:

```powershell
.\scripts\train.cmd
```

The launcher checks CUDA, the tokenizer and data, uses batch size 1 and gradient accumulation 4, and trains 1000 additional steps per invocation. Use `--dry-run` to check setup without training, `--max-steps 1000` for a fixed total step target, and `--mode sft_single` / `--mode sft_multi` for fine-tuning. See [scripts/README.md](scripts/README.md) for data selection, validation, checkpoint selection and resume examples.

After saving the final checkpoint, the launcher keeps the training process and model loaded until you type `exit` or press Ctrl+C. It does not automatically train additional steps. Use `--exit-on-complete` to exit automatically after saving.

Pretrain:

```bash
python main.py train --mode pretrain --language all
```

With an explicit config:

```bash
python main.py --config configs/default.yaml train --mode pretrain --language all
```

Single-turn SFT from a pretrain checkpoint:

```bash
python main.py train --mode sft_single --load-checkpoint model/checkpoints/pretrain.pt
```

Multi-turn SFT:

```bash
python main.py train --mode sft_multi --load-checkpoint model/checkpoints/sft_single.pt
```

Train with validation:

```bash
python main.py train --mode pretrain --eval-data-path data/eval --eval-steps 100 --eval-batches 20
```

Set `--seed` to make model initialization, DataLoader worker seeds, and sampling reproducible:

```bash
python main.py train --mode pretrain --seed 42
```

Use `--save-checkpoint` to choose a custom save path. If a checkpoint from another training mode is loaded and `--save-checkpoint` is not provided, training saves to the current mode's default checkpoint path.

An explicit `--load-checkpoint` must exist and cannot be combined with `--no-resume`. A missing default checkpoint still starts a new training run. Checkpoints are written to a temporary file in the destination directory and atomically replace the previous file only after a successful write.

New checkpoints include the tokenizer's SHA-256 fingerprint and full special-token configuration. Training and inference reject incompatible tokenizers even when their vocabulary sizes match. Legacy checkpoints without a fingerprint remain loadable with a warning; keep the original tokenizer when loading them.

Same-mode resumes restore the current epoch, consumed batch count, Python/PyTorch random states, and DataLoader generator states. The current epoch is replayed up to the saved batch position without training, which can take time for large datasets. The data files (paths, sizes, modification times), batch size, worker count, accumulation count, and seed must match. Continuous and resumed training use the same data order and random sequence when the data, training schedule, and execution environment are unchanged; CUDA kernels may still be nondeterministic.

Checkpoints retain separate corpus cursors keyed by resolved data path and training mode. Switching between Chinese and English automatically resumes each corpus at its saved epoch and batch position; a new corpus starts at the beginning. Model weights, optimizer state, random streams, and cumulative steps remain global. End of corpus starts the next epoch. Legacy single-cursor checkpoints do not migrate data positions; a corpus without a saved entry starts at the beginning.

When intentionally restarting a corpus, or changing its files or batching configuration, explicitly reset only its traversal cursor while retaining other corpus cursors and global training state:

```bash
python main.py train --mode pretrain --load-checkpoint model/checkpoints/pretrain.pt --data-path data/new-pretrain --reset-data-state --train-steps 100
```

Cross-mode fine-tuning starts a new data cursor and optimizer state automatically. Legacy checkpoints without training state warn that the data cursor and random streams cannot be restored and restart data traversal from the beginning.

Gradient accumulation and reported training/validation losses are weighted by the number of non-ignored target tokens. Partial accumulation windows at the end of an epoch use the actual token count. Accumulation buffers CPU token/label batches for one window while processing model activations one microbatch at a time.

Overriding `paths.sft_single_turn_dir` or `paths.sft_multi_turn_dir` also changes the corresponding default `chinese`/`english` subdirectories. Explicit language-subdirectory settings take precedence.

## Tests

Run the model, data, checkpoint, and training regression tests (including multi-worker resume):

```bash
python -m unittest discover -s tests -v
```

## Inference

```bash
python main.py infer --checkpoint model/checkpoints/sft_single.pt --prompt "Explain RoPE."
```

Disable stopping at `<SEP>`:

```bash
python main.py infer --checkpoint model/checkpoints/sft_single.pt --prompt "Explain RoPE." --no-stop-at-sep
```

Use `--seed` for reproducible sampling:

```bash
python main.py infer --checkpoint model/checkpoints/sft_single.pt --prompt "Explain RoPE." --seed 42
```
# XTransformer
