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

Pretrain samples are trained on all tokens.

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

Only assistant messages are trained. System and user messages are context only.

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

## Tests

Run the core regression tests:

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
