# Training launcher

Run from an Anaconda Prompt or a Conda-enabled PowerShell terminal:

```powershell
.\scripts\train.cmd
```

This selects the `python_31208` Conda environment, validates CUDA, the existing tokenizer, and sample readability, then starts Chinese pretraining with `batch_size=1`, `grad_accum_steps=4`, `num_workers=0`, validation every 100 steps, and 10 validation batches. It uses `data/pretrain/wiki_zh.txt` and `data/eval/pretrain/wiki_zh_eval.txt` when using this project's flat Wiki data layout. Configured language directories and explicit `--data-path` overrides are supported as well.

Each invocation trains **1000 additional optimizer steps**. The current stage's checkpoint is resumed automatically. Missing pretraining checkpoints start a new model. On the first SFT run, single-turn SFT loads the pretraining checkpoint and multi-turn SFT loads the single-turn checkpoint. Later SFT invocations resume their own checkpoints. Stages are started separately so that you can decide when pretraining or fine-tuning has reached the desired quality.

After the final checkpoint is saved, the training process stays alive with the model loaded in memory (and GPU memory when using CUDA). No more optimizer steps run while waiting. Type `exit` / `quit`, or press Ctrl+C, to stop it. If standard input is closed, it waits for process termination instead. Use `--exit-on-complete` for unattended runs that should exit after saving. Direct `main.py train` calls can opt in with `--keep-alive`.

The launcher uses the existing tokenizer without retraining it, resolves relative paths against the project root, and starts training without an extra confirmation prompt. Training output streams to the terminal. Configuration and compatibility checks in the core trainer still apply. If default validation data is absent, validation is disabled; an explicitly supplied missing path fails the setup check. SFT validation is disabled unless configured or supplied explicitly.

## Examples

Check the environment, data, and generated command without training:

```powershell
.\scripts\train.cmd --dry-run
```

Continue pretraining for 2000 more steps:

```powershell
.\scripts\train.cmd --train-steps 2000
```

Resume an interrupted run up to a fixed total step target:

```powershell
.\scripts\train.cmd --max-steps 1000
```

Start or resume Chinese single-turn / multi-turn fine-tuning:

```powershell
.\scripts\train.cmd --mode sft_single
.\scripts\train.cmd --mode sft_multi
```

Select English data:

```powershell
.\scripts\train.cmd --language english
```

`--language all` traverses all files in the selected stage directory in sorted order; it does not mix or shuffle languages.

Override data and checkpoint paths, including when intentionally changing batching on resume:

```powershell
.\scripts\train.cmd --data-path "data/custom corpus" --eval-data-path data/custom-eval --checkpoint model/checkpoints/custom.pt --train-steps 1000
.\scripts\train.cmd --batch-size 2 --reset-data-state
```

Use `--no-eval` to disable validation. Use `--config configs/custom.yaml` for a different model/configuration, `--load-checkpoint` for a custom source checkpoint, or `--seed` to override the seed. The core trainer requires `--reset-data-state` when changing training data, batching, worker count, accumulation count, or seed on a same-mode resume.

When the correct Python environment is already active, the Python launcher can also be called directly:

```powershell
python scripts/train.py --dry-run
python scripts/train.py
python scripts/train.py --help
```

For a different Conda environment, invoke it explicitly:

```powershell
conda run --no-capture-output -n another_environment python scripts/train.py
```

`--allow-cpu` permits a configuration selecting CPU, for small development runs. Relative paths work from any current directory when using the launcher's absolute path.

Training launches append the invocation, resolved command, configuration, starting progress, validation loss, checkpoint saves, and completion summary to `logs/YYYYMMDD.log` under the project root. Entries include local timestamps; repeated runs append to the same daily UTF-8 file. Frequent training loss/speed lines remain console-only. Logging takes effect on the next launch.
