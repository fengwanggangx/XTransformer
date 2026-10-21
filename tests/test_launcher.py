import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sentencepiece as spm
import torch
import yaml

from config import build_paths, load_config
from scripts import train as launcher
from main import wait_after_training


class TrainingLauncherTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="launcher with spaces ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = load_config()
        self.cfg.project_dir = self.root
        self.cfg.config_path = self.root / "custom config.yaml"
        self.cfg.paths = build_paths(self.root, {})
        Path(self.cfg.paths.pretrain_dir).mkdir(parents=True)
        self.chinese = Path(self.cfg.paths.pretrain_dir) / "wiki_zh.txt"
        self.english = Path(self.cfg.paths.pretrain_dir) / "wiki_en_balanced.txt"
        self.chinese.write_text("sample\n", encoding="utf-8")
        self.english.write_text("sample\n", encoding="utf-8")

    def args(self, *arguments):
        return launcher.build_parser().parse_args(list(arguments))

    def checkpoint(self, mode):
        path = Path(self.cfg.get_checkpoint_path(mode))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint placeholder")
        return path

    def test_flat_wiki_data_and_existing_language_directory_are_selected(self):
        self.assertEqual(launcher.training_data_path(self.cfg, self.args()), self.chinese)
        self.assertEqual(
            launcher.training_data_path(self.cfg, self.args("--language", "english")),
            self.english,
        )
        language_dir = Path(self.cfg.paths.pretrain_chinese_dir)
        language_dir.mkdir()
        self.assertEqual(launcher.training_data_path(self.cfg, self.args()), language_dir)

    def test_each_sft_stage_prefers_its_own_checkpoint_before_previous_stage(self):
        pretrained = self.checkpoint("pretrain")
        args = self.args("--mode", "sft_single")
        source, output = launcher.checkpoint_paths(self.cfg, args)
        self.assertEqual(source, pretrained)
        self.assertEqual(output, Path(self.cfg.paths.sft_single_checkpoint_path))
        single = self.checkpoint("sft_single")
        self.assertEqual(launcher.checkpoint_paths(self.cfg, args)[0], single)
        args = self.args("--mode", "sft_multi")
        self.assertEqual(launcher.checkpoint_paths(self.cfg, args)[0], single)
        multi = self.checkpoint("sft_multi")
        self.assertEqual(launcher.checkpoint_paths(self.cfg, args), (multi, multi))

    def test_sft_cannot_silently_start_from_random_weights(self):
        for mode in ("sft_single", "sft_multi"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "preceding stage"):
                launcher.checkpoint_paths(self.cfg, self.args("--mode", mode))

    def test_missing_explicit_checkpoint_does_not_fall_back_to_existing_weights(self):
        self.checkpoint("pretrain")
        args = self.args("--load-checkpoint", str(self.root / "missing.pt"))
        with self.assertRaisesRegex(ValueError, "checkpoint does not exist"):
            launcher.checkpoint_paths(self.cfg, args)

    def test_custom_output_and_paths_with_spaces_reach_core_cli_as_separate_arguments(self):
        self.checkpoint("pretrain")
        output = self.root / "new model.pt"
        data = self.root / "custom corpus.txt"
        args = self.args("--checkpoint", str(output), "--data-path", str(data), "--max-steps", "2000")
        command, selected_data, _, source, selected_output = launcher.build_command(self.cfg, args)
        self.assertEqual(selected_data, data)
        self.assertEqual(selected_output, output)
        self.assertIsNone(source)
        self.assertEqual(command[command.index("--checkpoint") + 1], str(output))
        self.assertEqual(command[command.index("--data-path") + 1], str(data))
        self.assertEqual(command[command.index("--max-steps") + 1], "2000")
        self.assertNotIn("--train-steps", command)
        self.assertNotIn("--load-checkpoint", command)

    def test_additional_steps_and_reset_flag_reach_core_cli(self):
        args = self.args("--train-steps", "25", "--reset-data-state")
        command = launcher.build_command(self.cfg, args)[0]
        self.assertEqual(command[command.index("--train-steps") + 1], "25")
        self.assertIn("--reset-data-state", command)

    def test_keep_alive_is_default_and_can_be_disabled(self):
        command = launcher.build_command(self.cfg, self.args())[0]
        self.assertIn("--keep-alive", command)
        command = launcher.build_command(self.cfg, self.args("--exit-on-complete"))[0]
        self.assertNotIn("--keep-alive", command)

    def test_completed_training_waits_for_explicit_exit(self):
        with patch("builtins.input", side_effect=["", "unknown", "exit"]) as prompt, \
                contextlib.redirect_stdout(io.StringIO()):
            wait_after_training(object())
        self.assertEqual(prompt.call_count, 3)

    def test_closed_stdin_waits_without_spinning(self):
        with patch("builtins.input", side_effect=EOFError), \
                patch("main.threading.Event") as event, \
                contextlib.redirect_stdout(io.StringIO()):
            wait_after_training(object())
        event.return_value.wait.assert_called_once_with()

    def test_no_eval_overrides_configured_validation_path(self):
        self.cfg.train.eval_data_path = "missing-configured-evaluation"
        args = self.args("--no-eval")
        command, _, evaluation, _, _ = launcher.build_command(self.cfg, args)
        self.assertIsNone(evaluation)
        self.assertEqual(command[command.index("--eval-data-path") + 1], "")

    def test_dry_run_checks_inputs_without_starting_training(self):
        with patch.object(launcher, "load_config", return_value=self.cfg), \
                patch.object(launcher, "check_inputs", return_value=32000) as checks, \
                patch.object(launcher.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--dry-run"]), 0)
        checks.assert_called_once()
        run.assert_not_called()

    def test_failed_setup_never_starts_training(self):
        with patch.object(launcher, "load_config", return_value=self.cfg), \
                patch.object(launcher, "check_inputs", side_effect=ValueError("invalid tokenizer")), \
                patch.object(launcher.subprocess, "run") as run, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main([]), 1)
        run.assert_not_called()

    def test_launcher_trains_and_resumes_from_another_working_directory(self):
        tokenizer_path = Path(self.cfg.paths.tokenizer_model_path)
        tokenizer_path.parent.mkdir(parents=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter([
                "sample alpha beta gamma delta epsilon explain a simple test answer yes or no"
            ] * 20),
            model_prefix=self.cfg.paths.tokenizer_model_prefix,
            vocab_size=64, model_type="bpe", num_threads=1, minloglevel=2,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<PAD>", unk_piece="<UNK>", bos_piece="<BOS>", eos_piece="<EOS>",
            control_symbols=["<SYSTEM>", "<USER>", "<ASSISTANT>", "<SEP>"],
        )
        raw = load_config().raw
        raw["runtime"]["device"] = "cpu"
        raw["model"].update({
            "vocab_size": 64, "d_embed": 16, "n_heads": 2,
            "n_layers": 1, "d_ff": 32, "max_seq_len": 16, "dropout": 0,
        })
        raw["train"].update({"warmup_steps": 0, "log_steps": 1, "save_steps": 1})
        raw["paths"] = {
            "data_dir": self.cfg.paths.data_dir,
            "model_dir": self.cfg.paths.model_dir,
        }
        self.cfg.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        output = self.root / "checkpoint with spaces.pt"
        command = [
            sys.executable, str(launcher.PROJECT_ROOT / "scripts/train.py"),
            "--config", str(self.cfg.config_path), "--checkpoint", str(output),
            "--allow-cpu", "--train-steps", "1", "--no-eval",
        ]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        for expected_step in (1, 2):
            result = subprocess.run(
                command, cwd=self.root, env=environment, capture_output=True,
                input="exit\n",
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(torch.load(output, weights_only=True)["step"], expected_step)


if __name__ == "__main__":
    unittest.main()
