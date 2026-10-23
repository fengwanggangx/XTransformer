import contextlib
import copy
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sentencepiece as spm
import torch

import inference
import train
from checkpoint import validate_checkpoint_config
from config import load_config
from data import TDataSet
from model import make_model
from reproducibility import capture_rng_state, restore_rng_state
from tokenizer import tokenizer_load


class InterruptedTraining(Exception):
    pass


class UnequalTokenDataset(TDataSet):
    def __iter__(self):
        yield {"input_ids": [10] * 3, "labels": [0, -100, -100]}
        yield {"input_ids": [10] * 9, "labels": [1] * 9}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, input_ids):
        return torch.stack([
            self.weight.expand(input_ids.shape),
            torch.zeros_like(input_ids, dtype=torch.float32),
        ], dim=-1)


class TrainingRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer_bytes = cls.make_tokenizer(
            "alpha beta gamma delta epsilon explain a simple test answer yes or no"
        )
        cls.other_tokenizer_bytes = cls.make_tokenizer(
            "red green blue orange violet yellow black white circle square triangle"
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_tokenizer(sentence):
        output = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter([sentence] * 20), model_writer=output,
            model_type="bpe", vocab_size=64, num_threads=1, minloglevel=2,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<PAD>", unk_piece="<UNK>", bos_piece="<BOS>", eos_piece="<EOS>",
            control_symbols=["<SYSTEM>", "<USER>", "<ASSISTANT>", "<SEP>"],
        )
        return output.getvalue()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = load_config()
        self.cfg.runtime.device = "cpu"
        self.cfg.model.vocab_size = 64
        self.cfg.model.d_embed = 16
        self.cfg.model.n_heads = 2
        self.cfg.model.n_layers = 1
        self.cfg.model.d_ff = 32
        self.cfg.model.max_seq_len = 32
        self.cfg.model.dropout = 0.2
        self.cfg.train.warmup_steps = 0
        self.cfg.train.log_steps = 100
        self.cfg.train.save_steps = 1
        self.cfg.paths.tokenizer_model_path = str(self.root / "tokenizer.model")
        Path(self.cfg.paths.tokenizer_model_path).write_bytes(self.tokenizer_bytes)
        self.data_path = self.root / "train.txt"
        self.data_path.write_text(
            "alpha\nbeta gamma\ndelta epsilon\nexplain a simple test\nanswer yes or no\n",
            encoding="utf-8",
        )

    def run_training(self, path, **overrides):
        options = {
            "data_path": str(self.data_path), "mode": "pretrain",
            "checkpoint_path": str(path), "save_checkpoint_path": str(path),
            "resume": False, "max_steps": 4, "batch_size": 1,
            "grad_accum_steps": 1, "num_workers": 0,
        }
        options.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return train.train(self.cfg, **options)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def save_initial_checkpoint(self, path):
        model = make_model(self.cfg)
        optimizer = torch.optim.AdamW(model.parameters())
        train.save_checkpoint(self.cfg, path, model, optimizer, None, 10, "pretrain")
        return model

    def test_corpus_switch_preserves_cursors_and_reset_is_local(self):
        path = self.root / "switch.pt"
        other = self.root / "english.txt"
        other.write_text("alpha beta\ngamma delta\nepsilon\n", encoding="utf-8")
        self.run_training(path, max_steps=2)
        first = torch.load(path, weights_only=True)["training_state"]
        first_key = first["active_data_key"]
        self.run_training(path, resume=True, data_path=str(other), max_steps=3)
        second = torch.load(path, weights_only=True)["training_state"]
        second_key = second["active_data_key"]
        self.assert_nested_equal(second["data_states"][first_key], first["data_states"][first_key])
        self.run_training(path, resume=True, max_steps=4)
        checkpoint = torch.load(path, weights_only=True)
        state = checkpoint["training_state"]
        self.assertEqual(checkpoint["step"], 4)
        self.assertEqual(state["batches_in_epoch"], 3)
        self.assert_nested_equal(state["data_states"][second_key], second["data_states"][second_key])
        self.run_training(path, resume=True, reset_data_state=True, max_steps=5)
        state = torch.load(path, weights_only=True)["training_state"]
        self.assertEqual(state["batches_in_epoch"], 1)
        self.assert_nested_equal(state["data_states"][second_key], second["data_states"][second_key])

    def test_legacy_cursor_is_not_migrated(self):
        path = self.root / "legacy.pt"
        self.run_training(path, max_steps=2)
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["training_state"].pop("data_states")
        checkpoint["training_state"].pop("active_data_key")
        torch.save(checkpoint, path)
        self.run_training(path, resume=True, max_steps=3)
        checkpoint = torch.load(path, weights_only=True)
        state = checkpoint["training_state"]
        self.assertEqual(checkpoint["step"], 3)
        self.assertEqual(state["batches_in_epoch"], 1)
        self.assertEqual(len(state["data_states"]), 1)

    def test_resumed_corpus_rolls_into_next_epoch(self):
        path = self.root / "epoch.pt"
        self.run_training(path, max_steps=5)
        self.run_training(path, resume=True, max_steps=6)
        state = torch.load(path, weights_only=True)["training_state"]
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["batches_in_epoch"], 1)

    def test_changed_corpus_metadata_requires_explicit_reset(self):
        path = self.root / "changed.pt"
        self.run_training(path, max_steps=1)
        with self.data_path.open("a", encoding="utf-8") as stream:
            stream.write("alpha beta\n")
        with self.assertRaisesRegex(ValueError, "reset-data-state"):
            self.run_training(path, resume=True, max_steps=2)

    def test_failed_serialization_preserves_previous_checkpoint(self):
        path = self.root / "saved.pt"
        model = self.save_initial_checkpoint(path)
        original = path.read_bytes()
        optimizer = torch.optim.AdamW(model.parameters())
        with patch("torch.serialization._save", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                train.save_checkpoint(self.cfg, path, model, optimizer, None, 20, "pretrain")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(torch.load(path, weights_only=True)["step"], 10)
        self.assertEqual(list(self.root.glob(".saved.pt.*.tmp")), [])

    def test_failed_replace_preserves_previous_checkpoint(self):
        path = self.root / "saved.pt"
        model = self.save_initial_checkpoint(path)
        original = path.read_bytes()
        optimizer = torch.optim.AdamW(model.parameters())
        with patch("train.os.replace", side_effect=PermissionError("locked destination")):
            with self.assertRaises(PermissionError):
                train.save_checkpoint(self.cfg, path, model, optimizer, None, 20, "pretrain")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".saved.pt.*.tmp")), [])

    def test_accumulation_matches_large_batch_with_unequal_target_counts(self):
        self.cfg.train.weight_decay = 0
        results = []
        with patch.object(train, "TDataSet", UnequalTokenDataset), \
                patch.object(train, "make_model", side_effect=lambda cfg: ToyModel()), \
                patch.object(train, "zero_padding_embedding"):
            # The last case exercises a partial accumulation window.
            for batch_size, accumulation in ((2, 1), (1, 2), (1, 3)):
                model = self.run_training(
                    self.root / f"batch-{batch_size}-{accumulation}.pt",
                    max_steps=1, batch_size=batch_size, grad_accum_steps=accumulation,
                )
                results.append(model.weight.detach().clone())
        self.assertLess(results[0].item(), -1.0)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=1e-7)

    def test_explicit_missing_checkpoint_fails_before_model_initialization(self):
        with patch.object(train, "make_model") as factory:
            with self.assertRaises(FileNotFoundError):
                self.run_training(
                    self.root / "out.pt", resume=True,
                    load_checkpoint_path=str(self.root / "missing.pt"),
                )
            factory.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            train.load_checkpoint(self.cfg, self.root / "missing.pt", make_model(self.cfg))

    def test_missing_default_checkpoint_allows_new_training(self):
        path = self.root / "new.pt"
        self.run_training(path, resume=True, max_steps=1)
        self.assertEqual(torch.load(path, weights_only=True)["step"], 1)

    def test_explicit_load_cannot_be_silently_disabled(self):
        with self.assertRaisesRegex(ValueError, "no-resume"):
            self.run_training(self.root / "out.pt", load_checkpoint_path="ignored.pt")

    def test_checkpoint_rejects_changed_tokenizer_with_same_vocab_and_special_ids(self):
        path = self.root / "saved.pt"
        model = self.save_initial_checkpoint(path)
        Path(self.cfg.paths.tokenizer_model_path).write_bytes(self.other_tokenizer_bytes)
        # Both vocabularies satisfy the project's existing special-token checks.
        self.assertEqual(tokenizer_load(self.cfg).GetPieceSize(), 64)
        with self.assertRaisesRegex(RuntimeError, "tokenizer fingerprint mismatch"):
            train.load_checkpoint(self.cfg, path, model)
        with self.assertRaisesRegex(RuntimeError, "tokenizer fingerprint mismatch"):
            inference.load_model(self.cfg, path)

    def test_checkpoint_rejects_changed_role_ids_and_token_strings(self):
        path = self.root / "saved.pt"
        self.save_initial_checkpoint(path)
        checkpoint = torch.load(path, weights_only=True)
        for attribute, value in (("user_idx", 6), ("assistant_token", "<BOT>")):
            with self.subTest(attribute=attribute):
                cfg = copy.deepcopy(self.cfg)
                setattr(cfg.tokens, attribute, value)
                with self.assertRaisesRegex(RuntimeError, "token configuration mismatch"):
                    validate_checkpoint_config(checkpoint, cfg)

    def test_legacy_checkpoint_warns_about_unrecoverable_state(self):
        path = self.root / "legacy.pt"
        self.run_training(path, max_steps=1)
        checkpoint = torch.load(path, weights_only=True)
        checkpoint.pop("tokenizer_sha256")
        checkpoint.pop("training_state")
        torch.save(checkpoint, path)
        with self.assertWarnsRegex(RuntimeWarning, "no tokenizer fingerprint"):
            validate_checkpoint_config(checkpoint, self.cfg)
        # Retain the fingerprint here to isolate the missing-cursor warning.
        from checkpoint import tokenizer_fingerprint
        checkpoint["tokenizer_sha256"] = tokenizer_fingerprint(self.cfg)
        torch.save(checkpoint, path)
        with self.assertWarnsRegex(RuntimeWarning, "no training_state"):
            step, _ = train.load_checkpoint(
                self.cfg, path, make_model(self.cfg), training_state={},
            )
        self.assertEqual(step, 1)

    def verify_resume(self, num_workers, interrupt_step, **options):
        full_path = self.root / "full.pt"
        resumed_path = self.root / "resumed.pt"
        options = {"max_steps": 4, "num_workers": num_workers, **options}
        self.run_training(full_path, **options)
        original_save = train.save_checkpoint

        def interrupt_after_save(*args, **kwargs):
            original_save(*args, **kwargs)
            if args[5] == interrupt_step:
                raise InterruptedTraining()

        with patch.object(train, "save_checkpoint", side_effect=interrupt_after_save):
            with self.assertRaises(InterruptedTraining):
                self.run_training(resumed_path, **options)
        self.run_training(resumed_path, resume=True, **options)
        full = torch.load(full_path, weights_only=True)
        resumed = torch.load(resumed_path, weights_only=True)
        for key in (
            "step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
            "scaler_state_dict", "training_state",
        ):
            self.assert_nested_equal(resumed[key], full[key])

    def test_resume_matches_uninterrupted_training_with_dropout_and_evaluation(self):
        self.verify_resume(
            0, 2, eval_data_path=str(self.data_path), eval_steps=1, eval_batches=2,
        )

    def test_resume_at_epoch_boundary_with_partial_accumulation(self):
        self.verify_resume(0, 2, batch_size=2, grad_accum_steps=2)

    def test_resume_with_multiple_workers_preserves_batch_order(self):
        self.verify_resume(2, 1, batch_size=2, grad_accum_steps=2)

    def test_changed_batching_requires_explicit_cursor_reset(self):
        path = self.root / "saved.pt"
        self.run_training(path, max_steps=1)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "reset-data-state"):
            self.run_training(path, resume=True, max_steps=2, batch_size=2)
        self.assertEqual(path.read_bytes(), original)
        self.run_training(path, resume=True, max_steps=2, batch_size=2, reset_data_state=True)
        self.assertEqual(torch.load(path, weights_only=True)["step"], 2)

    def test_changed_data_is_not_silently_skipped_on_resume(self):
        path = self.root / "saved.pt"
        self.run_training(path, max_steps=1)
        with self.data_path.open("a", encoding="utf-8") as stream:
            stream.write("new data\n")
        with self.assertRaisesRegex(ValueError, "training data"):
            self.run_training(path, resume=True, max_steps=2)

    def test_cross_mode_finetuning_starts_a_new_data_cursor(self):
        pretrained = self.root / "pretrain.pt"
        self.run_training(pretrained, max_steps=1)
        sft_data = self.root / "sft.jsonl"
        sft_data.write_text(json.dumps({"instruction": "explain", "response": "yes"}) + "\n")
        finetuned = self.root / "sft.pt"
        self.run_training(
            finetuned, mode="sft_single", data_path=str(sft_data),
            resume=True, load_checkpoint_path=str(pretrained), max_steps=1,
        )
        checkpoint = torch.load(finetuned, weights_only=True)
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(checkpoint["training_state"]["batches_in_epoch"], 1)
        result = inference.load_model(self.cfg, finetuned)
        self.assertFalse(result.training)

    def test_rng_state_roundtrip_restores_python_and_torch_sequences(self):
        state = capture_rng_state()
        expected_python = random.random()
        expected_torch = torch.rand(5)
        restore_rng_state(state)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(5), expected_torch))


class EvaluationRegressionTest(unittest.TestCase):
    def test_loss_is_invariant_to_batch_partition_and_ignored_targets(self):
        cfg = load_config()
        cfg.runtime.device = "cpu"

        class FixedModel(torch.nn.Module):
            def forward(self, input_ids):
                probability = torch.where(input_ids == 10, 0.9, 0.1)
                return torch.stack([probability.log(), (1 - probability).log()], dim=-1)

        model = FixedModel()
        separated = [
            {"input_ids": torch.tensor([[10, 10]]), "labels": torch.tensor([[0, -100]])},
            {"input_ids": torch.full((1, 9), 11), "labels": torch.zeros((1, 9), dtype=torch.long)},
        ]
        combined = [{key: torch.cat([item[key] for item in separated], dim=1) for key in separated[0]}]
        expected = (-math.log(0.9) - 9 * math.log(0.1)) / 10
        for batches in (separated, combined):
            self.assertAlmostEqual(train.evaluate(cfg, model, batches), expected, places=6)
            self.assertTrue(model.training)
        ignored = [{"input_ids": torch.tensor([[10]]), "labels": torch.tensor([[-100]])}]
        self.assertIsNone(train.evaluate(cfg, model, ignored))

    def test_evaluation_restores_training_mode_after_an_exception(self):
        cfg = load_config()
        model = ToyModel()
        with self.assertRaises(KeyError):
            train.evaluate(cfg, model, [{}])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
