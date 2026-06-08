import copy
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from config import load_config
from data import TDataSet
from inference import inference_session_cache_key, normalize_stop_token_ids
from model import make_model
from reproducibility import seed_everything
from tokenizer import encode_content


def small_cfg(max_seq_len=8):
    cfg = load_config()
    cfg.model.vocab_size = 32
    cfg.model.n_layers = 2
    cfg.model.d_model = 16
    cfg.model.d_ff = 32
    cfg.model.n_heads = 4
    cfg.model.dropout = 0.0
    cfg.model.max_seq_len = max_seq_len
    return cfg


def small_model(cfg):
    model = make_model(cfg)
    model.eval()
    return model


class DummyTokenizer:
    def encode(self, text, out_type=int):
        return [10 + index for index, _ in enumerate(str(text))]


class SpecialTokenTokenizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, text, out_type=int):
        pieces = str(text).split()
        token_ids = []
        for piece in pieces:
            token_ids.append(self.cfg.tokens.all_token_ids.get(piece, 10))
        return token_ids


class ConfigContractTest(unittest.TestCase):
    def test_relative_path_overrides_are_resolved_from_project_dir(self):
        raw = copy.deepcopy(load_config().raw)
        raw["paths"].update(
            {
                "checkpoint_dir": "runs/checkpoints",
                "pretrain_checkpoint_path": "runs/pretrain.pt",
                "tokenizer_model_prefix": "artifacts/tokenizer/custom",
                "tokenizer_model_path": "custom/tokenizer.model",
                "pretrain_dir": "datasets/pretrain",
                "sft_single_turn_chinese_dir": "datasets/sft-cn",
                "eval_dir": "datasets/eval",
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            cfg = load_config(config_path)

        project_dir = cfg.project_dir
        self.assertEqual(Path(cfg.paths.checkpoint_dir), project_dir / "runs/checkpoints")
        self.assertEqual(Path(cfg.paths.pretrain_checkpoint_path), project_dir / "runs/pretrain.pt")
        self.assertEqual(Path(cfg.paths.sft_single_checkpoint_path), project_dir / "runs/checkpoints/sft_single.pt")
        self.assertEqual(Path(cfg.paths.tokenizer_model_prefix), project_dir / "artifacts/tokenizer/custom")
        self.assertEqual(Path(cfg.paths.tokenizer_model_path), project_dir / "custom/tokenizer.model")
        self.assertEqual(Path(cfg.paths.tokenizer_vocab_path), project_dir / "artifacts/tokenizer/custom.vocab")
        self.assertEqual(Path(cfg.paths.pretrain_dir), project_dir / "datasets/pretrain")
        self.assertEqual(Path(cfg.paths.pretrain_chinese_dir), project_dir / "datasets/pretrain/chinese")
        self.assertEqual(Path(cfg.paths.sft_single_turn_chinese_dir), project_dir / "datasets/sft-cn")
        self.assertEqual(Path(cfg.paths.eval_dir), project_dir / "datasets/eval")


class ModelCoreTest(unittest.TestCase):
    def test_seed_everything_makes_model_init_reproducible(self):
        cfg = small_cfg()
        seed_everything(123)
        model_a = small_model(cfg)
        weight_a = model_a.get_input_embeddings().weight.detach().clone()

        cfg = small_cfg()
        seed_everything(123)
        model_b = small_model(cfg)
        weight_b = model_b.get_input_embeddings().weight.detach().clone()

        self.assertTrue(torch.allclose(weight_a, weight_b))

    def test_kv_cache_matches_full_forward(self):
        cfg = small_cfg()
        model = small_model(cfg)
        input_ids = torch.tensor([[2, 8, 9, 10, 11]], device=cfg.device)

        with torch.no_grad():
            full_logits = model(input_ids)
            _, past_key_values = model(input_ids[:, :-1], use_cache=True)
            cached_logits, present_key_values = model(
                input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

        self.assertTrue(
            torch.allclose(
                full_logits[:, -1],
                cached_logits[:, -1],
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertEqual(len(present_key_values), 2)
        self.assertEqual(len(present_key_values[0]), 3)

    def test_forward_rejects_sequences_longer_than_max_seq_len(self):
        cfg = small_cfg(max_seq_len=4)
        model = small_model(cfg)
        input_ids = torch.tensor([[2, 8, 9, 10, 11]], device=cfg.device)

        with self.assertRaisesRegex(ValueError, "sequence length exceeds max_seq_len"):
            model(input_ids)

        with torch.no_grad():
            _, past_key_values = model(input_ids[:, :4], use_cache=True)

        with self.assertRaisesRegex(ValueError, "sequence length exceeds max_seq_len"):
            model(input_ids[:, 4:], past_key_values=past_key_values, use_cache=True)

    def test_cache_keeps_past_padding_mask(self):
        cfg = small_cfg()
        model = small_model(cfg)
        input_ids = torch.tensor(
            [
                [cfg.tokens.bos_idx, 8, cfg.tokens.padding_idx],
                [cfg.tokens.bos_idx, 8, 9],
            ],
            device=cfg.device,
        )
        next_ids = torch.tensor([[10], [10]], device=cfg.device)

        with torch.no_grad():
            _, past_key_values = model(input_ids, use_cache=True)
            _, present_key_values = model(
                next_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        self.assertEqual(
            past_key_values[0][2].tolist(),
            [[True, True, False], [True, True, True]],
        )
        self.assertEqual(
            present_key_values[0][2].tolist(),
            [[True, True, False, True], [True, True, True, True]],
        )

    def test_weight_tying_and_padding_embedding_zero(self):
        cfg = small_cfg()
        model = small_model(cfg)
        input_embeddings = model.get_input_embeddings().weight
        output_projection = model.generator.proj.weight

        self.assertEqual(input_embeddings.data_ptr(), output_projection.data_ptr())
        self.assertFalse(model.generator.proj.bias)
        self.assertTrue(
            torch.allclose(
                input_embeddings[cfg.tokens.padding_idx],
                torch.zeros_like(input_embeddings[cfg.tokens.padding_idx]),
            )
        )


class InferenceContractTest(unittest.TestCase):
    def test_session_cache_key_changes_when_config_values_change(self):
        cfg = small_cfg()
        checkpoint_path = "missing-checkpoint.pt"
        cache_key = inference_session_cache_key(cfg, checkpoint_path)

        cfg.inference.temperature = cfg.inference.temperature + 0.1

        self.assertNotEqual(cache_key, inference_session_cache_key(cfg, checkpoint_path))

    def test_session_cache_key_changes_when_tokenizer_file_changes(self):
        cfg = small_cfg()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            tokenizer_path = tmp_dir / "tokenizer.model"
            tokenizer_path.write_text("v1", encoding="utf-8")
            cfg.paths.tokenizer_model_path = str(tokenizer_path)
            checkpoint_path = tmp_dir / "missing-checkpoint.pt"

            cache_key = inference_session_cache_key(cfg, checkpoint_path)
            tokenizer_path.write_text("version two", encoding="utf-8")

            self.assertNotEqual(
                cache_key,
                inference_session_cache_key(cfg, checkpoint_path),
            )

    def test_stop_token_ids_are_configurable(self):
        cfg = small_cfg()
        default_stop_token_ids = normalize_stop_token_ids(cfg)
        self.assertIn(cfg.tokens.eos_idx, default_stop_token_ids)
        self.assertIn(cfg.tokens.sep_idx, default_stop_token_ids)
        self.assertIn(cfg.tokens.padding_idx, default_stop_token_ids)

        no_sep_stop_token_ids = normalize_stop_token_ids(cfg, stop_at_sep=False)
        self.assertIn(cfg.tokens.eos_idx, no_sep_stop_token_ids)
        self.assertNotIn(cfg.tokens.sep_idx, no_sep_stop_token_ids)
        self.assertIn(cfg.tokens.padding_idx, no_sep_stop_token_ids)

        custom_stop_token_ids = normalize_stop_token_ids(cfg, stop_token_ids=(cfg.tokens.eos_idx,))
        self.assertEqual(custom_stop_token_ids, frozenset({cfg.tokens.eos_idx}))


class DataContractTest(unittest.TestCase):
    def test_encode_content_masks_injected_structure_tokens(self):
        cfg = small_cfg()
        tokenizer = SpecialTokenTokenizer(cfg)

        token_ids = encode_content(
            cfg,
            tokenizer,
            "<SYSTEM> hello <ASSISTANT> <SEP> <EOS> <PAD> <UNK>",
        )

        self.assertEqual(
            token_ids,
            [
                cfg.tokens.unk_idx,
                10,
                cfg.tokens.unk_idx,
                cfg.tokens.unk_idx,
                cfg.tokens.unk_idx,
                cfg.tokens.unk_idx,
                cfg.tokens.unk_idx,
            ],
        )

    def test_sft_single_masks_only_assistant_targets(self):
        cfg = small_cfg()
        dataset = TDataSet(
            cfg,
            Path("unused.jsonl"),
            mode="sft_single",
            max_seq_len=32,
            ignore_index=cfg.train.ignore_index,
        )
        dataset.tokenizer = DummyTokenizer()

        sample = dataset.build_single_turn_samples(
            {
                "instruction": "u",
                "response": "ab",
            }
        )

        self.assertIsNotNone(sample)
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        trainable_positions = [
            index for index, label in enumerate(labels)
            if label != cfg.train.ignore_index
        ]

        assistant_index = input_ids.index(cfg.tokens.assistant_idx)
        self.assertEqual(trainable_positions[0], assistant_index)
        self.assertEqual(
            labels[trainable_positions[0]:],
            [10, 11, cfg.tokens.sep_idx, cfg.tokens.eos_idx],
        )
        for index in range(assistant_index):
            self.assertEqual(labels[index], cfg.train.ignore_index)


if __name__ == "__main__":
    unittest.main()
