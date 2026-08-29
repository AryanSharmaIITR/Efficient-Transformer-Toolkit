"""Tests for tokenizer, dataset, and collator."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.data.collator import DataCollator
from src.data.dataset import TextDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Minimal tokenizer stub for testing without network / HF dependency."""

    def __init__(self, vocab_size=100, pad_token_id=0, eos_token_id=1, bos_token_id=2):
        self._vocab_size = vocab_size
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._bos_token_id = bos_token_id

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def bos_token_id(self):
        return self._bos_token_id

    def encode(self, text, truncation=True, max_length=None, add_special_tokens=True):
        # Simple: hash-based deterministic token ids
        tokens = [abs(hash(c)) % self._vocab_size for c in text]
        if add_special_tokens:
            tokens = [self._bos_token_id] + tokens + [self._eos_token_id]
        if truncation and max_length is not None and len(tokens) > max_length:
            # Keep bos + first (max_length-1) tokens, then eos
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        skip = {self._pad_token_id, self._eos_token_id, self._bos_token_id} if skip_special_tokens else set()
        return "".join(chr(t % 256) for t in token_ids if t not in skip)


# ---------------------------------------------------------------------------
# DataCollator
# ---------------------------------------------------------------------------

class TestDataCollator:
    def _collator(self, pad_to_multiple_of=None, max_length=None):
        tok = FakeTokenizer()
        return DataCollator(
            tok,
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
        )

    def test_causal_collation(self):
        collator = self._collator()
        batch = [
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
            {"input_ids": [40, 50], "attention_mask": [1, 1], "labels": [40, 50]},
        ]
        result = collator(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_causal_labels_shifted(self):
        collator = self._collator()
        batch = [
            {"input_ids": [10, 20, 30, 40], "attention_mask": [1, 1, 1, 1], "labels": [10, 20, 30, 40]},
        ]
        result = collator(batch)
        # labels shifted: drop first, append -100
        expected = torch.tensor([[20, 30, 40, -100]])
        torch.testing.assert_close(result["labels"], expected)

    def test_padding_values(self):
        collator = self._collator()
        batch = [
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
            {"input_ids": [40], "attention_mask": [1], "labels": [40]},
        ]
        result = collator(batch)
        # Second sequence padded with 0
        assert result["input_ids"][1, 1].item() == 0
        assert result["input_ids"][1, 2].item() == 0
        # Attention mask should be 0 for padded positions
        assert result["attention_mask"][1, 1].item() == 0

    def test_empty_batch(self):
        collator = self._collator()
        result = collator([])
        assert result == {}

    def test_pad_to_multiple_of(self):
        collator = self._collator(pad_to_multiple_of=8)
        batch = [
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [10, 20, 30]},
        ]
        result = collator(batch)
        # 3 tokens + 1 shift = 3, pad to multiple of 8 -> 8
        assert result["input_ids"].shape[1] == 8

    def test_max_length(self):
        collator = self._collator(max_length=2)
        batch = [
            {"input_ids": [10, 20, 30, 40], "attention_mask": [1, 1, 1, 1], "labels": [10, 20, 30, 40]},
        ]
        result = collator(batch)
        assert result["input_ids"].shape[1] == 2

    def test_encoder_decoder_collation(self):
        collator = self._collator()
        batch = [
            {
                "input_ids": [10, 20],
                "attention_mask": [1, 1],
                "decoder_input_ids": [30, 40, 50],
                "decoder_attention_mask": [1, 1, 1],
                "labels": [30, 40, 50],
            },
            {
                "input_ids": [60, 70, 80],
                "attention_mask": [1, 1, 1],
                "decoder_input_ids": [90],
                "decoder_attention_mask": [1],
                "labels": [90],
            },
        ]
        result = collator(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["decoder_input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_no_labels_batch(self):
        collator = self._collator()
        batch = [
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]},
        ]
        result = collator(batch)
        assert "labels" not in result
        assert result["input_ids"].shape == (1, 3)


# ---------------------------------------------------------------------------
# Static methods
# ---------------------------------------------------------------------------

class TestCollatorStatics:
    def test_make_causal_mask(self):
        mask = DataCollator.make_causal_mask(seq_len=4, batch_size=2)
        assert mask.shape == (2, 1, 4, 4)
        # Lower triangular
        assert mask[0, 0, 0, 0] == 1.0
        assert mask[0, 0, 0, 1] == 0.0  # upper tri
        assert mask[0, 0, 1, 0] == 1.0
        assert mask[0, 0, 1, 1] == 1.0

    def test_make_padding_mask(self):
        attn_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        pad_mask = DataCollator.make_padding_mask(attn_mask)
        assert pad_mask.shape == (2, 1, 1, 3)
        assert pad_mask[0, 0, 0, 2] == 0

    def test_shift_labels_for_causal(self):
        labels = torch.tensor([[10, 20, 30, 40]])
        shifted = DataCollator._shift_labels_for_causal(labels)
        expected = torch.tensor([[20, 30, 40, -100]])
        torch.testing.assert_close(shifted, expected)

    def test_shift_labels_batch(self):
        labels = torch.tensor([[10, 20], [30, 40]])
        shifted = DataCollator._shift_labels_for_causal(labels)
        expected = torch.tensor([[20, -100], [40, -100]])
        torch.testing.assert_close(shifted, expected)


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_source_texts_mode(self):
        tok = FakeTokenizer()
        texts = ["hello world", "foo bar baz"]
        ds = TextDataset(tok, source_texts=texts, max_seq_len=64)
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_source_texts_encoder_decoder(self):
        tok = FakeTokenizer()
        src = ["hello world", "foo bar"]
        tgt = ["world hello", "bar foo"]
        ds = TextDataset(tok, source_texts=src, target_texts=tgt, max_seq_len=64)
        item = ds[0]
        assert "decoder_input_ids" in item
        assert "decoder_attention_mask" in item
        assert "labels" in item

    def test_filepaths_mode(self):
        tok = FakeTokenizer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("First document.\n\nSecond document.")
            f.flush()
            tmp_path = f.name

        ds = TextDataset(tok, filepaths=tmp_path, max_seq_len=64)
        assert ds.num_documents == 2
        assert len(ds) == 2

    def test_max_seq_len_truncation(self):
        tok = FakeTokenizer(vocab_size=100)
        long_text = "x" * 200
        ds = TextDataset(tok, source_texts=[long_text], max_seq_len=10)
        item = ds[0]
        assert len(item["input_ids"]) == 10

    def test_no_source_no_filepath_raises(self):
        tok = FakeTokenizer()
        with pytest.raises(ValueError, match="Provide either"):
            TextDataset(tok)

    def test_get_document(self):
        tok = FakeTokenizer()
        texts = ["first", "second"]
        ds = TextDataset(tok, source_texts=texts, max_seq_len=64)
        assert ds.get_document(0) == "first"
        assert ds.get_document(1) == "second"

    def test_num_documents(self):
        tok = FakeTokenizer()
        ds = TextDataset(tok, source_texts=["a", "b", "c"], max_seq_len=64)
        assert ds.num_documents == 3
