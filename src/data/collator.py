from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import Tokenizer


class DataCollator:
    """Collates variable-length token-id sequences into padded batches.

    Supports three modes:

    1. **Causal LM** (decoder-only):
       ``labels`` are the *input_ids* shifted right by one position.
       The first token in each sequence is dropped from the labels and
       the last position is filled with ``-100`` (ignored by cross-
       entropy).

    2. **Masked LM** (encoder-only, e.g. BERT):
       ``labels`` are a copy of *input_ids* with some tokens masked
       (handled externally).

    3. **Encoder-decoder** (seq2seq):
       Pads encoder *input_ids* + *attention_mask* **and** decoder
       *decoder_input_ids* + *decoder_attention_mask* independently.
       ``labels`` are decoder target ids with pad positions set to
       ``-100``.

    Args:
        tokenizer: A :class:`Tokenizer` used only for its
            ``pad_token_id``.
        pad_to_multiple_of: Round the final sequence length up to this
            value (useful for hardware efficiency).
        max_length: Hard cap on sequence length.  ``None`` means no
            cap.
        label_pad_id: Value used to pad ``labels`` (default ``-100``,
            which PyTorch cross-entropy ignores).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        pad_to_multiple_of: int | None = None,
        max_length: int | None = None,
        label_pad_id: int = -100,
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.label_pad_id = label_pad_id

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Collate a list of examples into a padded batch.

        Detects the mode (causal-LM, encoder-decoder, or encoder-only)
        from the keys present in the first example and dispatches
        accordingly.

        Args:
            batch: List of dicts returned by :meth:`TextDataset.__getitem__`.

        Returns:
            Dict of tensors ready for model consumption.
        """
        if not batch:
            return {}

        has_decoder = "decoder_input_ids" in batch[0]
        has_labels = "labels" in batch[0]

        if has_decoder:
            return self._collate_encoder_decoder(batch)
        return self._collate_causal_or_mlm(batch, has_labels)

    # ------------------------------------------------------------------
    # Causal / MLM
    # ------------------------------------------------------------------

    def _collate_causal_or_mlm(
        self,
        batch: list[dict[str, list[int]]],
        has_labels: bool,
    ) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]

        input_ids = self._pad_sequence(input_ids)
        attention_mask = self._pad_sequence(attention_mask, pad_value=0)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if has_labels:
            labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
            labels = self._pad_sequence(labels, pad_value=self.label_pad_id)
            # For causal LM, shift labels right: drop first token, append -100
            labels = self._shift_labels_for_causal(labels)
            result["labels"] = labels

        return result

    # ------------------------------------------------------------------
    # Encoder-decoder
    # ------------------------------------------------------------------

    def _collate_encoder_decoder(
        self,
        batch: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        # Encoder side
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        input_ids = self._pad_sequence(input_ids)
        attention_mask = self._pad_sequence(attention_mask, pad_value=0)

        # Decoder side
        dec_ids = [torch.tensor(ex["decoder_input_ids"], dtype=torch.long) for ex in batch]
        dec_mask = [torch.tensor(ex["decoder_attention_mask"], dtype=torch.long) for ex in batch]
        dec_ids = self._pad_sequence(dec_ids)
        dec_mask = self._pad_sequence(dec_mask, pad_value=0)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": dec_ids,
            "decoder_attention_mask": dec_mask,
        }

        if "labels" in batch[0]:
            labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
            labels = self._pad_sequence(labels, pad_value=self.label_pad_id)
            result["labels"] = labels

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_sequence(
        self,
        sequences: list[torch.Tensor],
        *,
        pad_value: int | None = None,
    ) -> torch.Tensor:
        """Pad a list of 1-D tensors to equal length.

        Optionally clips to ``max_length`` and rounds up to
        ``pad_to_multiple_of``.
        """
        if not sequences:
            return torch.empty(0, 0, dtype=torch.long)

        max_len = max(s.size(0) for s in sequences)

        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        if pad_value is None:
            pad_value = self.pad_token_id

        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_value,
        )

        if padded.size(1) < max_len:
            pad_amount = max_len - padded.size(1)
            pad_tensor = torch.full(
                (padded.size(0), pad_amount), pad_value, dtype=padded.dtype
            )
            padded = torch.cat([padded, pad_tensor], dim=1)
        elif padded.size(1) > max_len:
            padded = padded[:, :max_len]

        return padded

    @staticmethod
    def _shift_labels_for_causal(labels: torch.Tensor) -> torch.Tensor:
        """Shift labels right for causal language modelling.

        ``labels[:, 0]`` is dropped and a ``-100`` column is appended.
        This means the model predicts token *t* from positions
        ``0 … t-1``.
        """
        _batch_size, seq_len = labels.shape
        shifted = torch.full_like(labels, -100)
        shifted[:, : seq_len - 1] = labels[:, 1:]
        return shifted

    # ------------------------------------------------------------------
    # 4-D causal mask utility
    # ------------------------------------------------------------------

    @staticmethod
    def make_causal_mask(
        seq_len: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Create a 4-D causal attention mask.

        Returns a ``[batch_size, 1, seq_len, seq_len]`` float tensor
        where ``1`` = attend and ``0`` = mask (lower-triangular).

        This is useful when the model expects an explicit mask rather
        than building one internally.

        Args:
            seq_len: Sequence length.
            batch_size: Batch size (broadcastable).
            device: Target device.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    @staticmethod
    def make_padding_mask(
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a 2-D attention mask to 4-D for multi-head attention.

        Args:
            attention_mask: ``[batch_size, seq_len]`` with 1 = real
                token, 0 = padding.

        Returns:
            ``[batch_size, 1, 1, seq_len]`` broadcastable with
            ``[batch_size, n_heads, seq_len, seq_len]`` attention
            scores.
        """
        return attention_mask.unsqueeze(1).unsqueeze(2)
