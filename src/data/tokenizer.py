from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

load_dotenv()


class Tokenizer:
    """Thin wrapper around a HuggingFace tokenizer with pad/eos convenience.

    Provides a stable interface for encoding, decoding, and padding,
    and exposes special-token ids as properties.

    Reads ``HF_TOKEN`` from the environment (or a ``.env`` file at the
    project root) and forwards it to ``AutoTokenizer.from_pretrained``
    so that gated / private models can be downloaded without warnings.

    Args:
        pretrained_model_name_or_path: Any valid HuggingFace tokenizer
            identifier (e.g. ``"gpt2"``, ``"meta-llama/Llama-2-7b-hf"``,
            or a local directory).
        **kwargs: Extra keyword arguments forwarded to
            ``AutoTokenizer.from_pretrained`` (``use_fast``,
            ``trust_remote_code``, …).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> None:
        token = os.getenv("HF_TOKEN")
        if token and "token" not in kwargs:
            kwargs["token"] = token

        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(
                str(pretrained_model_name_or_path),
                **kwargs,
            )
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Id of the padding token."""
        return self._tokenizer.pad_token_id  # type: ignore[return-value]

    @property
    def eos_token_id(self) -> int | None:
        """Id of the end-of-sequence token."""
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Id of the beginning-of-sequence token."""
        return self._tokenizer.bos_token_id

    @property
    def unk_token_id(self) -> int | None:
        """Id of the unknown token."""
        return self._tokenizer.unk_token_id

    @property
    def model_max_length(self) -> int:
        """Maximum sequence length the model supports."""
        return self._tokenizer.model_max_length

    @property
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Return the underlying HuggingFace tokenizer."""
        return self._tokenizer

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str | list[str],
        *,
        max_length: int | None = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> list[int] | list[list[int]]:
        """Encode one or more strings into token ids.

        Args:
            text: A single string or a list of strings.
            max_length: Truncate to this many tokens.  ``None`` means
                use the tokenizer's default.
            truncation: Whether to truncate.
            add_special_tokens: Whether to prepend BOS / append EOS.

        Returns:
            A single list of ids when *text* is a string, or a list of
            lists when *text* is a list.
        """
        result = self._tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            padding=False,
            return_tensors=None,
        )
        ids = result["input_ids"]
        if isinstance(text, str):
            return ids[0] if isinstance(ids[0], list) else ids
        return ids  # type: ignore[return-value]

    def decode(
        self,
        token_ids: list[int] | torch.Tensor | list[list[int]],
        skip_special_tokens: bool = True,
    ) -> str | list[str]:
        """Decode token ids back to string(s).

        Args:
            token_ids: 1-D tensor/list for a single sequence, 2-D for
                a batch.
            skip_special_tokens: Drop special tokens from the output.

        Returns:
            A single string or a list of strings.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if token_ids and isinstance(token_ids[0], list):
            return [
                self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in token_ids
            ]
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    def pad(
        self,
        batch: list[list[int]],
        *,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> torch.Tensor:
        """Pad a batch of token-id sequences to equal length.

        Args:
            batch: List of 1-D token-id lists.
            max_length: Pad to this length.  ``None`` means pad to the
                longest sequence in *batch*.
            pad_to_multiple_of: Round up to the nearest multiple.

        Returns:
            ``[batch_size, seq_len]`` integer tensor with pad tokens
            filled in.
        """
        if not batch:
            return torch.empty(0, 0, dtype=torch.long)

        # batch is already token ids, not text -- use the tokenizer's
        # pre-tokenized padding API (BatchEncoding.pad), not __call__,
        # which encodes strings.
        result = self._tokenizer.pad(
            {"input_ids": batch},
            max_length=max_length,
            padding="longest" if max_length is None else "max_length",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        return result["input_ids"]  # type: ignore[return-value]

    def pad_batch(
        self,
        batch: list[list[int]],
        *,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pad a batch and return both ``input_ids`` and ``attention_mask``.

        Returns:
            Dict with keys ``"input_ids"`` and ``"attention_mask"``,
            each ``[batch_size, seq_len]``.
        """
        if not batch:
            return {"input_ids": torch.empty(0, 0, dtype=torch.long),
                    "attention_mask": torch.empty(0, 0, dtype=torch.long)}

        # batch is already token ids, not text -- use the tokenizer's
        # pre-tokenized padding API (BatchEncoding.pad), not __call__,
        # which encodes strings.
        result = self._tokenizer.pad(
            {"input_ids": batch},
            max_length=max_length,
            padding="longest" if max_length is None else "max_length",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    # ------------------------------------------------------------------
    # Special-token helpers
    # ------------------------------------------------------------------

    def encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode *text* and always include special tokens."""
        return self.encode(text, add_special_tokens=True)  # type: ignore[return-value]

    def add_bos(self, token_ids: list[int]) -> list[int]:
        """Prepend ``bos_token_id`` if not already present."""
        if self.bos_token_id is not None and (
            not token_ids or token_ids[0] != self.bos_token_id
        ):
            return [self.bos_token_id] + token_ids
        return token_ids

    def add_eos(self, token_ids: list[int]) -> list[int]:
        """Append ``eos_token_id`` if not already present."""
        if self.eos_token_id is not None and (
            not token_ids or token_ids[-1] != self.eos_token_id
        ):
            return token_ids + [self.eos_token_id]
        return token_ids
