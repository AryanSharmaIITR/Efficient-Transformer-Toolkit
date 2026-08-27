from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """Plain-text dataset for language-model pre-training.

    Reads one or more text files, tokenizes each *document* (separated
    by blank lines by default), and stores the resulting token-id
    sequences for training.

    For **causal (decoder-only)** models each document is a
    self-contained sequence: ``labels == input_ids`` and a lower-
    triangular attention mask is applied at collation time.

    For **encoder-decoder** models the caller can supply
    ``source_texts`` and ``target_texts`` to create parallel pairs.

    Args:
        tokenizer: A :class:`Tokenizer` instance.
        filepaths: Path(s) to plain-text files.
        max_seq_len: Maximum sequence length (truncated if longer).
        stride: Step size when sliding a window over long documents.
            ``0`` means no stride (one window per document).
        source_texts: Optional explicit source texts (encoder input).
            Mutually exclusive with *filepaths*.
        target_texts: Optional explicit target texts (decoder target).
        column: Column name used in the returned dict (``"input_ids"``).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        filepaths: str | Path | list[str | Path] | None = None,
        *,
        max_seq_len: int = 2048,
        stride: int = 0,
        source_texts: list[str] | None = None,
        target_texts: list[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        # Decoder-start token for teacher forcing (encoder-decoder mode):
        # prefer BOS, fall back to pad (the common T5-style convention) when
        # the tokenizer has no dedicated BOS token.
        self._decoder_start_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.pad_token_id
        )

        if source_texts is not None:
            self._source_texts = source_texts
            self._target_texts = target_texts or []
            self._documents: list[str] = []
        elif filepaths is not None:
            self._source_texts = None
            self._target_texts = None
            self._documents = self._load_documents(filepaths)
        else:
            raise ValueError("Provide either filepaths or source_texts")

        self._tokenized: list[list[int]] = self._tokenize_documents()
        self._tokenized_targets: list[list[int]] = self._tokenize_targets()
        self._windows: list[tuple[int, int]] = self._build_windows()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_documents(filepaths: str | Path | list[str | Path]) -> list[str]:
        """Read text files and split into documents on blank lines."""
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]

        documents: list[str] = []
        for fp in filepaths:
            text = Path(fp).read_text(encoding="utf-8")
            docs = [d.strip() for d in text.split("\n\n") if d.strip()]
            documents.extend(docs)
        return documents

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize_documents(self) -> list[list[int]]:
        """Tokenize all documents."""
        if self._source_texts is not None:
            texts = self._source_texts
        else:
            texts = self._documents

        tokenized: list[list[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_seq_len)
            tokenized.append(ids)
        return tokenized

    def _tokenize_targets(self) -> list[list[int]]:
        """Tokenize target texts (encoder-decoder mode)."""
        if self._target_texts is None:
            return []
        return [
            self.tokenizer.encode(t, truncation=True, max_length=self.max_seq_len)
            for t in self._target_texts
        ]

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def _build_windows(self) -> list[tuple[int, int]]:
        """Build (doc_idx, start) pairs for each training window."""
        windows: list[tuple[int, int]] = []
        for idx, ids in enumerate(self._tokenized):
            if self.stride <= 0 or len(ids) <= self.max_seq_len:
                windows.append((idx, 0))
            else:
                for start in range(0, len(ids), self.stride):
                    if start + self.max_seq_len > len(ids):
                        break
                    windows.append((idx, start))
                if not windows or windows[-1] != (idx, max(0, len(ids) - self.max_seq_len)):
                    windows.append((idx, max(0, len(ids) - self.max_seq_len)))
        return windows

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._source_texts is not None:
            return len(self._source_texts)
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        """Return a single training example.

        Returns:
            Dict with:
              - ``input_ids``: Token-id list.
              - ``attention_mask``: 1s where there are real tokens.
              - ``labels``: Copy of *input_ids* (shifted right at
                collation time for causal LM).
              - For encoder-decoder mode, also ``decoder_input_ids``
                and ``decoder_attention_mask``.
        """
        if self._source_texts is not None:
            return self._get_encoder_decoder_item(idx)
        return self._get_causal_item(idx)

    def _get_causal_item(self, idx: int) -> dict[str, list[int]]:
        doc_idx, start = self._windows[idx]
        ids = self._tokenized[doc_idx]
        end = min(start + self.max_seq_len, len(ids))
        window = ids[start:end]
        mask = [1] * len(window)

        return {
            "input_ids": window,
            "attention_mask": mask,
            "labels": list(window),
        }

    def _get_encoder_decoder_item(self, idx: int) -> dict[str, list[int]]:
        src_ids = self._tokenized[idx]
        targets = self._tokenized_targets

        item: dict[str, list[int]] = {
            "input_ids": src_ids,
            "attention_mask": [1] * len(src_ids),
        }

        if idx < len(targets):
            tgt_ids = targets[idx]
            # Teacher forcing: decoder_input_ids is the target shifted right
            # by one (prefixed with a decoder-start token, last token
            # dropped) so that position t sees target[0..t-1] and predicts
            # labels[t] == target[t]. Without the shift, decoder_input_ids
            # and labels were identical, so causal self-attention let
            # position t attend to its own label -- a degenerate, trivially
            # "solvable" training signal.
            decoder_input_ids = [self._decoder_start_token_id, *tgt_ids[:-1]]
            item["decoder_input_ids"] = decoder_input_ids
            item["decoder_attention_mask"] = [1] * len(decoder_input_ids)
            item["labels"] = list(tgt_ids)

        return item

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_document(self, idx: int) -> str:
        """Return the raw text of document *idx*."""
        if self._source_texts is not None:
            return self._source_texts[idx]
        doc_idx, _ = self._windows[idx]
        return self._documents[doc_idx]

    @property
    def num_documents(self) -> int:
        """Number of raw documents (before windowing)."""
        if self._source_texts is not None:
            return len(self._source_texts)
        return len(self._documents)
