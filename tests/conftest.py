import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.transformer import TransformerConfig  # noqa: E402
from tests.helpers import BATCH, D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN, VOCAB  # noqa: E402


@pytest.fixture
def small_config():
    """Tiny TransformerConfig for fast tests."""
    return TransformerConfig(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=2,
        d_ff=D_MODEL * 4,
        dropout=0.0,
        max_seq_len=SEQ_LEN,
        attn_type="alibi",
        pos_encoding="sinusoidal",
        causal=True,
        use_flash=True,
        tie_embeddings=True,
        use_bias=False,
        ffn_activation="swiglu",
        layer_norm_type="pre",
    )


@pytest.fixture
def dummy_input():
    """Random token ids [B, T]."""
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture
def dummy_hidden():
    """Random hidden states [B, T, d_model]."""
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)
