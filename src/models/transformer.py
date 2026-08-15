import math
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.base import BaseAttention
from ..attention.flash import FlashAttentionWrapper
from ..attention.alibi import Alibi
from ..attention.multiheadattention import MultiHeadAttention
from ..attention.gqa import GroupedQueryAttention
from ..attention.mqa import MultiQueryAttention

from ..position.rope import RotaryPositionalEmbedding
from ..position.learn import LearnedPositionalEmbedding
from ..position.sinusoidal import SinusoidalPositionalEncoding

@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 768
    dk: int = 64 # use when attn_type is mqa
    dv: int = 64 # use when attn_type is mqa
    n_heads: int = 12
    n_layers: int = 6
    d_ff: Optional[int] = None
    dropout: float = 0.1
    eps: float = 1e-5
    max_seq_len: int = 2048
    attn_type: str = "flashv2"  #"flashv1", flashv2, "alibi", "gqa", "mqa"
    pos_encoding: str = "rope"  # "rope", "sinusoidal", "learned"
    n_kv_heads: Optional[int] = None  # For GQA/MQA, if None then equal to n_heads
    causal: bool = True  # Default causal
    use_flash: bool = True  # Try to use FlashAttention if available
    tie_embeddings: bool = True
    use_bias: bool = False
    ffn_activation: str = "swiglu"  # "gelu", "relu", "silu", "swiglu"
    ffn_expansion: Optional[int] = None  # If None, use 4
    layer_norm_type: str = "pre"  # "pre" or "post"
    

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.ffn_expansion is None:
            self.ffn_expansion = 4
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.attn_type in ("gqa", "mqa"):
            if self.attn_type == "mqa":
                self.n_kv_heads = 1

            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = expansion_factor * d_model
        self.W1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.W2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.W3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.W1(x)
        x2 = self.W2(x)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        x = self.W3(x)
        return x


def get_ffn(d_model: int, d_ff: int, activation: str, dropout: float) -> nn.Module:
    if activation == "swiglu":
        return SwiGLU(d_model, d_ff // d_model, dropout)
    else:
        return FeedForward(d_model, d_ff, activation, dropout)


def get_positional_encoding(config: TransformerConfig) -> Optional[nn.Module]:
    if config.pos_encoding == "rope":
        return RotaryPositionalEmbedding(
            config.d_model // config.n_heads,
            config.max_seq_len,
            base=10000
        )
    elif config.pos_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
    elif config.pos_encoding == "learned":
        return LearnedPositionalEmbedding(config.d_model, config.max_seq_len)
    else:
        return None


def get_attention(config: TransformerConfig) -> BaseAttention:
    d_model = config.d_model
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    dropout = config.dropout
    use_flash = config.use_flash
    use_bias = config.use_bias
    casual = config.causal

    if config.attn_type == "flashv1":
        if use_flash:
            return FlashAttentionWrapper(
                d_model, n_heads,bias=use_bias,
                dropout=dropout, block_size=128,
                casual=casual, use_v2=False
            )
        else:
            raise ImportError("FlashAttention disabled")
    elif config.attn_type == "flashv2":
        if use_flash:
            return FlashAttentionWrapper(
                d_model, n_heads,bias=use_bias,
                dropout=dropout, block_size=128,
                casual=casual, use_v2=True
            )
        else:
            raise ImportError("FlashAttention disabled")
    elif config.attn_type == "alibi":
        return Alibi(d_model, n_heads,
                    dropout=dropout, max_seq_len=config.max_seq_len,
                    causal=casual, bias=use_bias
                )
    elif config.attn_type == "gqa":
        return GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout,
            bias=use_bias, use_flash=use_flash,
            casual=casual
        )
    elif config.attn_type == "mqa":
        return MultiQueryAttention(
            d_model, n_heads, config.dk, config.dv,
            dropout=dropout, bias=use_bias,
            casual=casual
        )
    else:
        raise ValueError(f"Unsupported attention type: {config.attn_type}")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attention = get_attention(config)
        self.ffn = get_ffn(
            config.d_model,
            config.d_ff,
            config.ffn_activation,
            config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm_type = config.layer_norm_type


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm
        if self.layer_norm_type == "pre":
            residual = x
            x_norm = self.norm1(x)
            attn_out = self.attention(x_norm, x_norm, x_norm, mask, causal=self.config.causal)
            x = residual + self.dropout(attn_out)

            # FFN
            residual = x
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = residual + self.dropout(ffn_out)
        else:
            # Post-norm
            attn_out = self.attention(x, x, x, mask, causal=self.config.causal)
            x = self.norm1(x + self.dropout(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.self_attn = get_attention(config)
        # Cross-attention uses same config but separate instance
        self.cross_attn = get_attention(config)
        self.ffn = get_ffn(
            config.d_model,
            config.d_ff,
            config.ffn_activation,
            config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm_type = config.layer_norm_type


    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        if self.layer_norm_type == "pre":
            residual = x
            x_norm = self.norm1(x)
            attn_out = self.self_attn(x_norm, x_norm, x_norm, self_mask, causal=True)
            x = residual + self.dropout(attn_out)

            # Cross-attention
            if encoder_output is not None:
                residual = x
                x_norm = self.norm2(x)
                cross_out = self.cross_attn(
                    x_norm, encoder_output, encoder_output, cross_mask, causal=False
                )
                x = residual + self.dropout(cross_out)

            # FFN
            residual = x
            x_norm = self.norm3(x)
            ffn_out = self.ffn(x_norm)
            x = residual + self.dropout(ffn_out)
        else:
            # Post-norm
            attn_out = self.self_attn(x, x, x, self_mask, causal=True)
            x = self.norm1(x + self.dropout(attn_out))

            if encoder_output is not None:
                cross_out = self.cross_attn(x, encoder_output, encoder_output, cross_mask, causal=False)
                x = self.norm2(x + self.dropout(cross_out))

            ffn_out = self.ffn(x)
            x = self.norm3(x + self.dropout(ffn_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model, eps=config.eps)

        self.pos_encoding = get_positional_encoding(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, positions)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, positions)

        x = self.norm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model, eps=config.eps)
        self.pos_encoding = get_positional_encoding(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, positions)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask, positions)

        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        use_encoder: bool = True,
        use_decoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_encoder = use_encoder
        self.use_decoder = use_decoder

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.tie_embeddings = config.tie_embeddings

        if use_encoder:
            self.encoder = TransformerEncoder(config)
        else:
            self.encoder = None

        if use_decoder:
            self.decoder = TransformerDecoder(config)
        else:
            self.decoder = None

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        return_encoder_output: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        For encoder-only:
            input_ids: [B, T]
            attention_mask: [B, T] or [B, 1, T, T]
            returns logits: [B, T, vocab_size]

        For decoder-only:
            input_ids: [B, T]
            attention_mask: [B, T] or [B, 1, T, T] (causal)
            returns logits: [B, T, vocab_size]

        For encoder-decoder:
            input_ids: [B, T_enc]
            attention_mask: [B, T_enc] or [B, 1, T_enc, T_enc]
            decoder_input_ids: [B, T_dec]
            decoder_attention_mask: [B, T_dec] or [B, 1, T_dec, T_dec]
            returns: dict with logits and optionally encoder_output
        """
        # Token embedding
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Encoder
        if self.use_encoder:
            if attention_mask is not None and attention_mask.dim() == 2:
                # Convert to 4D mask if needed (for attention)
                # We'll handle mask inside layers; they expect either 2D or 4D
                pass
            encoder_output = self.encoder(x, attention_mask, positions)

        # Decoder-only: use input as both encoder and decoder?
        if self.use_decoder and not self.use_encoder:
            # Decoder-only (like GPT)
            if decoder_input_ids is None:
                decoder_input_ids = input_ids
            dec = self.token_embedding(decoder_input_ids)
            dec = self.dropout(dec)
            # For decoder-only, we treat input as both encoder_output? no
            # We just run decoder with self-attention
            out = self.decoder(
                dec,
                encoder_output=None,  # no cross-attention
                self_mask=attention_mask,  # causal mask
                cross_mask=None,
                positions=positions,
            )
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": None}
            else:
                return logits

        # Encoder-decoder
        if self.use_encoder and self.use_decoder:
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids required for encoder-decoder")
            dec = self.token_embedding(decoder_input_ids)
            dec = self.dropout(dec)
            out = self.decoder(
                dec,
                encoder_output=encoder_output,
                self_mask=decoder_attention_mask,  # causal mask for self
                cross_mask=attention_mask,  # cross attention mask (encoder mask)
                positions=positions,
            )
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": encoder_output}
            else:
                return logits

        # Encoder-only (like BERT)
        if self.use_encoder and not self.use_decoder:
            out = self.encoder(x, attention_mask, positions)
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": out}
            else:
                return logits

        raise ValueError("Invalid combination: must have at least one of encoder/decoder")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation for decoder-only or encoder-decoder.

        Args:
            input_ids: [B, T] initial prompt.
            max_new_tokens: maximum number of new tokens to generate.
            temperature: sampling temperature.
            top_k: top-k sampling.
            top_p: top-p (nucleus) sampling.
            do_sample: if True, sample; else greedy.
            pad_token_id: token id for padding (to stop generation).
            eos_token_id: token id for end-of-sequence.

        Returns:
            Generated token ids [B, T + max_new_tokens].
        """
        self.eval()
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if self.use_encoder and self.use_decoder:
                    if _ == 0:
                        enc_out = self.encoder(
                            input_ids,
                            attention_mask=None,  # assume full mask
                            positions=None
                        )
                    outputs = self(
                        input_ids=input_ids,
                        decoder_input_ids=generated,
                        encoder_output=enc_out,
                        return_encoder_output=False
                    )
                else:
                    outputs = self(input_ids=generated, return_encoder_output=False)

                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('inf')
                    if top_p > 0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)

                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
                if pad_token_id is not None and (next_token == pad_token_id).any():
                    break

        return generated

    def save_pretrained(self, path: str):
        import json
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str):
        import json
        with open(f"{path}/config.json", "r") as f:
            cfg_dict = json.load(f)
        config = TransformerConfig(**cfg_dict)
        model = cls(config)
        model.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
        return model



if __name__ == "__main__":
    # Example 1: Decoder-only (GPT-like)
    config_gpt = TransformerConfig(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=6,
        attn_type="flash",
        pos_encoding="rope",
        causal=True,
        use_encoder=False,
        use_decoder=True,
    )
    model_gpt = Transformer(config_gpt, use_encoder=False, use_decoder=True)
    print("GPT-like model parameters:", sum(p.numel() for p in model_gpt.parameters()))

    config_bert = TransformerConfig(
        vocab_size=30522,
        d_model=768,
        n_heads=12,
        n_layers=6,
        attn_type="vanilla",
        pos_encoding="learned",
        causal=False,
        use_encoder=True,
        use_decoder=False,
    )
    model_bert = Transformer(config_bert, use_encoder=True, use_decoder=False)
    print("BERT-like model parameters:", sum(p.numel() for p in model_bert.parameters()))

    config_t5 = TransformerConfig(
        vocab_size=32128,
        d_model=512,
        n_heads=8,
        n_layers=6,
        attn_type="gqa",
        n_kv_heads=4,
        pos_encoding="sinusoidal",
        causal=False,  # cross-attention non-causal
        use_encoder=True,
        use_decoder=True,
    )
    model_t5 = Transformer(config_t5, use_encoder=True, use_decoder=True)
    print("T5-like model parameters:", sum(p.numel() for p in model_t5.parameters()))