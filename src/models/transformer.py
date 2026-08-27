import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..attention.alibi import Alibi
from ..attention.base import BaseAttention
from ..attention.flash import FlashAttentionWrapper
from ..attention.gqa import GroupedQueryAttention
from ..attention.mqa import MultiQueryAttention
from ..position.learn import LearnedPositionalEmbedding
from ..position.rope import RotaryPEMultiHeadAttention
from ..position.sinusoidal import SinusoidalPositionalEncoding


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 768
    dk: int = 64
    dv: int = 64
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int | None = None
    dropout: float = 0.1
    eps: float = 1e-5
    max_seq_len: int = 2048
    attn_type: str = "flashv2"
    pos_encoding: str = "rope"
    n_kv_heads: int | None = None
    causal: bool = True
    use_flash: bool = True
    tie_embeddings: bool = True
    use_bias: bool = False
    ffn_activation: str = "swiglu"
    ffn_expansion: int | None = None
    layer_norm_type: str = "pre"

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
        if name == "relu":
            return nn.ReLU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    return FeedForward(d_model, d_ff, activation, dropout)


def get_positional_encoding(config: TransformerConfig) -> nn.Module | None:
    if config.pos_encoding == "rope":
        # RoPE isn't an additive embedding-level encoding -- it rotates Q/K
        # *inside* attention. get_attention() selects a rotary-aware
        # attention module for this case instead.
        return None
    if config.pos_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
    if config.pos_encoding == "learned":
        return LearnedPositionalEmbedding(config.d_model, config.max_seq_len)
    return None


def get_attention(config: TransformerConfig) -> BaseAttention:
    d_model = config.d_model
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    dropout = config.dropout
    use_flash = config.use_flash
    use_bias = config.use_bias
    causal = config.causal

    if config.pos_encoding == "rope":
        # RoPE rotates Q/K before scoring, which none of flash/alibi/gqa/mqa
        # implement -- RotaryPEMultiHeadAttention is the only attention
        # module that applies it, so it overrides attn_type here.
        if config.attn_type != "flashv2":
            warnings.warn(
                f"pos_encoding='rope' requires RotaryPEMultiHeadAttention; "
                f"ignoring attn_type={config.attn_type!r} (flashv2 is the default, "
                f"so this only fires when attn_type was explicitly set).",
                stacklevel=2,
            )
        return RotaryPEMultiHeadAttention(
            d_model, n_heads, rope_percentage=0.5,
            dropout_prob=dropout, causal=causal, bias=use_bias,
        )

    if config.attn_type == "flashv1":
        if use_flash:
            return FlashAttentionWrapper(
                d_model, n_heads, bias=use_bias,
                dropout=dropout, block_size=128,
                causal=causal, use_v2=False,
            )
        raise ImportError("FlashAttention disabled")
    if config.attn_type == "flashv2":
        if use_flash:
            return FlashAttentionWrapper(
                d_model, n_heads, bias=use_bias,
                dropout=dropout, block_size=128,
                causal=causal, use_v2=True,
            )
        raise ImportError("FlashAttention disabled")
    if config.attn_type == "alibi":
        return Alibi(
            d_model, n_heads,
            dropout=dropout, max_seq_len=config.max_seq_len,
            causal=causal, bias=use_bias,
        )
    if config.attn_type == "gqa":
        return GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout,
            bias=use_bias, use_flash=use_flash,
            causal=causal,
        )
    if config.attn_type == "mqa":
        return MultiQueryAttention(
            d_model, n_heads, config.dk, config.dv,
            dropout=dropout, bias=use_bias,
            causal=causal,
        )
    raise ValueError(f"Unsupported attention type: {config.attn_type}")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attention = get_attention(config)
        self.ffn = get_ffn(
            config.d_model, config.d_ff,
            config.ffn_activation, config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm_type = config.layer_norm_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.layer_norm_type == "pre":
            residual = x
            x_norm = self.norm1(x)
            attn_out = self.attention(x_norm, x_norm, x_norm, mask, positions)
            x = residual + self.dropout(attn_out)

            residual = x
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = residual + self.dropout(ffn_out)
        else:
            attn_out = self.attention(x, x, x, mask, positions)
            x = self.norm1(x + self.dropout(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.self_attn = get_attention(config)
        self.cross_attn = get_attention(config)
        self.ffn = get_ffn(
            config.d_model, config.d_ff,
            config.ffn_activation, config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm_type = config.layer_norm_type

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Note: positions is threaded into self-attention only. Cross-attention
        # mixes the decoder's query positions with the encoder's key
        # positions, which are different sequences with independent position
        # indices -- a single shared `positions` tensor can't correctly
        # describe both, so cross-attention keeps its default (independent
        # 0..seq_len-1 per side) rather than risk misapplying decoder
        # positions to the encoder side.
        if self.layer_norm_type == "pre":
            residual = x
            x_norm = self.norm1(x)
            attn_out = self.self_attn(x_norm, x_norm, x_norm, self_mask, positions)
            x = residual + self.dropout(attn_out)

            if encoder_output is not None:
                residual = x
                x_norm = self.norm2(x)
                cross_out = self.cross_attn(x_norm, encoder_output, encoder_output, cross_mask)
                x = residual + self.dropout(cross_out)

            residual = x
            x_norm = self.norm3(x)
            ffn_out = self.ffn(x_norm)
            x = residual + self.dropout(ffn_out)
        else:
            attn_out = self.self_attn(x, x, x, self_mask, positions)
            x = self.norm1(x + self.dropout(attn_out))

            if encoder_output is not None:
                cross_out = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
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
        encoder_output: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
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
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_output: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        return_encoder_output: bool = False,
    ) -> torch.Tensor | dict:
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        if decoder_attention_mask is not None and decoder_attention_mask.dim() == 2:
            decoder_attention_mask = decoder_attention_mask[:, None, None, :]

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        if self.use_encoder:
            encoder_output = self.encoder(x, attention_mask, positions)

        if self.use_decoder and not self.use_encoder:
            if decoder_input_ids is None:
                decoder_input_ids = input_ids
            dec = self.token_embedding(decoder_input_ids)
            dec = self.dropout(dec)
            out = self.decoder(dec, encoder_output=None, self_mask=attention_mask, cross_mask=None, positions=positions)
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": None}
            return logits

        if self.use_encoder and self.use_decoder:
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids required for encoder-decoder")
            dec = self.token_embedding(decoder_input_ids)
            dec = self.dropout(dec)
            out = self.decoder(dec, encoder_output=encoder_output, self_mask=decoder_attention_mask, cross_mask=attention_mask, positions=positions)
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": encoder_output}
            return logits

        if self.use_encoder and not self.use_decoder:
            out = self.encoder(x, attention_mask, positions)
            logits = self.lm_head(out)
            if return_encoder_output:
                return {"logits": logits, "encoder_output": out}
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
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if self.use_encoder and self.use_decoder:
                    if _ == 0:
                        enc_emb = self.token_embedding(input_ids)
                        enc_emb = self.dropout(enc_emb)
                        enc_out = self.encoder(enc_emb)
                    outputs = self(input_ids=input_ids, decoder_input_ids=generated, encoder_output=enc_out, return_encoder_output=False)
                else:
                    outputs = self(input_ids=generated, return_encoder_output=False)

                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("inf")
                    if top_p > 0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float("inf")
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
        cfg_dict = self.config.__dict__.copy()
        cfg_dict["_use_encoder"] = self.use_encoder
        cfg_dict["_use_decoder"] = self.use_decoder
        with open(f"{path}/config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str):
        import json
        with open(f"{path}/config.json", "r") as f:
            cfg_dict = json.load(f)
        use_encoder = cfg_dict.pop("_use_encoder", True)
        use_decoder = cfg_dict.pop("_use_decoder", False)
        config = TransformerConfig(**cfg_dict)
        model = cls(config, use_encoder=use_encoder, use_decoder=use_decoder)
        model.load_state_dict(torch.load(f"{path}/pytorch_model.bin"), strict=False)
        return model


if __name__ == "__main__":
    config_gpt = TransformerConfig(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=6,
        attn_type="flashv2",
        pos_encoding="rope",
        causal=True,
    )
    model_gpt = Transformer(config_gpt, use_encoder=False, use_decoder=True)
    print("GPT-like model parameters:", sum(p.numel() for p in model_gpt.parameters()))

    config_bert = TransformerConfig(
        vocab_size=30522,
        d_model=768,
        n_heads=12,
        n_layers=6,
        attn_type="flashv2",
        pos_encoding="learned",
        causal=False,
        use_flash=False,
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
        causal=False,
    )
    model_t5 = Transformer(config_t5, use_encoder=True, use_decoder=True)
    print("T5-like model parameters:", sum(p.numel() for p in model_t5.parameters()))
