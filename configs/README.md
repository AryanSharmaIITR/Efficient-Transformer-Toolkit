# Configs Directory

Central location for all YAML configuration files. These files define
model architecture, training hyperparameters, data settings, and
benchmark parameters — enabling fully reproducible experiments without
changing any code.

## Files

| File | Purpose |
|---|---|
| `default_config.yaml` | Master config with all defaults. Start here. |
| `training_config.yaml` | Quick-debug overrides (small model, fewer steps). |
| `benchmark_config.yaml` | Benchmark-only settings (seq lengths, iteration count). |

## How Configs Map to Code

Every key under `model:` corresponds 1-to-1 with a field in the
`TransformerConfig` dataclass (`src/models/transformer.py`):

```python
from src.utils.config import load_config
from src.models.transformer import TransformerConfig

cfg = load_config("configs/default_config.yaml")
model_config = TransformerConfig(**cfg["model"])
```

Every key under `training:` corresponds to a field in `TrainerConfig`
(`src/training/trainer.py`):

```python
from src.training.trainer import TrainerConfig

trainer_config = TrainerConfig(**cfg["training"])
```

## Using with the CLI Scripts

The training script (`scripts/run_train.py`) accepts `--config` and
merges YAML values onto its flat CLI arguments. YAML keys that match a
parser argument destination override the defaults:

```bash
# Use the full default config
python scripts/run_train.py --config configs/default_config.yaml

# Override specific fields from the command line
python scripts/run_train.py \
    --config configs/training_config.yaml \
    --num_epochs 20 \
    --lr 1e-4

# Benchmark with dedicated config
python scripts/run_benchmark.py --config configs/benchmark_config.yaml
```

## Creating a New Experiment

1. Copy the default config:
   ```bash
   cp configs/default_config.yaml configs/my_experiment.yaml
   ```
2. Edit only the fields you want to change.
3. Run:
   ```bash
   python scripts/run_train.py --config configs/my_experiment.yaml
   ```

Or use the per-experiment configs under `experiments/`:

```bash
python scripts/run_train.py --config experiments/exp_001_baseline/config.yaml
```

## Config Key Reference

### `model.*` (TransformerConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | int | 50257 | Vocabulary size |
| `d_model` | int | 768 | Model embedding dimension |
| `dk` | int | 64 | Head dimension for MQA |
| `dv` | int | 64 | Value dimension for MQA |
| `n_heads` | int | 12 | Number of attention heads |
| `n_layers` | int | 6 | Number of transformer layers |
| `d_ff` | int | null | FFN hidden dim (auto = 4 * d_model) |
| `dropout` | float | 0.1 | Dropout rate |
| `eps` | float | 1e-5 | LayerNorm epsilon |
| `max_seq_len` | int | 2048 | Maximum sequence length |
| `attn_type` | str | flashv2 | `flashv1`, `flashv2`, `alibi`, `gqa`, `mqa` |
| `pos_encoding` | str | rope | `rope`, `sinusoidal`, `learned`, or omit |
| `n_kv_heads` | int | 12 | KV heads (1 = MQA, n_heads = MHA) |
| `causal` | bool | true | Use causal masking |
| `use_flash` | bool | true | Enable FlashAttention |
| `tie_embeddings` | bool | true | Tie input/output embeddings |
| `use_bias` | bool | false | Use bias in linear layers |
| `ffn_activation` | str | swiglu | `gelu`, `relu`, `silu`, `swiglu` |
| `ffn_expansion` | int | 4 | FFN expansion factor |
| `layer_norm_type` | str | pre | `pre` or `post` normalization |

### `training.*` (TrainerConfig)

| Key | Type | Default | Description |
|---|---|---|---|
| `lr` | float | 3e-4 | Peak learning rate |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `warmup_ratio` | float | 0.03 | Fraction of steps for warmup |
| `max_grad_norm` | float | 1.0 | Gradient clipping norm |
| `num_epochs` | int | 10 | Training epochs |
| `use_amp` | bool | true | Automatic mixed precision |
| `device` | str | cuda | `cuda` or `cpu` |
| `seed` | int | 42 | Random seed |
