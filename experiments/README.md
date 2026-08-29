# Experiments

This directory holds configuration files for reproducible training runs.
Each experiment lives in its own subdirectory with a `config.yaml` file
that can be passed directly to the training script.

## Directory Structure

```
experiments/
  exp_001_baseline/config.yaml   # Vanilla multi-head attention baseline
  exp_002_flash/config.yaml      # FlashAttention v2 with tiled blocks
  exp_003_gqa/config.yaml        # Grouped-Query Attention
  README.md
```

## How to Run

### Using `scripts/run_train.py`

```bash
# Baseline experiment
python scripts/run_train.py --config experiments/exp_001_baseline/config.yaml

# FlashAttention experiment
python scripts/run_train.py --config experiments/exp_002_flash/config.yaml

# GQA experiment
python scripts/run_train.py --config experiments/exp_003_gqa/config.yaml
```

### Overriding Config Values

Individual CLI flags override the YAML config:

```bash
python scripts/run_train.py \
    --config experiments/exp_001_baseline/config.yaml \
    --num_epochs 20 \
    --lr 1e-4 \
    --batch_size 16
```

### Running Benchmarks

```bash
python scripts/run_benchmark.py \
    --attention_types vanilla flashv2 gqa mqa \
    --seq_lengths 256 512 1024 2048 \
    --output_dir outputs/reports
```

### Evaluation

```bash
python scripts/run_eval.py \
    --checkpoint outputs/checkpoints/exp_001_baseline/best.pt \
    --data_path data/val.txt \
    --model_type decoder_only
```

## Adding a New Experiment

1. Copy an existing config directory:
   ```bash
   cp -r experiments/exp_001_baseline experiments/exp_004_new
   ```
2. Edit `experiments/exp_004_new/config.yaml` with your changes.
3. Run:
   ```bash
   python scripts/run_train.py --config experiments/exp_004_new/config.yaml
   ```

## Config Reference

All config fields map directly to `TransformerConfig` (model) and
`TrainerConfig` (training loop) parameters. See the source code for
full documentation of each field.
