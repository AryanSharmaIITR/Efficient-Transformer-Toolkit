# Docker Setup

This directory contains the Docker configuration for running the
Efficient-Transformer-Toolkit in a reproducible GPU-enabled container.

## Prerequisites

- **Docker Desktop** (Windows / macOS) or **Docker Engine** (Linux)
- **NVIDIA Container Toolkit** — required for GPU passthrough.
  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Quick Start

### Build the image

```bash
docker build -f docker/Dockerfile -t transformer_toolkit .
```

### Run with docker-compose (recommended)

```bash
cd docker
docker-compose up --build
```

This starts Jupyter Lab on **http://localhost:8888** with:
- Live source-code sync (edit files on host, see changes inside the container)
- GPU access via NVIDIA Container Toolkit
- Persistent `data/` and `outputs/` volumes

### Run with plain Docker

```bash
docker run --gpus all -p 8888:8888 \
  -v "$(pwd):/app" \
  -v "$(pwd)/data:/app/data" \
  transformer_toolkit
```

## Accessing Jupyter Lab

After `docker-compose up`, open:

```
http://localhost:8888
```

No token is set by default (development only).

## Useful Commands

| Action | Command |
|---|---|
| Start (detached) | `docker-compose up -d` |
| View logs | `docker-compose logs -f` |
| Stop & remove | `docker-compose down` |
| Rebuild from scratch | `docker-compose build --no-cache` |
| Shell into container | `docker exec -it transformer_toolkit bash` |
| Run a script | `docker exec transformer_toolkit python scripts/run_train.py --config experiments/exp_001_baseline/config.yaml` |

## GPU Notes

- The `deploy.resources.reservations.devices` block in `docker-compose.yml`
  requests all available NVIDIA GPUs.
- Set `NVIDIA_VISIBLE_DEVICES` to limit which GPUs are visible
  (e.g., `NVIDIA_VISIBLE_DEVICES=0,1`).
- Verify GPU access inside the container:
  ```bash
  docker exec transformer_toolkit python -c "import torch; print(torch.cuda.is_available())"
  ```

## Troubleshooting

**"could not select device driver" or GPU not visible**
: Ensure the NVIDIA Container Toolkit is installed and Docker has been
  restarted after installation. Run `nvidia-smi` on the host to verify.

**Port 8888 already in use**
: Change the host port mapping in `docker-compose.yml`:
  `ports: ["8889:8888"]`

**Container runs out of memory**
: Reduce `--max_seq_len` and `--batch_size` in your experiment config,
  or allocate specific GPUs with `NVIDIA_VISIBLE_DEVICES=0`.
