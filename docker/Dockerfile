# ---------------------------------------------------------------------------
# Efficient-Transformer-Toolkit — CUDA-enabled development image
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# ---- environment ----
ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache \
    PIP_NO_CACHE_DIR=1

# ---- working directory ----
WORKDIR /app

# ---- dependencies (layer cached) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab notebook ipywidgets

# ---- copy project source ----
COPY . /app

# ---- expose Jupyter port ----
EXPOSE 8888

# ---- entrypoint ----
ENTRYPOINT ["jupyter", "lab", \
            "--ip=0.0.0.0", \
            "--port=8888", \
            "--no-browser", \
            "--allow-root", \
            "--NotebookApp.token=''"]
