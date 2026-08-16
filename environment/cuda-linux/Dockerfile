# syntax=docker/dockerfile:1.7
# Build and run this image only on Linux/WSL with an NVIDIA GPU.
FROM --platform=linux/amd64 condaforge/miniforge3:24.11.3-0 AS conda
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

COPY --from=conda /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    FORCE_CUDA=1 \
    DTLR_NATIVE_ROOT=/workspace/dtlr/.native \
    PYTHONPATH=/workspace/dtlr/.native:/workspace/dtlr \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git ninja-build ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && conda create -y -n dtlr python=3.11.0 pip=24.3.1 \
    && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "dtlr", "/bin/bash", "-c"]
WORKDIR /workspace/dtlr
COPY environment/cuda-linux/requirements.txt /tmp/dtlr-requirements.txt
RUN python -m pip install \
      --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    && python -m pip install -r /tmp/dtlr-requirements.txt \
    && python -m pip install \
      'git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0'

COPY . /workspace/dtlr

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dtlr", "bash", "environment/cuda-linux/entrypoint.sh"]
CMD ["python", "environment/cuda-linux/preflight.py", "--require-gpu"]
