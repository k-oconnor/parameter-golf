# Parameter Golf — CUDA training image (RunPod / single-node multi-GPU).
# Base ships PyTorch+CUDA; we add Python deps without reinstalling torch from PyPI.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/parameter-golf

# Dependencies (torch already in base image)
RUN pip install \
    numpy \
    tqdm \
    huggingface-hub \
    kernels \
    setuptools \
    "typing-extensions==4.15.0" \
    datasets \
    tiktoken \
    sentencepiece

COPY . .

# Default: open shell; RunPod should set start command to scripts/runpod_train.sh or torchrun.
CMD ["bash", "-lc", "echo 'Run: bash scripts/runpod_train.sh  (or torchrun --standalone --nproc_per_node=8 train_gpt.py)' && exec bash"]
