FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        adb \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0

RUN python -m pip install --no-cache-dir \
        "protobuf>=4.21.6,<=4.25.4" \
        "psutil>=5.9.0" \
        "ruamel.yaml>=0.17.21" \
        "scipy>=1.9.3" \
        "tqdm>=4.64.1" \
        "fast-histogram>=0.11" \
        "numpy<=1.26.4" \
        "onnx>=1.16.1" \
        "onnxruntime>=1.17.0" \
        "opencv-python-headless>=4.5.5.64" \
    && python -m pip install --no-cache-dir --no-deps rknn-toolkit2==2.3.2

WORKDIR /workspace

ENTRYPOINT ["python", "scripts/testing/rknn_toolkit2_profile.py"]