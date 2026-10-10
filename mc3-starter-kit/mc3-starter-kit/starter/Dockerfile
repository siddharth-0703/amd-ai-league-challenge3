# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Install system dependencies (tesseract-ocr for multimodal images / diagrams)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies defensively to protect the ROCm torch installation
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verify that torch is intact and remains the ROCm build
RUN python3 -c "import torch; assert 'rocm' in torch.__version__, f'PyTorch build clobbered: {torch.__version__}'; print('Verified ROCm torch:', torch.__version__)"

COPY app/ /app/

# Pre-create necessary directories
RUN mkdir -p /app/corpus /app/output /app/index

# No service, no port. The container stays running to be exec'd into.
CMD ["sleep", "infinity"]
