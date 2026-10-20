# CUDA 12.4 runtime with PyTorch 2.6 already installed.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copy dependency definition and source code
COPY pyproject.toml .
COPY src/ /app/src/

# Install dependencies directly from pyproject.toml while keeping pre-installed PyTorch
RUN pip install --no-cache-dir --no-build-isolation .

ENTRYPOINT ["python3", "-m", "src.run_inference"]