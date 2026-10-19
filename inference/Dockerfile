# SageMaker PyTorch 2.6 inference image with CUDA 12.4, for ml.g5.xlarge jobs.
FROM public.ecr.aws/deep-learning-containers/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker-v1.83

WORKDIR /app

COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir .

ENTRYPOINT ["python3", "-m", "src.run_inference"]
