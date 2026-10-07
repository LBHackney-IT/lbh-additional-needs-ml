# Production Inference

Contains the core application code, PyTorch models, and Docker container configuration used to run batch prediction jobs against historical case notes.

## Local Setup & Testing

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements-dev.txt

# Run  unit tests
pytest tests/
```

## Running Container Locally
To test the container runtime before pushing to AWS ECR:

```bash
docker build -t housing-additional-needs-ml .
docker run --rm housing-additional-needs-ml
```