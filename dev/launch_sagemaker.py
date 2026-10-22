"""
Training / Processing Job Orchestrator (Requires AWS Access).
Bundles this `dev/` package and sends it to the docker container (ml.g5.xlarge).
Note: Can be tested locally using Docker if you set config.is_local to True.
However, you still need AWS credentials to run locally since it requires AWS ECR
(Elastic Container Repository) to download the container images.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from boto3 import session as boto_session
from sagemaker import session
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch

# This file lives inside the `dev/` package — always upload this directory,
# regardless of the caller's working directory.
DEV_PACKAGE_DIR = str(Path(__file__).resolve().parent)


@dataclass
class Config:
    # Training config
    model_name: str = "housing-additional-needs-relations"
    model_desc: str = "alBERT-base-v2 relations model"
    source_dir: str = DEV_PACKAGE_DIR
    entry_point: str = "train_relation.py"
    instance_type: str = "ml.g5.xlarge"

    # AWS Config
    is_local: bool = False
    aws_profile_name: str = "data-platform-admin-dev"
    aws_region: str = "eu-west-2"
    role_arn: str = "arn:aws:iam::484466746276:role/service-role/AmazonSageMaker-ExecutionRole-20260617T135400"
    s3_bucket: str = "s3://additional-needs-data-dev/"
    tags = [
        {"Key": "Application", "Value": "housing-additional-needs"},
        {"Key": "TeamEmail", "Value": "shared.services@hackney.gov.uk"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Confidentiality", "Value": "Internal"},
    ]

    # S3 URIs
    s3_train_data: str = f"{s3_bucket}data/train_data.json"
    s3_val_data: str = f"{s3_bucket}data/val_data.json"
    s3_output_path: str = f"{s3_bucket}output/"


def _remote_source_dir(source: Path) -> str:
    """Directory the remote SDK will tar. It packs every top-level entry and has no ignore list.

    .venv is about 2 GB, so including it makes fit() sit with no output. Symlinks are
    followed when the SDK builds the archive, so the tarball still contains the real files.
    """
    skip = {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
    staging = Path(tempfile.mkdtemp(prefix="sm-src-"))
    for child in source.iterdir():
        if child.name in skip or child.name.endswith(".egg-info"):
            continue
        os.symlink(child.resolve(), staging / child.name)
    return str(staging)


def _setup_session(config: Config):
    aws_session = boto_session.Session(
        profile_name=config.aws_profile_name, region_name=config.aws_region
    )
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=aws_session)
        sagemaker_session.config = {"local": {"local_code": True}}
    else:
        sagemaker_session = session.Session(boto_session=aws_session)
    return sagemaker_session


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Config()
    sagemaker_session = _setup_session(config)
    source_dir = config.source_dir
    if not config.is_local:
        source_dir = _remote_source_dir(Path(config.source_dir))
        logging.info("Remote source dir excludes .venv: %s", source_dir)

    logger = logging.getLogger("train_sagemaker")

    # Set up training job
    estimator = PyTorch(
        entry_point=config.entry_point,
        source_dir=source_dir,
        base_job_name=f"train-{config.model_name}",
        requirements="requirements-sagemaker.txt",
        sagemaker_session=sagemaker_session,
        role=config.role_arn,
        framework_version="2.1.0",
        py_version="py310",
        instance_count=1,
        instance_type="local" if config.is_local else config.instance_type,
        tags=config.tags,
        output_path=config.s3_output_path,
    )

    # Kick off the job
    logger.info("Starting training...")
    inputs = {
        "train": config.s3_train_data,
        "val": config.s3_val_data,
    }
    estimator.fit(inputs=inputs)


if __name__ == "__main__":
    main()
