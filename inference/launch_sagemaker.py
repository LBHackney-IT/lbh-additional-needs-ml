"""Run the inference container locally through SageMaker Local Mode.
Requires AWS ECR access in data platform pre-production account.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from boto3 import session as boto_session
from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

from src.gateway.athena_gateway import DEFAULT_WHERE

CONTAINER_MODEL_DIR = "/opt/ml/processing/model"
CONTAINER_OUTPUT_DIR = "/opt/ml/processing/output"
ML_BUCKET = "s3://dataplatform-stg-housing-additional-needs-ml-storage"
LOCAL_OUTPUT_DIR = Path("data/output")


# =============
# Arg Handling
# =============


@dataclass
class Config:
    # Pipeline config
    job_prefix: str = "housing-additional-needs"
    source: str = "mmh"
    athena_database: str = "prod__housing-refined-zone"
    athena_table: str = "additional_needs_notes_reshaped"
    athena_workgroup: str = "housing"
    where: str = DEFAULT_WHERE
    batch_size: int = 128
    flush_every_n_batches: int = 50
    span_model_name: str = "span_v1"
    relation_model_name: str = "relation_v1"

    # Local execution config
    aws_profile_name: str | None = None
    aws_region: str = "eu-west-2"
    image_uri: str = ""
    s3_bucket: str = ML_BUCKET
    s3_models: str = ""

    def __post_init__(self) -> None:
        # Handle S3 URIs
        bucket = self.s3_bucket.strip().rstrip("/")
        if bucket and not bucket.startswith("s3://"):
            bucket = f"s3://{bucket}"
        self.s3_bucket = bucket
        if bucket and not self.s3_models:
            self.s3_models = f"{bucket}/models/"

    def pipeline_args(self) -> list[str]:
        return [
            "--source",
            self.source,
            "--athena-database",
            self.athena_database,
            "--athena-table",
            self.athena_table,
            "--athena-workgroup",
            self.athena_workgroup,
            "--where",
            self.where,
            "--aws-region",
            self.aws_region,
            "--model-root",
            CONTAINER_MODEL_DIR,
            "--output-dir",
            CONTAINER_OUTPUT_DIR,
            "--batch-size",
            str(self.batch_size),
            "--flush-every-n-batches",
            str(self.flush_every_n_batches),
            "--span-model-name",
            self.span_model_name,
            "--relation-model-name",
            self.relation_model_name,
        ]


def _parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Run local batch inference")
    parser.add_argument("--where", default=defaults.where, help="Athena WHERE predicate")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--span-model-name",
        default=os.environ.get("SPAN_MODEL_NAME", defaults.span_model_name),
        help="Span model directory under the S3 models prefix.",
    )
    parser.add_argument(
        "--relation-model-name",
        default=os.environ.get("RELATION_MODEL_NAME", defaults.relation_model_name),
        help="Relation model directory under the S3 models prefix.",
    )
    parser.add_argument("--s3-models", default="")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("SAGEMAKER_INFERENCE_IMAGE_URI"),
        help="Local Docker image or ECR image URI.",
    )
    args = parser.parse_args()
    return Config(
        where=args.where,
        batch_size=args.batch_size,
        span_model_name=args.span_model_name,
        relation_model_name=args.relation_model_name,
        s3_models=args.s3_models,
        aws_profile_name=args.profile,
        image_uri=args.image_uri or "",
    )


def _require_config(config: Config) -> None:
    missing = []
    if not config.image_uri:
        missing.append("SAGEMAKER_INFERENCE_IMAGE_URI (or --image-uri)")
    if missing:
        raise ValueError(f"Missing required deployment configuration: {', '.join(missing)}")


def _setup_session(config: Config):
    session_kwargs = {"region_name": config.aws_region}
    if config.aws_profile_name:
        session_kwargs["profile_name"] = config.aws_profile_name
    aws_session = boto_session.Session(**session_kwargs)
    return aws_session


def _sagemaker_session(config: Config, aws_session) -> LocalSession:
    default_bucket = config.s3_bucket.removeprefix("s3://").split("/", 1)[0]
    sagemaker_session = LocalSession(
        boto_session=aws_session,
        default_bucket=default_bucket,
    )
    sagemaker_session.config = {"local": {"local_code": True}}
    return sagemaker_session


# =============
# Main
# =============


def main():
    # Load config + logging
    config = _parse_args()
    _require_config(config)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("launch_inference")

    # AWS Session
    aws_session = _setup_session(config)
    sagemaker_session = _sagemaker_session(config, aws_session)

    processor = Processor(
        role=None,
        image_uri=config.image_uri,
        instance_count=1,
        instance_type="local",
        entrypoint=["python3", "-m", "src.run_inference"],
        base_job_name=config.job_prefix,
        sagemaker_session=sagemaker_session,
    )

    logger.info("Starting local inference")
    logger.info("Image: %s", config.image_uri)
    logger.info("Models: %s", config.s3_models)
    logger.info("Output: %s", LOCAL_OUTPUT_DIR)
    logger.info("WHERE: %s", config.where or "<none>")

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor.run(
        inputs=[
            ProcessingInput(
                input_name="model",
                source=config.s3_models,
                destination=CONTAINER_MODEL_DIR,
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source=CONTAINER_OUTPUT_DIR,
                destination=f"file://{LOCAL_OUTPUT_DIR.resolve()}",
            ),
        ],
        arguments=config.pipeline_args(),
        wait=True,
        logs=True,
    )

    logger.info("Local inference complete. Output: %s", LOCAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
