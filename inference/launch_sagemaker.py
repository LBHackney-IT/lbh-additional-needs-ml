"""
Batch Inference Processing Job Launcher (Requires AWS Access).
Runs the versioned inference container published to ECR as a SageMaker Processing job.
Intended for manual runs or local debugging.

Requires `pip install -e ".[launch]"`. Set --local to run the job in Docker on this
machine. The image tag comes from the published ECR image. The execution role and
ML bucket are the staging resources created for this project.

Examples:
    export SAGEMAKER_INFERENCE_IMAGE_URI='<account>.dkr.ecr.eu-west-2.amazonaws.com/housing-additional-needs-ml:<sha>'
    python launch_sagemaker.py --where "note_id = '<note-id>'" --instance-type ml.m5.large
    Set $AWS_PROFILE or --aws-profile accordingly. You need AWS credentials even for a local run to download the image
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from boto3 import session as boto_session
from sagemaker import session
from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

from src.gateway.athena_gateway import DEFAULT_WHERE

CONTAINER_MODEL_DIR = "/opt/ml/processing/model"
CONTAINER_OUTPUT_DIR = "/opt/ml/processing/output"
EXECUTION_ROLE_NAME = "housing-additional-needs-sagemaker-execution-role"
ML_BUCKET = "s3://dataplatform-stg-housing-additional-needs-ml-storage"


# =============
# Arg Handling
# =============


@dataclass
class Config:
    # Job config
    job_prefix: str = "housing-additional-needs"
    instance_type: str = "ml.g5.xlarge"
    volume_size_gb: int = 50
    max_runtime_hours: int = 24

    # Pipeline config (forwarded to run_inference)
    source: str = "mmh"
    athena_database: str = "housing-refined-zone"
    athena_table: str = "additional_needs_notes_reshaped"
    athena_workgroup: str = "housing"
    where: str = DEFAULT_WHERE
    batch_size: int = 128
    flush_every_n_batches: int = 50
    span_model_name: str = "span_v1"
    relation_model_name: str = "relation_v1"

    # AWS Config
    is_local: bool = False
    aws_profile_name: str | None = None
    aws_region: str = "eu-west-2"
    role_arn: str = ""
    image_uri: str = ""
    s3_bucket: str = ML_BUCKET
    tags: list[dict] = field(
        default_factory=lambda: [
            {"Key": "Application", "Value": "housing-additional-needs"},
            {"Key": "TeamEmail", "Value": "mmh-project-team@hackney.gov.uk"},
            {"Key": "Environment", "Value": "stg"},
            {"Key": "Confidentiality", "Value": "Internal"},
        ]
    )

    # S3 URIs. The models prefix must contain span/ and relation/ (see README).
    s3_models: str = ""
    s3_predictions: str = ""

    def __post_init__(self) -> None:
        # Handle S3 URIs
        bucket = self.s3_bucket.strip().rstrip("/")
        if bucket and not bucket.startswith("s3://"):
            bucket = f"s3://{bucket}"
        self.s3_bucket = bucket
        if bucket and not self.s3_models:
            self.s3_models = f"{bucket}/models/"
        if bucket and not self.s3_predictions:
            self.s3_predictions = f"{bucket}/predictions/"

    def output_uri(self, job_name: str, run_date: str) -> str:
        return f"{self.s3_predictions}source={self.source}/run_date={run_date}/{job_name}/"

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


def _parse_args() -> tuple[Config, bool]:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Launch a SageMaker batch inference job")
    parser.add_argument("--where", default=defaults.where, help="Athena WHERE predicate")
    parser.add_argument("--instance-type", default=defaults.instance_type)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--span-model-name",
        default=os.environ.get("SPAN_MODEL_NAME", defaults.span_model_name),
        help="Span model directory under the S3 models prefix. Defaults to span_v1.",
    )
    parser.add_argument(
        "--relation-model-name",
        default=os.environ.get("RELATION_MODEL_NAME", defaults.relation_model_name),
        help="Relation model directory under the S3 models prefix. Defaults to relation_v1.",
    )
    parser.add_argument("--s3-models", default="")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("SAGEMAKER_INFERENCE_IMAGE_URI"),
        help="ECR image URI; defaults to SAGEMAKER_INFERENCE_IMAGE_URI.",
    )
    parser.add_argument(
        "--role-arn",
        default=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
        help="SageMaker execution role; defaults to SAGEMAKER_EXECUTION_ROLE_ARN.",
    )
    parser.add_argument("--local", action="store_true", help="Run the job in local Docker")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    args = parser.parse_args()
    return Config(
        where=args.where,
        instance_type=args.instance_type,
        batch_size=args.batch_size,
        span_model_name=args.span_model_name,
        relation_model_name=args.relation_model_name,
        s3_models=args.s3_models,
        aws_profile_name=args.profile,
        is_local=args.local,
        image_uri=args.image_uri or "",
        role_arn=args.role_arn or "",
    ), args.no_wait


def _require_config(config: Config) -> None:
    missing = []
    if not config.image_uri:
        missing.append("SAGEMAKER_INFERENCE_IMAGE_URI (or --image-uri)")
    if missing:
        raise ValueError(f"Missing required deployment configuration: {', '.join(missing)}")


# =============
# AWS Session
# =============


def _execution_role_arn(config: Config, aws_session) -> str:
    if config.role_arn:
        return config.role_arn
    account_id = aws_session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account_id}:role/{EXECUTION_ROLE_NAME}"


def _setup_session(config: Config):
    session_kwargs = {"region_name": config.aws_region}
    if config.aws_profile_name:
        session_kwargs["profile_name"] = config.aws_profile_name
    aws_session = boto_session.Session(**session_kwargs)
    return aws_session


def _sagemaker_session(config: Config, aws_session):
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=aws_session)
        sagemaker_session.config = {"local": {"local_code": True}}
    else:
        sagemaker_session = session.Session(boto_session=aws_session)
    return sagemaker_session


# =============
# Main
# =============


def main():
    # Load config + logging
    config, no_wait = _parse_args()
    _require_config(config)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("launch_inference")

    # AWS Session
    aws_session = _setup_session(config)
    config.role_arn = _execution_role_arn(config, aws_session)
    sagemaker_session = _sagemaker_session(config, aws_session)

    # Set up job
    now = datetime.now(UTC)
    job_name = f"{config.job_prefix}-{now:%Y%m%d-%H%M%S}"
    output_uri = config.output_uri(job_name, f"{now:%Y-%m-%d}")

    processor = Processor(
        role=config.role_arn,
        image_uri=config.image_uri,
        instance_count=1,
        instance_type="local" if config.is_local else config.instance_type,
        volume_size_in_gb=config.volume_size_gb,
        max_runtime_in_seconds=config.max_runtime_hours * 3600,
        base_job_name=config.job_prefix,
        sagemaker_session=sagemaker_session,
        tags=config.tags,
    )

    logger.info("Starting %s on %s", job_name, config.instance_type)
    logger.info("Image: %s", config.image_uri)
    logger.info("Models: %s", config.s3_models)
    logger.info("Output: %s", output_uri)
    logger.info("WHERE: %s", config.where or "<none>")

    # Run job
    processor.run(
        job_name=job_name,
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
                destination=output_uri,
                s3_upload_mode="Continuous",
            ),
        ],
        arguments=config.pipeline_args(),
        wait=not no_wait,
        logs=not no_wait,
    )

    logger.info("Submitted %s", job_name)


if __name__ == "__main__":
    main()
