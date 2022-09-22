import argparse
import copy
import json
import os
from urllib.request import urlopen

import boto3
import cfn_flip
from metadata import IMAGE_PARAMS

current_dir = os.path.dirname(__file__)

BUILDKITE_CF_TEMPLATE_URL = (
    "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"
)


def format_params(*, stack_id, aws_region, ami_mapping):
    params = copy.deepcopy(IMAGE_PARAMS[stack_id])
    with open(
        os.path.join(current_dir, params["BootstrapScript"]),
        encoding="utf-8",
    ) as f:
        bootstrap_script = f.read()
    params["BaseImageId"] = ami_mapping[aws_region][params["BaseImageId"]]
    params["BootstrapScript"] = bootstrap_script
    return [{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()]


def get_ami_mapping():
    with urlopen(BUILDKITE_CF_TEMPLATE_URL) as response:
        buildkite_cf_template = response.read().decode("utf-8")
    cfn_obj = json.loads(cfn_flip.to_json(buildkite_cf_template))
    return cfn_obj["Mappings"]["AWSRegion2AMI"]


def get_full_stack_id(stack_id):
    return f"buildkite-{stack_id}-worker"


def main(args):
    with open(
        os.path.join(current_dir, "ec2-image-builder-pipeline-template.yml"),
        encoding="utf-8",
    ) as f:
        ec2_image_pipeline_template = f.read()

    ami_mapping = get_ami_mapping()

    for stack_id in IMAGE_PARAMS:
        stack_id_full = get_full_stack_id(stack_id)
        print(f"Creating EC2 image builder stack {stack_id_full}...")

        params = format_params(
            stack_id=stack_id, aws_region=args.aws_region, ami_mapping=ami_mapping
        )

        client = boto3.client("cloudformation", region_name=args.aws_region)
        response = client.create_stack(
            StackName=stack_id_full,
            TemplateBody=ec2_image_pipeline_template,
            Capabilities=[
                "CAPABILITY_IAM",
                "CAPABILITY_NAMED_IAM",
                "CAPABILITY_AUTO_EXPAND",
            ],
            OnFailure="ROLLBACK",
            EnableTerminationProtection=False,
            Parameters=params,
        )
        print(
            f"EC2 image builder stack {stack_id_full} is in progress in the background"
        )

    for stack_id in IMAGE_PARAMS:
        stack_id_full = get_full_stack_id(stack_id)
        waiter = client.get_waiter("stack_create_complete")
        waiter.wait(StackName=stack_id_full)
        print(f"EC2 image builder stack {stack_id_full} is now finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-region", type=str, required=True)
    args = parser.parse_args()
    main(args)
