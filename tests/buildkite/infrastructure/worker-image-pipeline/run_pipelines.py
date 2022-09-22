import argparse

import boto3
from create_worker_image_pipelines import get_full_stack_id
from metadata import IMAGE_PARAMS


def main(args):
    cf = boto3.resource("cloudformation", region_name=args.aws_region)
    builder_client = boto3.client("imagebuilder", region_name=args.aws_region)
    for stack_id in IMAGE_PARAMS:
        stack_id_full = get_full_stack_id(stack_id)
        pipeline_arn = cf.Stack(stack_id_full).Resource("Pipeline").physical_resource_id
        print(f"Running pipeline {pipeline_arn} to generate a new AMI...")
        r = builder_client.start_image_pipeline_execution(imagePipelineArn=pipeline_arn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-region", type=str, required=True)
    args = parser.parse_args()
    main(args)
