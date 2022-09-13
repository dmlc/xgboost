import argparse
import os

import boto3

current_dir = os.path.dirname(__file__)


def main(args):
    with open(
        os.path.join(current_dir, "service-user-template.yml"), encoding="utf-8"
    ) as f:
        service_user_template = f.read()

    stack_id = "buildkite-elastic-ci-stack-service-user"

    print("Create a new IAM user with suitable permissions...")
    client = boto3.client("cloudformation", region_name=args.aws_region)
    response = client.create_stack(
        StackName=stack_id,
        TemplateBody=service_user_template,
        Capabilities=[
            "CAPABILITY_IAM",
            "CAPABILITY_NAMED_IAM",
        ],
        Parameters=[{"ParameterKey": "UserName", "ParameterValue": args.user_name}],
    )
    waiter = client.get_waiter("stack_create_complete")
    waiter.wait(StackName=stack_id)
    user = boto3.resource("iam", region_name=args.aws_region).User(args.user_name)
    key_pair = user.create_access_key_pair()
    print("Finished creating an IAM users with suitable permissions.")
    print(f"Access Key ID: {key_pair.access_key_id}")
    print(f"Access Secret Access Key: {key_pair.secret_access_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument(
        "--user-name", type=str, default="buildkite-elastic-ci-stack-user"
    )
    args = parser.parse_args()
    main(args)
