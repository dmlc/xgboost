import argparse
import copy
import os
import re

import boto3
import botocore
from metadata import AMI_ID, COMMON_STACK_PARAMS, STACK_PARAMS

current_dir = os.path.dirname(__file__)

TEMPLATE_URL = "https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml"


def get_availability_zones(*, aws_region):
    client = boto3.client("ec2", region_name=aws_region)
    r = client.describe_availability_zones(
        Filters=[
            {"Name": "region-name", "Values": [aws_region]},
            {"Name": "zone-type", "Values": ["availability-zone"]},
        ]
    )
    return sorted([x["ZoneName"] for x in r["AvailabilityZones"]])


def get_default_vpc(*, aws_region):
    ec2 = boto3.resource("ec2", region_name=aws_region)
    default_vpc_id = None
    for x in ec2.vpcs.filter(Filters=[{"Name": "is-default", "Values": ["true"]}]):
        return x

    # Create default VPC if not exist
    client = boto3.client("ec2", region_name=aws_region)
    r = client.create_default_vpc()
    default_vpc_id = r["Vpc"]["VpcId"]

    return ec2.Vpc(default_vpc_id)


def format_params(args, *, stack_id, agent_iam_policy):
    default_vpc = get_default_vpc(aws_region=args.aws_region)
    azs = get_availability_zones(aws_region=args.aws_region)
    # For each of the first two availability zones (AZs), choose the default subnet
    subnets = [
        x.id
        for x in default_vpc.subnets.filter(
            Filters=[
                {"Name": "default-for-az", "Values": ["true"]},
                {"Name": "availability-zone", "Values": azs[:2]},
            ]
        )
    ]
    assert len(subnets) == 2

    params = copy.deepcopy(STACK_PARAMS[stack_id])
    params["ImageId"] = AMI_ID[stack_id][args.aws_region]
    params["BuildkiteQueue"] = stack_id
    params["CostAllocationTagValue"] = f"buildkite-{stack_id}"
    params["BuildkiteAgentToken"] = args.agent_token
    params["VpcId"] = default_vpc.id
    params["Subnets"] = ",".join(subnets)
    params["ManagedPolicyARN"] = agent_iam_policy
    params.update(COMMON_STACK_PARAMS)
    return [{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()]


def get_full_stack_id(stack_id):
    return f"buildkite-{stack_id}-autoscaling-group"


def stack_exists(args, *, stack_name):
    client = boto3.client("cloudformation", region_name=args.aws_region)
    waiter = client.get_waiter("stack_exists")
    try:
        waiter.wait(StackName=stack_name, WaiterConfig={"MaxAttempts": 1})
        return True
    except botocore.exceptions.WaiterError as e:
        return False


def create_or_update_stack(
    args, *, stack_name, template_url=None, template_body=None, params=None
):
    kwargs = {
        "StackName": stack_name,
        "Capabilities": [
            "CAPABILITY_IAM",
            "CAPABILITY_NAMED_IAM",
            "CAPABILITY_AUTO_EXPAND",
        ],
    }
    if template_url:
        kwargs["TemplateURL"] = template_url
    if template_body:
        kwargs["TemplateBody"] = template_body
    if params:
        kwargs["Parameters"] = params

    client = boto3.client("cloudformation", region_name=args.aws_region)

    if stack_exists(args, stack_name=stack_name):
        print(f"Stack {stack_name} already exists. Updating...")
        try:
            response = client.update_stack(**kwargs)
            return {"StackName": stack_name, "Action": "update"}
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ValidationError" and re.search(
                "No updates are to be performed", e.response["Error"]["Message"]
            ):
                print(f"No update was made to {stack_name}")
                return {"StackName": stack_name, "Action": "noop"}
            else:
                raise e
    else:
        kwargs.update({"OnFailure": "ROLLBACK", "EnableTerminationProtection": False})
        response = client.create_stack(**kwargs)
        return {"StackName": stack_name, "Action": "create"}


def wait(promise):
    client = boto3.client("cloudformation", region_name=args.aws_region)
    stack_name = promise["StackName"]
    print(f"Waiting for {stack_name}...")
    if promise["Action"] == "create":
        waiter = client.get_waiter("stack_create_complete")
        waiter.wait(StackName=stack_name)
        print(f"Finished creating stack {stack_name}")
    elif promise["Action"] == "update":
        waiter = client.get_waiter("stack_update_complete")
        waiter.wait(StackName=stack_name)
        print(f"Finished updating stack {stack_name}")
    elif promise["Action"] != "noop":
        raise ValueError(f"Invalid promise {promise}")


def create_agent_iam_policy(args):
    policy_stack_name = "buildkite-agent-iam-policy"
    print(f"Creating stack {policy_stack_name} for agent IAM policy...")
    with open(
        os.path.join(current_dir, "agent-iam-policy-template.yml"),
        encoding="utf-8",
    ) as f:
        policy_template = f.read()
    promise = create_or_update_stack(
        args, stack_name=policy_stack_name, template_body=policy_template
    )
    wait(promise)

    cf = boto3.resource("cloudformation", region_name=args.aws_region)
    policy = cf.StackResource(policy_stack_name, "BuildkiteAgentManagedPolicy")
    return policy.physical_resource_id


def main(args):
    agent_iam_policy = create_agent_iam_policy(args)

    client = boto3.client("cloudformation", region_name=args.aws_region)

    promises = []

    for stack_id in AMI_ID:
        stack_id_full = get_full_stack_id(stack_id)
        print(f"Creating elastic CI stack {stack_id_full}...")

        params = format_params(
            args, stack_id=stack_id, agent_iam_policy=agent_iam_policy
        )

        promise = create_or_update_stack(
            args, stack_name=stack_id_full, template_url=TEMPLATE_URL, params=params
        )
        promises.append(promise)
        print(f"CI stack {stack_id_full} is in progress in the background")

    for promise in promises:
        wait(promise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument("--agent-token", type=str, required=True)
    args = parser.parse_args()
    main(args)
