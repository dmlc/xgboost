import argparse
import copy

import boto3

from metadata import AMI_ID, COMMON_STACK_PARAMS, STACK_PARAMS

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


def format_params(args, *, stack_id):
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
    params.update(COMMON_STACK_PARAMS)
    return [{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()]


def get_full_stack_id(stack_id):
    return f"buildkite-{stack_id}-autoscaling-group"


def main(args):
    client = boto3.client("cloudformation", region_name=args.aws_region)

    for stack_id in AMI_ID:
        stack_id_full = get_full_stack_id(stack_id)
        print(f"Creating elastic CI stack {stack_id_full}...")

        params = format_params(args, stack_id=stack_id)

        response = client.create_stack(
            StackName=stack_id_full,
            TemplateURL=TEMPLATE_URL,
            Capabilities=[
                "CAPABILITY_IAM",
                "CAPABILITY_NAMED_IAM",
                "CAPABILITY_AUTO_EXPAND",
            ],
            OnFailure="ROLLBACK",
            EnableTerminationProtection=False,
            Parameters=params,
        )
        print(f"CI stack {stack_id_full} is in progress in the background")

    for stack_id in AMI_ID:
        stack_id_full = get_full_stack_id(stack_id)
        waiter = client.get_waiter("stack_create_complete")
        waiter.wait(StackName=stack_id_full)
        print(f"CI stack {stack_id_full} is now finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument("--agent-token", type=str, required=True)
    args = parser.parse_args()
    main(args)
