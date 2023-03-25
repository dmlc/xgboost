import re

import boto3
import botocore


def stack_exists(args, *, stack_name):
    client = boto3.client("cloudformation", region_name=args.aws_region)
    waiter = client.get_waiter("stack_exists")
    try:
        waiter.wait(StackName=stack_name, WaiterConfig={"MaxAttempts": 1})
        return True
    except botocore.exceptions.WaiterError as e:
        return False


def create_or_update_stack(
    args, *, client, stack_name, template_url=None, template_body=None, params=None
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


def replace_stack(
    args, *, client, stack_name, template_url=None, template_body=None, params=None
):
    """Delete an existing stack and create a new stack with identical name"""

    if not stack_exists(args, stack_name=stack_name):
        raise ValueError(f"Stack {stack_name} does not exist")
    r = client.delete_stack(StackName=stack_name)
    delete_waiter = client.get_waiter("stack_delete_complete")
    delete_waiter.wait(StackName=stack_name)

    kwargs = {
        "StackName": stack_name,
        "Capabilities": [
            "CAPABILITY_IAM",
            "CAPABILITY_NAMED_IAM",
            "CAPABILITY_AUTO_EXPAND",
        ],
        "OnFailure": "ROLLBACK",
        "EnableTerminationProtection": False,
    }
    if template_url:
        kwargs["TemplateURL"] = template_url
    if template_body:
        kwargs["TemplateBody"] = template_body
    if params:
        kwargs["Parameters"] = params
    response = client.create_stack(**kwargs)
    return {"StackName": stack_name, "Action": "create"}


def wait(promise, *, client):
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
