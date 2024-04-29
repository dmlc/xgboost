BuildKite CI Infrastructure
===========================

# Worker image builder (`worker-image-pipeline/`)

Use EC2 Image Builder to build machine images in a deterministic fashion.
The machine images are used to initialize workers in the CI/CD pipelines.

## Editing bootstrap scripts

Currently, we create two pipelines for machine images: one for Linux workers and another
for Windows workers.
You can edit the bootstrap scripts to change how the worker machines are initialized.

* `linux-amd64-gpu-bootstrap.yml`: Bootstrap script for Linux worker machines
* `windows-gpu-bootstrap.yml`: Bootstrap script for Windows worker machines

## Creating and running Image Builder pipelines

Run the following commands to create and run pipelines in EC2 Image Builder service:
```bash
python worker-image-pipeline/create_worker_image_pipelines.py --aws-region us-west-2
python worker-image-pipeline/run_pipelines.py --aws-region us-west-2
```
Go to the AWS CloudFormation console and verify the existence of two CloudFormation stacks:
* `buildkite-windows-gpu-worker`
* `buildkite-linux-amd64-gpu-worker`

Then go to the EC2 Image Builder console to check the status of the image builds. You may
want to inspect the log output should a build fails.
Once the new machine images are done building, see the next section to deploy the new
images to the worker machines.

# Elastic CI Stack for AWS (`aws-stack-creator/`)

Use EC2 Autoscaling groups to launch worker machines in EC2. BuildKite periodically sends
messages to the Autoscaling groups to increase or decrease the number of workers according
to the number of outstanding testing jobs.

## Deploy an updated CI stack with new machine images

First, edit `aws-stack-creator/metadata.py` to update the `AMI_ID` fields:
```python
AMI_ID = {
    # Managed by XGBoost team
    "linux-amd64-gpu": {
        "us-west-2": "...",
    },
    "linux-amd64-mgpu": {
        "us-west-2": "...",
    },
    "windows-gpu": {
        "us-west-2": "...",
    },
    "windows-cpu": {
        "us-west-2": "...",
    },
    # Managed by BuildKite
    # from https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml
    "linux-amd64-cpu": {
        "us-west-2": "...",
    },
    "pipeline-loader": {
        "us-west-2": "...",
    },
    "linux-arm64-cpu": {
        "us-west-2": "...",
    },
}
```
AMI IDs uniquely identify the machine images in the EC2 service.
Go to the EC2 Image Builder console to find the AMI IDs for the new machine images
(see the previous section), and update the following fields:

* `AMI_ID["linux-amd64-gpu"]["us-west-2"]`:
  Use the latest output from the `buildkite-linux-amd64-gpu-worker` pipeline
* `AMI_ID["linux-amd64-mgpu"]["us-west-2"]`:
  Should be identical to `AMI_ID["linux-amd64-gpu"]["us-west-2"]`
* `AMI_ID["windows-gpu"]["us-west-2"]`:
  Use the latest output from the `buildkite-windows-gpu-worker` pipeline
* `AMI_ID["windows-cpu"]["us-west-2"]`:
  Should be identical to  `AMI_ID["windows-gpu"]["us-west-2"]`

Next, visit https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml
to look up the AMI IDs for the following fields:

* `AMI_ID["linux-amd64-cpu"]["us-west-2"]`: Copy and paste the AMI ID from the field
  `Mappings/AWSRegion2AMI/us-west-2/linuxamd64`
* `AMI_ID["pipeline-loader"]["us-west-2"]`:
   Should be identical to `AMI_ID["linux-amd64-cpu"]["us-west-2"]`
* `AMI_ID["linux-arm64-cpu"]["us-west-2"]`: Copy and paste the AMI ID from the field
  `Mappings/AWSRegion2AMI/us-west-2/linuxarm64`

Finally, run the following commands to deploy the new machine images:
```
python aws-stack-creator/create_stack.py --aws-region us-west-2 --agent-token AGENT_TOKEN
```
Go to the AWS CloudFormation console and verify the existence of the following
CloudFormation stacks:
* `buildkite-pipeline-loader-autoscaling-group`
* `buildkite-linux-amd64-cpu-autoscaling-group`
* `buildkite-linux-amd64-gpu-autoscaling-group`
* `buildkite-linux-amd64-mgpu-autoscaling-group`
* `buildkite-linux-arm64-cpu-autoscaling-group`
* `buildkite-windows-cpu-autoscaling-group`
* `buildkite-windows-gpu-autoscaling-group`
