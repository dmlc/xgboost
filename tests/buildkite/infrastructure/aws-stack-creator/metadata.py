AMI_ID = {
    # Managed by XGBoost team
    "linux-amd64-gpu": {
        "us-west-2": "ami-00ed92bd37f77bc33",
    },
    "linux-amd64-mgpu": {
        "us-west-2": "ami-00ed92bd37f77bc33",
    },
    "windows-gpu": {
        "us-west-2": "ami-0a1a2ea551a07ad5f",
    },
    "windows-cpu": {
        "us-west-2": "ami-0a1a2ea551a07ad5f",
    },
    # Managed by BuildKite
    # from https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml
    "linux-amd64-cpu": {
        "us-west-2": "ami-075d4c25d5f0c17c1",
    },
    "pipeline-loader": {
        "us-west-2": "ami-075d4c25d5f0c17c1",
    },
    "linux-arm64-cpu": {
        "us-west-2": "ami-0952c6fb6db9a9891",
    },
}

STACK_PARAMS = {
    "linux-amd64-gpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceType": "g4dn.xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "8",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-amd64-mgpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceType": "g4dn.12xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "1",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "windows-gpu": {
        "InstanceOperatingSystem": "windows",
        "InstanceType": "g4dn.2xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "2",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "windows-cpu": {
        "InstanceOperatingSystem": "windows",
        "InstanceType": "c5a.2xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "2",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-amd64-cpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceType": "c5a.4xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "16",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "pipeline-loader": {
        "InstanceOperatingSystem": "linux",
        "InstanceType": "t3a.micro",
        "AgentsPerInstance": "1",
        "MinSize": "1",
        "MaxSize": "1",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-arm64-cpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceType": "c6g.4xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "8",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
}

COMMON_STACK_PARAMS = {
    "BuildkiteAgentTimestampLines": "false",
    "BuildkiteWindowsAdministrator": "true",
    "AssociatePublicIpAddress": "true",
    "ScaleOutForWaitingJobs": "false",
    "EnableCostAllocationTags": "true",
    "CostAllocationTagName": "CreatedBy",
    "ECRAccessPolicy": "full",
    "EnableSecretsPlugin": "false",
    "EnableECRPlugin": "false",
    "EnableDockerLoginPlugin": "false",
    "EnableDockerUserNamespaceRemap": "false",
    "BuildkiteAgentExperiments": "normalised-upload-paths,resolve-commit-after-checkout",
}
