AMI_ID = {
    # Managed by XGBoost team
    "linux-amd64-gpu": {
        "us-east-2": "ami-0d3719bde86b92cec",
        "us-west-2": "ami-00ed92bd37f77bc33",
    },
    "linux-amd64-mgpu": {
        "us-east-2": "ami-0d3719bde86b92cec",
        "us-west-2": "ami-00ed92bd37f77bc33",
    },
    "windows-gpu": {
        "us-east-2": "ami-01ff825c3ec5cc672",
        "us-west-2": "ami-0a1a2ea551a07ad5f",
    },
    # Managed by BuildKite
    "linux-amd64-cpu": {
        "us-east-2": "ami-00f6d034cc4ccc18b",
        "us-west-2": "ami-075d4c25d5f0c17c1",
    },
    "pipeline-loader": {
        "us-east-2": "ami-00f6d034cc4ccc18b",
        "us-west-2": "ami-075d4c25d5f0c17c1",
    },
    "linux-arm64-cpu": {
        "us-east-2": "ami-0e2269f2c64400c76",
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
        "MaxSize": "4",
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
        "ScaleInIdlePeriod": "600",  # in seconds
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
    "ManagedPolicyARN": "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "EnableSecretsPlugin": "false",
    "EnableECRPlugin": "false",
    "EnableDockerLoginPlugin": "false",
    "EnableDockerUserNamespaceRemap": "false",
    "BuildkiteAgentExperiments": "normalised-upload-paths,resolve-commit-after-checkout",
}
