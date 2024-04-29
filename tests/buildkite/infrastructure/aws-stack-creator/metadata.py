AMI_ID = {
    # Managed by XGBoost team
    "linux-amd64-gpu": {
        "us-west-2": "ami-070080d04e81c5e39",
    },
    "linux-amd64-mgpu": {
        "us-west-2": "ami-070080d04e81c5e39",
    },
    "windows-gpu": {
        "us-west-2": "ami-07c14abcf529d816a",
    },
    "windows-cpu": {
        "us-west-2": "ami-07c14abcf529d816a",
    },
    # Managed by BuildKite
    # from https://s3.amazonaws.com/buildkite-aws-stack/latest/aws-stack.yml
    "linux-amd64-cpu": {
        "us-west-2": "ami-0180f7fb0f07eb0bc",
    },
    "pipeline-loader": {
        "us-west-2": "ami-0180f7fb0f07eb0bc",
    },
    "linux-arm64-cpu": {
        "us-west-2": "ami-00686bdc2043a5505",
    },
}

STACK_PARAMS = {
    "linux-amd64-gpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceTypes": "g4dn.xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "8",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-amd64-mgpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceTypes": "g4dn.12xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "1",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "windows-gpu": {
        "InstanceOperatingSystem": "windows",
        "InstanceTypes": "g4dn.2xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "2",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "windows-cpu": {
        "InstanceOperatingSystem": "windows",
        "InstanceTypes": "c5a.2xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "2",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-amd64-cpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceTypes": "c5a.4xlarge",
        "AgentsPerInstance": "1",
        "MinSize": "0",
        "MaxSize": "16",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "pipeline-loader": {
        "InstanceOperatingSystem": "linux",
        "InstanceTypes": "t3a.micro",
        "AgentsPerInstance": "1",
        "MinSize": "2",
        "MaxSize": "2",
        "OnDemandPercentage": "100",
        "ScaleOutFactor": "1.0",
        "ScaleInIdlePeriod": "60",  # in seconds
    },
    "linux-arm64-cpu": {
        "InstanceOperatingSystem": "linux",
        "InstanceTypes": "c6g.4xlarge",
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
