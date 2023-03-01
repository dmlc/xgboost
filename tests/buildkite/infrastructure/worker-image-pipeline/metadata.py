IMAGE_PARAMS = {
    "linux-amd64-gpu": {
        "BaseImageId": "linuxamd64",
        # AMI ID is looked up from Buildkite's CloudFormation template
        "BootstrapScript": "linux-amd64-gpu-bootstrap.yml",
        "InstanceType": "g4dn.xlarge",
        "InstanceOperatingSystem": "Linux",
        "VolumeSize": "40",  # in GiBs
    },
    "windows-gpu": {
        "BaseImageId": "windows",
        # AMI ID is looked up from Buildkite's CloudFormation template
        "BootstrapScript": "windows-gpu-bootstrap.yml",
        "InstanceType": "g4dn.2xlarge",
        "InstanceOperatingSystem": "Windows",
        "VolumeSize": "120",  # in GiBs
    },
}
