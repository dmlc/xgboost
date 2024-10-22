packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = "~> 1"
    }
    windows-update = {
      version = "0.15.0"
      source  = "github.com/rgl/windows-update"
    }
  }
}

locals {
  ami_name_prefix = "xgboost-ci"
  image_name      = "RunsOn worker with Windows Server 2022 + ssh + CUDA driver"
  region          = "us-west-2"
  timestamp       = regex_replace(timestamp(), "[- TZ:]", "")
  volume_size     = 120
}

data "amazon-ami" "aws-windows-x64" {
  filters = {
    name                = "Windows_Server-2022-English-Full-Base-*"
    root-device-type    = "ebs"
    virtualization-type = "hvm"
  }
  most_recent = true
  owners      = ["amazon"]
}

source "amazon-ebs" "runs-on-windows" {
  source_ami                  = "${data.amazon-ami.aws-windows-x64.id}"
  ami_name                    = "${local.ami_name_prefix}-runs-on-windows-${local.timestamp}"
  ami_description             = "${local.image_name}"
  ami_regions                 = ["${local.region}"]
  ami_virtualization_type     = "hvm"
  associate_public_ip_address = true
  communicator                = "ssh"
  instance_type               = "g4dn.xlarge"
  region                      = "${local.region}"
  ssh_timeout                 = "10m"
  ssh_username                = "Administrator"
  ssh_file_transfer_method    = "sftp"
  user_data_file              = "setup_ssh.ps1"
  launch_block_device_mappings {
    device_name = "/dev/sda1"
    volume_size = "${local.volume_size}"
    volume_type = "gp3"
    delete_on_termination = true
  }
  aws_polling {   # Wait up to 2.5 hours until the AMI is ready
    delay_seconds = 15
    max_attempts = 600
  }
  fast_launch {
    enable_fast_launch = true
    target_resource_count = 10
  }
  snapshot_tags = {
    Name      = "${local.image_name}"
    BuildTime = "${local.timestamp}"
  }
  tags = {
    Name      = "${local.image_name}"
    BuildTime = "${local.timestamp}"
  }
}

build {
  sources = ["source.amazon-ebs.runs-on-windows"]

  provisioner "windows-update" {}

  provisioner "powershell" {
    script = "install_choco.ps1"
  }

  provisioner "windows-restart" {
    max_retries = 3
  }

  provisioner "powershell" {
    script = "bootstrap.ps1"
  }

  provisioner "powershell" {  # Sysprep should run the last
    script = "sysprep.ps1"
  }
}
