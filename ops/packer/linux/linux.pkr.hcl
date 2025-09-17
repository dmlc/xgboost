packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = "~> 1"
    }
  }
}

locals {
  ami_name_prefix = "xgboost-ci"
  image_name      = "RunsOn worker with Ubuntu 24.04 + CUDA driver"
  region          = "us-west-2"
  timestamp       = regex_replace(timestamp(), "[- TZ:]", "")
  volume_size     = 40
}

data "amazon-ami" "aws-ubuntu-x64" {
  filters = {
    name                = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
    root-device-type    = "ebs"
    virtualization-type = "hvm"
  }
  most_recent = true
  owners      = ["amazon"]
}

source "amazon-ebs" "runs-on-linux" {
  source_ami                  = "${data.amazon-ami.aws-ubuntu-x64.id}"
  ami_name                    = "${local.ami_name_prefix}-runs-on-linux-${local.timestamp}"
  ami_description             = "${local.image_name}"
  ami_regions                 = ["${local.region}"]
  ami_virtualization_type     = "hvm"
  associate_public_ip_address = true
  communicator                = "ssh"
  instance_type               = "g4dn.xlarge"
  region                      = "${local.region}"
  ssh_timeout                 = "10m"
  ssh_username                = "ubuntu"
  ssh_file_transfer_method    = "sftp"
  user_data_file              = "setup_ssh.sh"
  launch_block_device_mappings {
    device_name = "/dev/sda1"
    volume_size = "${local.volume_size}"
    volume_type = "gp3"
    delete_on_termination = true
  }
  aws_polling {   # Wait up to 1 hour until the AMI is ready
    delay_seconds = 15
    max_attempts = 240
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
  sources = ["source.amazon-ebs.runs-on-linux"]

  provisioner "shell" {
    script      = "install_drivers.sh"
    pause_after = "30s"
  }

  provisioner "shell" {
    expect_disconnect = true
    inline            = ["echo 'Reboot VM'", "sudo reboot"]
  }

  provisioner "shell" {
    pause_before = "1m0s"
    script       = "bootstrap.sh"
  }
}
