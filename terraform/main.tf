provider "aws" {
    region     = var.region
    access_key = var.access_key
    secret_key = var.secret_key
}

data "aws_ami" "ubuntu" {
    most_recent = true
    
    filter {
        name = "name"
        values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
    }
    
    owners = ["099720109477"]
}

resource "aws_instance" "dentai" {
    ami = data.aws_ami.ubuntu.id
    instance_type = var.instance
    key_name = var.public_ssh_key
    
    tags = {
        Name = var.name
    }
    
    vpc_security_group_ids = [aws_security_group.main.id]
    
    connection {
        type = "ssh"
        timeout = "4m"
        user = "ubuntu"
        host = self.public_ip
        private_key = file(var.private_ssh_key)
    }
    
}

resource "aws_security_group" "main" {
    egress = [
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            from_port        = 0
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            protocol         = "-1"
            security_groups  = []
            self             = false
            to_port          = 0
        }
    ]
    ingress = [
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            from_port        = 22
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            protocol         = "tcp"
            security_groups  = []
            self             = false
            to_port          = 22
        }
    ]
}
