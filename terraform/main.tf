provider "aws" {
    region     = var.region
    access_key = var.access_key
    secret_key = var.secret_key
}

resource "aws_vpc" "dentvpc" {
    cidr_block = "10.0.0.0/16"
    tags = {Name = "Dentai-VPC" }
}

resource "aws_subnet" "dentsubnet" {
    vpc_id = aws_vpc.dentvpc.id
    cidr_block = "10.0.1.0/24"
    map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "gw" {
    vpc_id = aws_vpc.dentvpc.id
}

resource "aws_route_table" "art" {
    vpc_id = aws_vpc.dentvpc.id
    
    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = aws_internet_gateway.gw.id
    }
}

resource "aws_route_table_association" "arta" {
    subnet_id = aws_subnet.dentsubnet.id
    route_table_id = aws_route_table.art.id
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
    
}

resource "aws_security_group" "main" {
    egress = [
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            self             = false
            security_groups  = []
            from_port        = 0
            protocol         = "-1"
            to_port          = 0
        }
    ]
    ingress = [
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            self             = false
            security_groups  = []
            from_port        = 80
            protocol         = "tcp"
            to_port          = 80
        },
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            self             = false
            security_groups  = []
            from_port        = 22
            protocol         = "tcp"
            to_port          = 22
        },
        {
            cidr_blocks      = [ "0.0.0.0/0", ]
            description      = ""
            ipv6_cidr_blocks = []
            prefix_list_ids  = []
            self             = false
            security_groups  = []
            from_port        = 5000
            protocol         = "tcp"
            to_port          = 5000
        }
    ]
}
