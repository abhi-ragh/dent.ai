variable "name" {
    description = "Project Name"
    type = string
}

variable "region" {
    description = "AWS region"
    type = string
}

variable "instance" {
    description = "AWS instance"
    type = string 
}

variable "access_key" {
    description = "AWS Access key"
    type = string
}

variable "secret_key" {
    description = "AWS secret key"
    type = string
}

variable "public_ssh_key" {
    description = "AWS public key"
    type = string
}
