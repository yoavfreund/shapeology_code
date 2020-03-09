#!/usr/bin/env python
import argparse
import subprocess
import json

from lib.common import load_config
from lib.common import query_status
from lib.common import DEFAULT_AMI


DEFAULT_TYPE = "m3.xlarge"


def create_cluster(args):
    all_status = query_status(args)
    if len(all_status):
        print("Error: A cluster with the name '{}' exists. ".format(args["name"]) +
              "Please choose a different cluster name.\n" +
              "Note: If you want to check the status of the cluster '{}', ".format(args["name"]) +
              "please use `./aws-jupyter.py check` or `./check-cluster.py`.")
        return
    credential = 'AWS_ACCESS_KEY_ID="{}" AWS_SECRET_ACCESS_KEY="{}"'.format(
        args["aws_access_key_id"], args["aws_secret_access_key"])
    create_command = """
    {} aws ec2 run-instances \
        --region {} \
        --image-id {} \
        --count {} \
        --instance-type {} \
        --key-name {} \
        --tag-specifications 'ResourceType=instance,Tags=[{{Key=cluster-name,Value={}}}]' \
        --associate-public-ip-address \
        --block-device-mappings \
            '[{{\"DeviceName\":\"/dev/xvdb\",\"VirtualName\":\"ephemeral0\"}}, \
              {{\"DeviceName\":\"/dev/xvdc\",\"VirtualName\":\"ephemeral1\"}}]' \
        --no-dry-run
    """.format(
        credential,
        args["region"],
        args["ami"],
        args["count"],
        args["type"],
        args["key"],
        args["name"]
    )
    if not args["ondemand"]:
        create_command = create_command.strip() + \
            """ --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice='3.0'}'"""
        print("We will use spot instances.")
    else:
        print("We will use on-demand instances.")
    print("Creating the cluster...")
    p = subprocess.run(create_command, shell=True, check=True, stdout=subprocess.PIPE)
    output = json.loads(p.stdout)
    print("Launched instances:")
    for instance in output["Instances"]:
        if args["ondemand"]:
            print("{} (on demand)".format(instance["InstanceId"]))
        else:
            print("{} ({})".format(instance["InstanceId"], instance["InstanceLifecycle"]))
    print()

    setup_security_group = """
    {} aws ec2 authorize-security-group-ingress \
        --region {} \
        --group-name default \
        --protocol tcp \
        --port 8888 \
        --cidr 0.0.0.0/0;
    {} aws ec2 authorize-security-group-ingress \
        --region {} \
        --group-name default \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0;
    """.format(credential, args["region"], credential, args["region"])
    print("Setting up security group...")
    subprocess.run(setup_security_group, shell=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crate a cluster using AWS spot instances")
    parser.add_argument("-c", "--count",
                        required=True,
                        help="the number of instances in the cluster")
    parser.add_argument("--name",
                        required=True,
                        help="cluster name")
    parser.add_argument("-t", "--type",
                        help="the type of the instances")
    parser.add_argument("--region",
                        help="Region name")
    parser.add_argument("--ami",
                         help="AMI type")
    parser.add_argument("--credential",
                        help="path to the credential file")
    args = vars(parser.parse_args())
    if args["ami"] is None:
        print("AMI is not specified. Default AMI set to '{}'".format(DEFAULT_AMI))
        args["ami"] = DEFAULT_AMI
    if args["type"] is None:
        print("Instance type is not specified. Default instance type set to '{}'".format(
            DEFAULT_TYPE))
        args["type"] = DEFAULT_TYPE
    config = load_config(args)
    create_cluster(config)
