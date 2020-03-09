#!/usr/bin/env python3

from lib.create_cluster import create_cluster
from lib.check_cluster import check_cluster
from lib.run_cluster import run_cluster
from lib.terminate_cluster import terminate_cluster
from lib.common import load_config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Jupyter notebook on a EC2 cluster")
    parser.add_argument(
        "task",
        nargs='?',
        help="Task to perform, should be one of 'config', 'create', 'check', 'terminate'")
    parser.add_argument("-c", "--count",
                        required=False,
                        help="the number of instances in the cluster")
    parser.add_argument("--name",
                        required=False,
                        default="aws-jupyter-default",
                        help="cluster name")
    parser.add_argument("--region",
                        help="region name")
    parser.add_argument("-t", "--type",
                        help="the type of the instances")
    parser.add_argument("--ondemand",
                        action='store_true',
                        help="use on-demand instances")
    parser.add_argument("--ami",
                         help="AMI type")
    parser.add_argument("--credential",
                        required=False,
                        help="path to the credential file")
    # --remote, --local, --script, --files, --output, --config
    config = load_config(vars(parser.parse_args()))
    if config["task"] == "config":
        print("Please set following configuration parameters."
              "Type Enter if the default value is correct.")
        print("Region [{}]: ".format(config["region"]), end='')
        s = input()
        if s.strip():
            config["region"] = s.strip()
        print("Instance type [{}]: ".format(config["type"]), end='')
        s = input()
        if s.strip():
            config["type"] = s.strip()
        print("AMI [{}]: ".format(config["ami"]), end='')
        s = input()
        if s.strip():
            config["ami"] = s.strip()
        print("Credential [{}]: ".format(config["credential"]), end='')
        s = input()
        if s.strip():
            config["credential"] = s.strip()
        load_config(config)
    elif config["task"] == "create":
        create_cluster(config)
    elif config["task"] == "check":
        status = check_cluster(config)
        if status is not None and 0 < status[0] and status[0] == status[1]:
            print("Cluster is running (yet might still being initialized). I will try to start Jupyter notebook now.")
            config["script"] = "./script-examples/install-project.sh"
            config["output"] = False
            run_cluster(config)
        elif status is not None and 0 < status[1]:
            print("Cluster is not ready. Please check again later.")
    elif config["task"] == "terminate":
        terminate_cluster(config)
    else:
        print("Error: Cannot reconize the task type '{}'.".format(config["task"]))
        exit(1)
