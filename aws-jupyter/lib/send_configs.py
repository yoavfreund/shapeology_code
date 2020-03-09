#!/usr/bin/env python
import argparse
import os
import subprocess

from lib.common import load_config
from lib.common import check_connections


def check_exists(path):
    return os.path.isfile(path) 


def parse_file_path(path):
    return (path, path.rsplit('/', 1)[1])


def send_configs(args):
    if not check_exists(args["key_path"]):
        print("Error: File '{}' does not exist.".format(args["key_path"]))
        return
    if not check_exists(args["neighbors"]):
        print("Error: File '{}' does not exist.".format(args["neighbors"]))
        return
    if not os.path.isdir(args["config"]):
        print("Error: Directory '{}' does not exist.".format(args["config"]))
        return

    with open(args["neighbors"]) as f:
        status = f.readline()
        if status[0] != 'R':  # Not "Ready."
            print("Please run `check-cluster.py` first and "
                  "make sure all instances in the cluster is up and running.")
            return
        instances = [t.strip() for t in f if t.strip()]

    local_path = args["config"]
    configs = os.listdir(local_path)
    if len(configs) != len(instances):
        print("Error: The number of configuration files ({}) in '{}' does not equal to the number "
              "of instances ({}).".format(args["config"], len(configs), len(instances)))

    # Send the files
    key = args["key_path"]
    base_path = args["base_path"]
    remote_path = os.path.join(base_path, "configuration")
    if not check_connections(instances, args):
        return
    for url, config in zip(instances, configs):
        print("Sending the config file to '{}'".format(url))
        config_path = os.path.join(local_path, config)
        command = ("scp -o StrictHostKeyChecking=no -i {} {} ubuntu@{}:{}"
                   "").format(key, config_path, url, remote_path)
        subprocess.run(command, shell=True, check=True)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Send a different configuration file to each instance of a cluster")
    parser.add_argument("--config",
                        required=True,
                        help="Path of the directory that contains all configuration files")
    parser.add_argument("--credential",
                        help="path to the credential file")
    args = vars(parser.parse_args())
    args["neighbors"] = os.path.abspath("./neighbors.txt")
    args["base_path"] = "/home/ubuntu/workspace"
    config = load_config(args)
    send_configs(config)