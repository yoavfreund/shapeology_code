#!/usr/bin/env python
import argparse
import os
import subprocess

from lib.common import load_config
from lib.common import check_connections


def check_exists(path):
    return os.path.isfile(path)


def retrieve_file(args):
    if not check_exists(args["key_path"]):
        print("Error: File '{}' does not exist.".format(args["key_path"]))
        return
    if not check_exists(args["neighbors"]):
        print("Error: File '{}' does not exist.".format(args["neighbors"]))
        return

    with open(args["neighbors"]) as f:
        status = f.readline()
        if status[0] != 'R':  # Not "Ready."
            print("Please run `check-cluster.py` first and "
                  "make sure all instances in the cluster is up and running.")
            return
        instances = [t.strip() for t in f if t.strip()]

    # Retrieve the files
    local_dir = args["local"]
    remote_files = args["remote"]
    key = args["key_path"]
    commands = []
    if not check_connections(instances, args):
        return
    for idx, url in enumerate(instances):
        local_path = os.path.join(local_dir, "worker-{}".format(idx))
        command = "mkdir -p {}".format(local_path)
        subprocess.run(command, shell=True, check=True)
        for filepath in remote_files:
            commands.append(("scp -o StrictHostKeyChecking=no -i {} ubuntu@{}:{} {}"
                             "").format(key, url, filepath, local_path))
    command = " & ".join(commands)
    subprocess.run(command, shell=True, check=True)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieve the files from the instances of a cluster")
    parser.add_argument("--remote",
                        required=True,
                        nargs='+',
                        help="Path of the remote files to be downloaded. "
                             "For multiple files, separate them using spaces")
    parser.add_argument("--local",
                        required=True,
                        help="Path of the local directory to download the remote files")
    parser.add_argument("--credential",
                        help="path to the credential file")
    args = vars(parser.parse_args())
    args["neighbors"] = os.path.abspath("./neighbors.txt")
    config = load_config(args)
    retrieve_file(config)