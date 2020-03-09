#!/usr/bin/env python
import argparse
import os

from operator import itemgetter
from lib.common import load_config
from lib.common import query_status

credential_format = """[default]
aws_access_key_id = {}
aws_secret_access_key = {}
"""

def setup_cluster(args):
    all_status = query_status(args)
    if len(all_status) == 0:
        print("No instance found in the cluster '{}'. Quit.".format(args["name"]))
        return
    print("{} clusters found with the name '{}'.".format(len(all_status), args["name"]))

    for idx, status in enumerate(all_status):
        total = len(status)
        if total == 0:
            continue
        print("\nCluster {}:".format(idx + 1))
        ready = sum(t[0] == "running" for t in status)
        neighbors = list(map(itemgetter(1), status))
        print("    Total instances: {}\n    Running: {}".format(total, ready))
        if ready == 0:
            print("    Instances status: {}".format(status[0][0]))
            continue
        print("Setting up the cluster...")
        urls = [t[1] for t in status if t[0] == "running"]
        for url in urls:
            print("Prepare to set up the instance `{}`".format(url))
            os.system("echo '{}' | ssh -o \"StrictHostKeyChecking no\"  "
                      "-i {} ubuntu@{} \"mkdir .aws && cat > .aws/credentials\"".format(
                credential_format.format(args["aws_access_key_id"], args["aws_secret_access_key"]),
                args["key_path"],
                url,
            ))
            os.system("ssh -o \"StrictHostKeyChecking no\"  "
                      "-i {} ubuntu@{} \"sudo apt-get install -y awscli\"".format(
                args["key_path"],
                url,
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup an existing cluster")
    parser.add_argument("--name",
                        required=True,
                        help="cluster name")
    parser.add_argument("--region",
                        help="Region name")
    parser.add_argument("--credential",
                        help="path to the credential file")
    config = load_config(vars(parser.parse_args()))
    setup_cluster(config)
