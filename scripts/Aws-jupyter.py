# from aws_jupyter.credentials import check_access
# check_access({})

from aws_jupyter.create_cluster import create_cluster

name = "t2"
create_cluster({
    "count": 1,
    "name": name,
    "type": "r5.xlarge",
    "ami": "ami-052aabb224c68baf8",
    "spot": 1.0,
})

from aws_jupyter.check_cluster import check_cluster
from aws_jupyter.common import load_config
from aws_jupyter.run_cluster import run_cluster
from time import sleep

def check_status(args):
    args = load_config(args)
    status = check_cluster(args)
    ready, total = 0, 0
    if status is not None:
        ready, total = status
    if ready > 0 and ready == total:
        print("Cluster is running (yet might still being initialized). \
                        I will try to start Jupyter notebook now.")
        args["files"] = ["./neighbors.txt"]
        is_worked = run_cluster(args, first_node_only=False)
        if not is_worked:
            return False
    elif total > 0:
        print("Cluster is not ready. Please check again later.")
        return False
    return True


while not check_status({"name": name,
                        "script": "/Users/kuiqian/Github/shapeology_code/install-project.sh",
                        "output": False,
                        }):
    sleep(30)

# from aws_jupyter.run_cluster import run_cluster
# run_cluster({
#     "script": "/Users/kuiqian/Github/shapeology_code/install-project.sh",
#     "output": False,
# })


# from aws_jupyter.terminate_cluster import terminate_cluster
#
# terminate_cluster({
#     "name": "t1",
# })