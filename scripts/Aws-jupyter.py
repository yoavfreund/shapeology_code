# from aws_jupyter.credentials import check_access
# check_access({})

from aws_jupyter.create_cluster import create_cluster

create_cluster({
    "count": 50,
    "name": "t1",
    "type": "r5.xlarge",
    "ami": "ami-052aabb224c68baf8",
    "spot": 1.0,
})

from aws_jupyter.check_cluster import check_cluster
from time import sleep

def check_status(args):
    status = check_cluster(args)
    ready, total = 0, 0
    if status is not None:
        ready, total = status
    if ready > 0 and ready == total:
        return True
    else:
        return False


while not check_status({"name": "t1"}):
    sleep(30)

from aws_jupyter.run_cluster import run_cluster
run_cluster({
    "script": "/Users/kuiqian/Github/shapeology_code/install-project.sh",
    "output": False,
})


# from aws_jupyter.terminate_cluster import terminate_cluster
#
# terminate_cluster({
#     "name": "t1",
# })