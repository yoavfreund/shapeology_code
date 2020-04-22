from aws_jupyter.credentials import check_access
check_access({})

from aws_jupyter.create_cluster import create_cluster

create_cluster({
    "count": 20,
    "name": "t1",
    "type": "r5.xlarge",
    "ami": "ami-052aabb224c68baf8",
    "spot": 1.0,
})

from aws_jupyter.check_cluster import check_status_and_init
from time import sleep

while not check_status_and_init({"name": "t1"}):
    sleep(2)
