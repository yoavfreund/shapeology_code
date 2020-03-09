#!/usr/bin/env bash

# This file is a part of demonstrating using `run-cluster.py`
# to execute a script on all instances in one cluster.
#
# On an m3.xlarge instance, it formats and mounts the SSD attached
# to the instance. In addition, it clone the GitHub repository to
# a local directory.


# Set variables
export BASE_PATH="/home/ubuntu/workspace"
export GIT_REPO="https://github.com/arapat/rust-tmsn.git"
export GIT_BRANCH="master"
echo "export EDITOR=vim" >> ~/.bashrc

# Install tools
(sudo apt-get update; sudo apt-get install -y awscli cargo) &

# Set up disks: two SSDs, RAID
sudo umount /mnt
sudo umount $BASE_PATH
yes | sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/xvdb /dev/xvdc
yes | sudo mkfs.ext4 -L MY_DISK /dev/md0
rm -rf $BASE_PATH
mkdir -p $BASE_PATH
sudo mount LABEL=MY_DISK $BASE_PATH
sudo chown -R ubuntu $BASE_PATH

# Download software and data
cd $BASE_PATH
git clone $GIT_REPO

# Wait for the background processes to be finished
wait