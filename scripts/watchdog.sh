#!/usr/bin/env bash
source /home/ubuntu/.bashrc
echo 'this is watchdog'  >> /home/ubuntu/watchdog.log
echo $PATH  >> /home/ubuntu/watchdog.log
watchdog.py >> /home/ubuntu/watchdog.log
