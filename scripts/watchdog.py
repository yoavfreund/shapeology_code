#!/usr/bin/env python3

from os.path import isfile,getmtime
from glob import glob
from time import sleep,time
from os import system

stack='s3://mousebraindata-open/MD657'
local_data='/dev/shm/data'
exec_dir='/home/ubuntu/shapeology_code/scripts'

def run(command):
    print('cmd=',command)
    system(command)

def Last_Modified(file_name):
    try:
        mtime = getmtime(file_name)
    except OSError:
        mtime = 0
    return(mtime)

if __name__=='__main__':
    Recent=False
    for logfile in glob(exec_dir+'/Controller*.log'):
        gap=time() - Last_Modified(logfile)
        if gap <600: # allow 10 minute idle
            print(logfile,'gap is %6.1f'%gap)
            Recent=True
            break
    if(not Recent):
        command='cd {0}; ./Controller.py {1} {2} &> Controller-{3}.log &'\
        .format(exec_dir,stack,local_data,int(time()))
        run(command)
