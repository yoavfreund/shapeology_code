#!/usr/bin/env python3

from os.path import isfile,getmtime
from glob import glob
from time import sleep,time
from os import system
from subprocess import Popen

stack='s3://mousebraindata-open/MD657'
local_data='/dev/shm/data'
exec_dir='/home/ubuntu/shapeology_code/scripts'

def run(command,out):`
    print('cmd=',command,'out=',out)
    outfile=open(out,'w')
    Popen(command.split(),stdout=outfile,stderr=outfile)

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
        if gap <120: # allow 2 minute idle
            print(logfile,'gap is %6.1f'%gap)
            Recent=True
            break
    if(not Recent):
        command='{0}/Controller.py {0} {1} {2}'\
        .format(exec_dir,stack,local_data)
        output='{0}/Controller-{1}.log'.format(exec_dir,int(time()))
        run(command,output)
