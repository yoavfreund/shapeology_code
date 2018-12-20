#!/usr/bin/env python3

from glob import glob
from time import sleep,time
from os import system
from subprocess import Popen,PIPE
from lib.utils import run,runPipe, Last_Modified

stack='s3://mousebraindata-open/MD657'
local_data='/dev/shm/data'
exec_dir='/home/ubuntu/shapeology_code/scripts'

if __name__=='__main__':
    Recent=False
    for logfile in glob(exec_dir+'/Controller*.log'):
        gap=time() - Last_Modified(logfile)
        if gap <120: # allow 2 minute idle
            print(logfile,'gap is %6.1f'%gap)
            Recent=True
            break
    if(not Recent):
        # Check that another 'controller' is not running
        stdout,stderr = runPipe('ps aux')
        Other_controller=False
        for line in stdout:
            if 'Controller.py' in line:
                Other_controller=True
                break
        
        if Other_controller:
            print('Other Controller.py is running')
        else:
            command='{0}/Controller.py {0} {1} {2}'\
                .format(exec_dir,stack,local_data)
            outfile=open('{0}/Controller-{1}.log'.format(exec_dir,int(time())),'w')
            errfile=open('{0}/Controller-{1}.err'.format(exec_dir,int(time())),'w')
            out,err=runPipe(command)
            outfile.write('\n'.join(out))
            errfile.write('\n'.join(out))            
            
