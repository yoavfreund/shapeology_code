from sys import argv
from os import getpid,system
from time import sleep
import datetime


def getLock(lockfile):
    try:
        l=open(lockfile, 'x')
        return l
    except FileExistsError:
        return None

stem=argv[1]

lockfilename=stem+'.lock'
logfilename=stem+'.log'

lockfile=getLock(lockfilename)
if not lockfile is None:

    command='python3 extractPatches.py %s shape_params.yaml > %s &'%(stem,logfilename)
    # put some info into the log/lock file
    now = datetime.datetime.now()
    print('pid=',getpid(), file=lockfile)
    print('start time=',now.isoformat(), file=lockfile)
    print('command=',command, file=lockfile)
    
    system(command)
    sleep(1)
else:
    print(lockfilename,'exists, skipping')
