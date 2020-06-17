from sys import argv
from os import getpid,system
from time import sleep
import datetime
from lib.utils import configuration
import argparse

def getLock(lockfile):
    try:
        l=open(lockfile, 'x')
        return l
    except FileExistsError:
        return None


parser = argparse.ArgumentParser()
parser.add_argument('stem',type=str,
                    help="the file to be processed")
parser.add_argument("yaml", type=str,
                    help="Path to Yaml file with parameters")

args = parser.parse_args()

config = configuration(args.yaml)
params=config.getParams()

stem=args.stem

config = configuration(args.yaml)
params=config.getParams()

tiles_dir=params['paths']['data_dir']+'/tiles/'
lockfilename=tiles_dir+stem+'.lock'
print('lockfilename=',lockfilename)
logfilename=tiles_dir+stem+'.log'

lockfile=getLock(lockfilename)
if not lockfile is None:

    command='python3 extractPatches.py %s %s > %s &'%(stem,args.yaml,logfilename)
    # put some info into the log/lock file
    now = datetime.datetime.now()
    print('pid=',getpid(), file=lockfile)
    print('start time=',now.isoformat(), file=lockfile)
    print('command=',command, file=lockfile)
    
    system(command)
    sleep(1)
else:
    print(lockfilename,'exists, skipping')
