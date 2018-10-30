### A top level script for performing analysis
import psutil
from os import getpid,system
from os.path import isfile
from glob import glob
from time import sleep
from os.path import isfile
from sys import argv

tile_pattern=argv[1]

max_load=700.
def processTiles(tile_pattern,max_load=700):
    print('processing tiles in',tile_pattern)
    
    i=0
    for infile in glob(tile_pattern):
        stem=infile[:-4]
        lockfile=stem+'.lock'
        if not isfile(lockfile):
            i+=1
            print('got lock',lockfile)
            command='python3 run_job.py %s &'%stem
            print(i,command,'\n')
            system(command)
            sleep(0.1)
        else:
            print('\r %s exists'%lockfile,end='')
            continue

        # Wait if load is too high
        load=sum(psutil.cpu_percent(percpu=True))
        print('\r %5d                            load: %6.2f'%(i,load))
        j=0
        while load>max_load:
            print('\r %5d    Sleep:%3d               load: %6.2f'%(i,j,load))
            j+=1
            sleep(2)
            load=sum(psutil.cpu_percent(percpu=True))
        print('\nload low enough',load)
