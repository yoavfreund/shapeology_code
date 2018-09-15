#!/usr/bin/env python3
import psutil
import socket
from os import getpid,mkdir,system
from subprocess import Popen,PIPE
from os.path import isfile
from glob import glob
from time import sleep,time
from os.path import isfile
from sys import argv
import re
import socket
import argparse
import numpy as np

def run(command):
    print('cmd=',command)
    system(command)
    
def runPipe(command):
    print('runPipe cmd=',command)
    p=Popen(command.split(),stdout=PIPE,stderr=PIPE)
    L=p.communicate()
    stdout=L[0].decode("utf-8").split('\n')
    stderr=L[1].decode("utf-8").split('\n')
    return stdout,stderr

def clock(message):
    print('%8.1f \t%s'%(time(),message))
    time_log.append((time(),message))

def printClock():
    t=time_log[0][0]
    for i in range(1,len(time_log)):
        print('%8.1f \t%s'%(time_log[i][0]-t,time_log[i][1]))
        t=time_log[i][0]

def get_file_table(stack_directory):
    """create a table of the files in a directory corresponding to a stack:
    stack_directory: the location of the directory on s3.
    example: s3://mousebraindata-open/MD657/
    """

    awsfiles='/home/ubuntu/shapeology_code/scripts/awsfiles.txt'
    stdout,stderr=runPipe("aws s3 ls %s/ "%(stack_directory))
    pat=re.compile(r'(.*)\.([^\.]*)$')

    T={}
    for file in stdout:
        parts=file.strip().split()
        if len(parts)!=4:
            continue
        filename=parts[3]
        if not 'lossless.' in filename:
            continue
        m=pat.match(filename)
        if m:
            file,ext= m.groups()
            info=(ext,parts[0]+' '+parts[1])
            if file in T:
                T[file].append(info)
            else:
                T[file]=[info]
        else:
            print(filname,'no match')
    return T

def find_and_lock(stack_directory):
    """ find a section file without a lock and lock it"""
    T=get_file_table(stack_directory)

    while True:
        #find a file without a lock
        found=False
        for item in T.items():
            if len(item[1])==1:
                found=True
                break
        if not found:
            return None
        
        filename=item[0]
        extensions=item[1]

        #create a lock
        hostname=socket.gethostname().replace('.','+')
        flagname=filename+'.lock-'+hostname
        open(scripts+'/'+flagname,'w').write(flagname+'\n')

        command='aws s3 cp %s %s/%s'%(scripts+'/'+flagname,stack_directory,flagname)
        run(command)

        # check to make sure that there is only one lock.
        T=get_file_table(stack_directory)
        extensions=T[filename]
        if len(extensions)==2:
            return filename
    
        # translation of date for better handling of two machines putting locks 
        # at nearly the same time
        # comparing the time stamps of the locks can be used to resolve who was firstand should continue
        # and who should look for another file.
        # from datetime import datetime
        # d1=datetime.strptime('2018-08-28 21:16:34','%Y-%m-%d %H:%M:%S')

def process_tiles(tile_pattern):
    i=0
    print('tile_pattern=',tile_pattern)
    for infile in glob(tile_pattern):
        stem=infile[:-4]
        #print ('infile=%s, stem=%s'%(infile,stem))
        lockfile=stem+'.lock'
        if not isfile(lockfile):
            i+=1
            print('got lock',lockfile,i)
            run('python3 {0}/run_job.py {0} {1}'.format(scripts,stem))
            sleep(0.1)
        else:
            #print('\r %s exists'%lockfile,end='')
            continue

        # Wait if load is too high
        load=np.mean(psutil.cpu_percent(percpu=True))
        print(' %5d                            load: %6.2f'%(i,load))
        j=0
        while load>85:
            print(' %5d    Sleep:%3d               load: %6.2f'%(i,j,load))
            j+=1
            sleep(2)
            load=np.mean(psutil.cpu_percent(percpu=True))

        print('\nload low enough',load)
    return i

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", type=str,
                        help="path to the directory with the scripts")
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument("local_data",type=str,
                        help="path to the local data directory")
    args = parser.parse_args()

    scripts=args.scripts
    stack_directory=args.s3location
    local_data=args.local_data
    
    time_log=[]

    clock('starting Controller with stack_directory=%s, local_data=%s'%(stack_directory,local_data))

    try:
        #preparations: make dirs data and data/tiles
        run('sudo chmod 0777 /dev/shm/')
        mkdir(local_data)
        mkdir(local_data+'/tiles')
        clock('created data directory')
    except:
        pass


    while True:
        #find an unprocessed file on S3
        stem=find_and_lock(stack_directory)
        clock('found and locked %s'%stem)

        if stem==None:
            print('all files processed')
            break

        system('rm -rf %s/tiles'%local_data)
        run('rm %s/'%(local_data))
        clock('cleaning local directory')


        #Bring in a file and break it into tiles
        run('aws s3 cp %s/%s.jp2 %s/%s.jp2'%(stack_directory,stem,local_data,stem))
        clock('copied from s3: %s'%stem)
        run('kdu_expand -i %s/%s.jp2 -o %s/%s.tif'%(local_data,stem,local_data,stem))
        clock('translated into tif')
        run('convert %s/%s.tif -crop 1000x1000  +repage  +adjoin  %s'%
            (local_data,stem,local_data)+'/tiles/tiles_%02d.tif')
        clock('broke into tiles')

        # perform analysis
        i=process_tiles('%s/tiles/tiles_*.tif'%local_data)
        clock('1 - processed %6d tiles'%i)
        i=process_tiles('%s/tiles/tiles_*.tif'%local_data)
        clock('2 - processed %6d tiles'%i)

        #copy results to s3
        run("tar czf {0}/{1}_patches.tgz {0}/tiles/*.pkl {0}/tiles/*.log {0}/tiles/*.lock".format(local_data,stem))
        clock('created tar file {0}/{1}_patches.tgz'.format(local_data,stem))

        run('aws s3 cp {0}/{1}_patches.tgz {2}/'.format(local_data,stem,stack_directory))
        clock('copy tar file to S3')

    printClock()
