#!/usr/bin/env python3
"""
A script for processing a single section.
"""
import psutil
from os.path import isfile
from glob import glob
from time import sleep
from os.path import isfile
import numpy as np
import argparse
from lib.utils import *


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
            run('python3 run_job.py {0}'.format(stem))
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

def process_file(local_data,s3_directory,stem):
    run('rm -rf %s/*'%(local_data))
    run('mkdir %s/tiles'%local_data)
    clock('cleaning local directory')
    
    #Bring in a file and break it into tiles
    run('aws s3 cp %s/%s.jp2 %s/%s.jp2'%(s3_directory,stem,local_data,stem))
    clock('copied from s3: %s'%stem)
    run('kdu_expand -i %s/%s.jp2 -o %s/%s.tif'%(local_data,stem,local_data,stem))
    clock('translated into tif')
    # convert MD657-N48-2017.02.22-16.41.55_MD657_2_0143_lossless.tif -crop 20x10@+100+100@  new_tiles/tiles_%02d.tif
    run('convert %s/%s.tif -crop 20x10@+100+100@  %s'%(local_data,stem,local_data)+'/tiles/tiles_%02d.tif')
    clock('broke into tiles')
    
    # perform analysis
    i=process_tiles('%s/tiles/tiles_*.tif'%local_data)
    clock('1 - processed %6d tiles'%i)
    i=process_tiles('%s/tiles/tiles_*.tif'%local_data)
    clock('2 - processed %6d tiles'%i)

    #copy results to s3
    run("tar czf {0}/{1}_patches.tgz {0}/tiles/*.pkl {0}/tiles/*.log {0}/tiles/*.lock".format(local_data,stem))
    clock('created tar file {0}/{1}_patches.tgz'.format(local_data,stem))
    
    run('aws s3 cp {0}/{1}_patches.tgz {2}/'.format(local_data,stem,s3_directory))
    clock('copy tar file to S3')
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script that takes in an S3 directory breaks it into tiles and extracts normalized patches from these tiles.")
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument('stem',type=str,
                        help='the file name stem')
    parser.add_argument("local_data",type=str,
                        help="path to the local data directory")
    #parser.
    # pattern=r'(.*)\.([^\.]*)$'
    args = parser.parse_args()

    s3_directory=args.s3location
    stem=args.stem
    local_data=args.local_data

    clock('starting to process %s/%s'%(s3_directory,stem))
    process_file(local_data,s3_directory,stem)
    clock('finished processing %s/%s'%(s3_directory,stem))

    printClock()
