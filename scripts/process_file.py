#!/usr/bin/env python3
"""
A script for processing a single section.
"""
import psutil
from glob import glob
from time import sleep
from os.path import isfile,isdir, getsize
from os import chdir
import numpy as np
import argparse
from lib.utils import *
import sys
sys.path.append('/home/ubuntu/Datajoint_Interface/project_schemas/atlas_schema_python_v3/setup')
from utilities import *

def process_tiles(tile_pattern,scripts_dir,yaml_file):
    i=0
    print('tile_pattern=',tile_pattern)
    for infile in glob(tile_pattern):
        stem=infile.split('/')[-1]
        stem=stem[:-4]
        #print ('infile=%s, stem=%s'%(infile,stem))
        lockfile=stem+'.lock'
        if not isfile(lockfile):
            i+=1
            print('got lock',lockfile,i)
            run('python3 {0}/run_job.py {1} {2} &'.format(scripts_dir,stem,yaml_file))
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

def process_file(local_data,s3_directory,stem,scripts_dir,params,yaml_file):
    print('processing %s, local_data=%s, s3_directory=%s, scripts_dir=%s'%(stem,local_data,s3_directory,scripts_dir))

    # pickle_dir=params['paths']['pickle_subdir']

    # check if files already exist
    patches_fn=  '{0}_patches.tgz'.format(stem)
    extracted_fn = "{0}_extracted.tgz".format(stem)
    filename='{1}/{0}'.format(extracted_fn,s3_directory)
    print('checking for',filename)
    run('aws s3 cp {1} {0}/'.format(local_data,filename))
    if isfile('{0}/{1}'.format(local_data,extracted_fn)):
        print ("file already computed")
    else:
        if not isdir(local_data):
            run('sudo mkdir %s'%local_data)
            run('sudo chmod a+rwx %s'%local_data)
            
        if isfile('%s/%s.tif'%(local_data,stem)):
            print('found %s/%s.tif skipping download and kdu'%(local_data,stem))
        else:
            #Bring in a file and break it into tiles
            run('aws s3 cp %s/%s.jp2 %s/%s.jp2'%(s3_directory,stem,local_data,stem))
            clock('copied from s3: %s'%stem)
            run('kdu_expand -i %s/%s.jp2 -o %s/%s.tif'%(local_data,stem,local_data,stem))
            clock('translated into tif')

        # cleanup work dir
        run('rm -rf %s/tiles'%(local_data))
        run('mkdir %s/tiles/'%local_data)
        run('mkdir %s/tiles/pickles'%(local_data))
        clock('cleaning local directory')

        # Break image into tiles
        run('convert %s/%s.tif -crop 20x10@+100+100@  %s'%(local_data,stem,local_data)+'/tiles/tiles_%02d.tif')
        clock('broke into tiles')

        chdir(scripts_dir)

        # perform analysis
        i=process_tiles('%s/tiles/tiles_*.tif'%local_data, scripts_dir, yaml_file)
        clock('1 - processed %6d tiles'%i)
        i=process_tiles('%s/tiles/tiles_*.tif'%local_data,scripts_dir, yaml_file)
        clock('2 - processed %6d tiles'%i)

        #copy results to s3
        chdir(local_data)


        run("tar czf {0}/{1} tiles/*.log  tiles/*.lock  tiles/*.jpg".format(local_data,patches_fn))
        clock('created tar file {0}/{1}'.format(local_data,patches_fn))

        chdir("{0}/tiles/pickles".format(local_data))  #chdir to pickles directory

        run("tar czf {0}/{1} *.pkl".format(local_data,extracted_fn))
        extracted_size=getsize('{0}/{1}'.format(local_data,extracted_fn))
        clock('created tar file  {0}/{1}'.format(local_data,extracted_fn,'of size',extracted_size))

        chdir(scripts_dir)

        run('aws s3 cp {0}/{1} {2}/'.format(local_data,extracted_fn,s3_directory))
        run('aws s3 cp {0}/{1} {2}/'.format(local_data,patches_fn,s3_directory))
        run('rm  {0}/*'.format(local_data))
        clock('copy tar file to S3')
 
    return patches_fn, extracted_fn, extracted_size

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script that takes in an S3 directory breaks it into tiles and extracts normalized patches from these tiles.")
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument('stem',type=str,
                        help='the file name stem')
    parser.add_argument("yaml", type=str,
                    help="Path to Yaml file with parameters")

    args = parser.parse_args()

    config = configuration(args.yaml)
    params=config.getParams()


    scripts_dir=params['paths']['scripts_dir']
    s3_directory=args.s3location
    stem=args.stem
    local_data=params['paths']['data_dir']
    print('processing %s, local_data=%s, s3_directory=%s, scripts_dir=%s'%(stem,local_data,s3_directory,scripts_dir))
    
    clock('starting to process %s/%s'%(s3_directory,stem))
    process_file(local_data,s3_directory,stem,scripts_dir,params)
    clock('finished processing %s/%s'%(s3_directory,stem))

    printClock()
