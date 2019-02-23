#!/usr/bin/env python3
from os import mkdir
import argparse
from lib.utils import *
import re
import socket
"""
The controller runs a given script on a set of files on S3. The application is intended to run on a set of ec2 instances in parallel. 
Lock files are used to insure that each file is processed exactly once.

This python file will be rewritten in a simpler way using datajoin
"""

def find_and_lock(s3_directory):
    """ find an s3 file without a lock and lock it

    :param s3_directory: location of files and locks
    :param pattern: regular expression that should match the file stem
    """
    T=get_file_table(s3_directory)

    found=False
    for filename,extensions in T.items():
        if '_lossless.jp2' in extensions:
            found=True
            for ext in extensions:
                if '.lock' in ext:
                    found=False
                    break
        if found:
            break
    if not found:
        return None
        
    #create a lock
    hostname=socket.gethostname().replace('.','+')
    flagname=filename+'.lock-'+hostname
    open('/tmp/'+flagname,'w').write(flagname+'\n')

    command='aws s3 cp %s %s/%s'%('/tmp/'+flagname,s3_directory,flagname)
    run(command)

    return filename

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A master script that locks a section and runs process_file.py on it")
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument("yaml", type=str,
                    help="Path to Yaml file with parameters")

    args = parser.parse_args()
    s3_directory=args.s3location

    config = configuration(args.yaml)
    params=config.getParams()
    
    scripts_dir=params['paths']['scripts_dir']
    local_data=params['paths']['data_dir']

    
    args = parser.parse_args()

    time_log=[]
    clock('starting Controller with s3_directory=%s, local_data=%s'%(s3_directory,local_data))


    #preparations: make dirs data and data/tiles
    try:
        #run('sudo chmod 0777 /dev/shm/')
        mkdir(local_data)
        mkdir(local_data+'/tiles')
        clock('created data directory')
    except:
        pass

    #main loop
    while True:
        #find an unprocessed file on S3
        stem=find_and_lock(s3_directory)
        clock('found and locked %s'%stem)

        if stem==None:
            print('all files processed')
            break
# python process_file.py s3://mousebraindata-open/MD657 MD657-N48-2017.02.22-16.41.55_MD657_2_0143_lossless ../shape_params.yaml
        cmd='python3 {0}/{1} {2} {3} {4}'.format(scripts_dir,'process_file.py',s3_directory,stem+'_lossless',args.yaml)
        run(cmd)
        
    printClock()
