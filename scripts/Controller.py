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
def get_file_table(s3_directory,pattern=r'(.*_\d_\d{4})(.*)'):
    """generate a dictionary of the files in an s3 directory 
    that fit a given regex pattern.

    :param s3_directory: s3 directory, example: s3://mousebraindata-open/MD657/
    :param pattern: a regular expression defining the files to be considered.
    :returns: A dictionary in which the file name stem is the key and the value is a list of descriptors of files which have that stem.
    :rtype: 

    """
    
    stdout = list_s3_files(s3_directory)
    pat=re.compile(pattern)

    T={}
    for filename in stdout:
        m=pat.match(filename)
        if m:
            #print('matched, groups=',m.groups())
            file,ext= m.groups()
            if file in T:
                T[file].append(ext)
            else:
                T[file]=[ext]
        else:
             print(filename,'no match')
    return T

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts_dir", type=str,
                        help="path to the directory with the scripts")
    parser.add_argument("script",type=str,
                        help='the name of the script that is to run on each file')
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument("local_data",type=str,
                        help="path to the local data directory")
    # pattern=r'(.*)\.([^\.]*)$'
    args = parser.parse_args()
    scripts_dir=args.scripts_dir
    script=args.script
    s3_directory=args.s3location
    local_data=args.local_data

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

        cmd='python3 {0}/{1} {0} {2} {3} {4}'.format(scripts_dir,script,s3_directory,stem+'_lossless',local_data)
        run(cmd)
        
    printClock()
