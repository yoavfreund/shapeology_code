#!/usr/bin/env python3
from os import mkdir
import argparse
from lib.utils import *
import re
"""
The controller runs a given script on a set of files on S3. The application is intended to run on a set of ec2 instances in parallel. 
Lock files are used to insure that each file is processed exactly once.

This python file will be rewritten in a simpler way using datajoin
"""
def get_file_table(s3_directory,pattern=r'.*'):
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
    for file in stdout:
        parts=file.strip().split()
        if len(parts)!=4:
            continue
        filename=parts[3]
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

def find_and_lock(s3_directory,pattern=r'.*'):
    """ find an s3 file without a lock and lock it

    :param s3_directory: location of files and locks
    :param pattern: regular expression that should match the file stem
    """
    T=get_file_table(s3_directory,pattern=pattern)

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
        open(scripts_dir+'/'+flagname,'w').write(flagname+'\n')

        command='aws s3 cp %s %s/%s'%(scripts_dir+'/'+flagname,s3_directory,flagname)
        run(command)

        # check to make sure that there is only one lock.
        T=get_file_table(s3_directory)
        extensions=T[filename]
        if len(extensions)==2:
            return filename
    


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts_dir", type=str,
                        help="path to the directory with the scripts")
    parser.add_argument("script",type=str,
                        help='the name of the script that is to run on each file')
    parser.add_argument("s3location", type=str,
                        help="path to the s3 directory with the lossless images")
    parser.add_argument("pattern",type=str,
                        help="pattern for filtering files from S3 directory")
    parser.add_argument("local_data",type=str,
                        help="path to the local data directory")
    # pattern=r'(.*)\.([^\.]*)$'
    args = parser.parse_args()
    scripts_dir=args.scripts_dir
    s3_directory=args.s3location
    local_data=args.local_data
    
    time_log=[]
    clock('starting Controller with s3_directory=%s, local_data=%s'%(s3_directory,local_data))


    #preparations: make dirs data and data/tiles
    try:
        run('sudo chmod 0777 /dev/shm/')
        mkdir(local_data)
        mkdir(local_data+'/tiles')
        clock('created data directory')
    except:
        pass

    #main loop
    while True:
        #find an unprocessed file on S3
        stem=find_and_lock(s3_directory,pattern=r'(.*)\.([^\.]*)$')
        clock('found and locked %s'%stem)

        if stem==None:
            print('all files processed')
            break

        cmd='python3 %s/%s --local_data  %s --s3location %s --stem %s'\
            %(scripts_dir,script,local_data,s3_directory,stem)
        run(cmd)
        
    printClock()


