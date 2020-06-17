from os import system

command="python3 process_file.py s3://mousebraindata-open/MD657 %s  /dev/shm/data > logs/logfile%s"

files = open('S3FileList.txt','r').readlines()

print(len(files))

for file in files:
    if 'lossless.jp2' in file:
        _,_,_,filename=file.split()
        filename=filename[:-4]
        this_command=command%(filename,filename[-20:])
        print(this_command)
        system(this_command)
