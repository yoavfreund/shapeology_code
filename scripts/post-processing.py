"""Process the patch files generated by extractPatches.py and prepare
them for analysis by Kmeans and diffusion-maps"""

from glob import glob
import pickle as pk
from os import system
import numpy as np
from os.path import isdir
from minio import Minio

def pad_patch(patch):
    too_big=True
    size=patch.shape[0]
    for sz_block in size_tresholds:
        if size<sz_block:
            too_big=False
            break
    if too_big:
        return None,size

    pad=np.zeros([sz_block,sz_block],dtype=np.float16)
    _from=int((sz_block-size)/2)
    _to = int(sz_block-_from)
    # print(size,sz_block,_from,_to)
    pad[_from:_to,_from:_to]=patch
    return pad,sz_block
    
def extract_vectors():
    Vec_dict={}
    for sz in size_tresholds:
        Vec_dict[sz]=[]
    
    for filename in glob('%s/tiles/*extracted*'%local_data):
        #print('filename=',filename)
        D=pk.load(open(filename,'rb'))
        for e in D:
            patch=np.array(e['normalized_patch'],dtype=np.float16)
            padded,sz=pad_patch(patch)
            if not padded is None:
                Vec_dict[sz].append(padded)
    #return a dict of arrays partitioned according to size.
    return Vec_dict

def run(cmd):
    print('cmd=',cmd)
    return system(cmd)

############################
local_data='/dev/shm/data'
exec_dir='/home/ubuntu/shapeology_code/scripts'
stack='s3://mousebraindata-open/MD657/'
size_tresholds=[15,27,51,81,151,201]
files=!aws s3 ls $_dir | grep lossless_patches | grep "\-N"

i=0
f=0
N=0
total_MB=0
for line in files:
    ALL_V={}
    patch_file=line.split()[-1]
    #print('file=',patch_file)
    run('rm %s/tiles/*'%local_data)
    run('aws s3 cp %s%s %s/%s'%(stack,patch_file,local_data,patch_file))
    !ls -l $local_data/$patch_file
    run('tar xzvf %s/%s -C /'%(local_data,patch_file))
    if isdir(local_data+'/tiles/dev'):
        print('moving from dev')
        run('mv {0}/tiles/dev/shm/data/tiles/ {0}/tmp_tiles/'.format(local_data))
        run('rm -rf {0}/tiles/'.format(local_data))
        run('mv {0}/tmp_tiles/ {0}/tiles/'.format(local_data))
    V=extract_vectors()
    for sz in size_tresholds:
        if len(V[sz])>0:
            Array=np.array(V[sz],dtype=float16)
            print(sz,Array.shape)
            if not sz in ALL_V:
                ALL_V[sz]=Array
            else:
                ALL_V[sz]=concatenate([ALL_V[sz],Array])
    pk.dump(ALL_V,open(local_data+'/'+patch_file+'.pkl','wb'))

###############
import pickle as pk
import numpy as np
from glob import glob

K=100  #number of files used for randomization
size_tresholds=[15,27,51,81,151,201]
data_dir="/dev/shm/data"

for _size in [15,27,81,151,201]:
    permuted_dir=data_dir+'/permuted-%s'%(_size)
    !rm -rf $permuted_dir
    !mkdir $permuted_dir
    !ls $permuted_dir
    stem=permuted_dir+'/permuted'
    print(stem)

    fp=[]
    for i in range(K):
        fp.append(open(stem+str(i)+'.bin','bw'))


    patches=np.zeros([0])
    k=0
    for filename in glob(data_dir+'/*.pkl'):
        print(filename)
        pkl_file=open(filename,'br')
        try:
            V=pk.load(pkl_file)
        except:
            print('could not load',filename)
            continue
        print(V[_size].shape)
        patches=V[_size]

        for i in range(patches.shape[0]):
            j=np.random.randint(K)
            patch=np.array(patches[i,:,:],dtype=np.float16)
            patch.tofile(fp[j])
            if i%1000==0:
                print('\r',filename,i,end='')

        print('finished',k,filename,patches.shape)
        k+=1
        #!rm $filename

    print('finished it all!!')
    for handle in fp:
        handle.close()

    !ls -lrth $permuted_dir | wc

    # read and permute each file
    for filename in glob(permuted_dir+'/*.bin'):

        D=fromfile(filename,dtype=np.float16) #,count=_size*_size)
        pics=D.reshape([-1,_size,_size])

        L=D.shape[0]
        _order=permutation(pics.shape[0])
        permuted_pics=pics[_order,:,:]
        permuted_pics.tofile(filename)
        _s=pics.shape
        error=L - (_s[0]*_s[1]*_s[2])
        print(filename,'shape=',pics.shape,'error=',error)

    tar_filename=data_dir+'/permuted-%d'%_size+'.tgz'

    !tar czvf $tar_filename $permuted_dir/

    #write file to s3
    !aws s3 mv $tar_filename s3://mousebraindata-open/MD657/permuted/

    !rm -rf $permuted_dir