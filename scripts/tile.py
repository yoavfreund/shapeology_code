import re
import pickle
import glymur
from glob import glob
from time import time
import numpy as np
from os import mkdir

def tile_image(fullres,tile_path,stem):

    x,y,z=fullres.shape

    nx=int(np.round(x/1000))
    ny=int(np.round(y/1000))
    dx=int(np.floor(x/nx))
    dy=int(np.floor(y/ny))

    i=0
    for ulx in range(0,x-dx,dx):
        for uly in range(0,y-dy,dy):
            #print('\r',ulx,uly,end='')
            i+=1
            window=fullres[ulx:ulx+dx,uly:uly+dy]
            pickle.dump(window,open('%s/%s_%d_%d.pkl'%(tile_path,stem,ulx,uly),'bw'),protocol=4)
    print('x=%6d,y=%6d nx=%3d,ny=%3d,tile no=%d'%(x,y,nx,ny,i))

path='/home/ubuntu/workspace/data/'
tile_path=path+'tiles/'
try:
    mkdir(tile_path)
except:
    pass
filenames=glob(path+'*_lossless.jp2')
for filename in filenames:               
    pat=re.compile(r'_(\w{2}\d{3}_\d_\d{4})_lossless.jp2')
    m=pat.search(filename)
    if m:
        stem=m.group(1)
        print(stem)
    else:
        print('cant parse\n',filename)
        continue
    print('starting to read %s'%filename)
    t0=time()
    jp2 = glymur.Jp2k(filename)
    fullres = jp2[:]
    

t1=time()
print('finished reading %s (%5.1f sec)'%(filename,time()-t0))
print()
tile_image(fullres,tile_path,stem)
print('finished tiling (%5.1f sec)'%(time()-t1))