import psutil
from os import system
from subprocess import Popen,PIPE
from time import time

import numpy as np
from numpy import sqrt, mean
#utils that do not require plotting
from subprocess import Popen,PIPE
import os
from os import system
from os.path import isfile,getmtime

import cv2

#import matplotlib.pyplot as plt
#from astropy.convolution import Gaussian2DKernel,convolve

import yaml

class configuration():
    def __init__(self,yamlFile):
        self.D=yaml.load(open(yamlFile,'r'))

    def getParams(self):
        return self.D

def run(command):
    print('run cmd=',command)
    system(command)

def setup_download_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.path.join(os.environ['ROOT_DIR'], rel_fp)

    if os.path.exists(local_fp):
        print('ALREADY DOWNLOADED FILE')
        return

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(s3_fp, local_fp))
    else:
        run('aws s3 cp {0} {1}'.format(s3_fp, local_fp))

def setup_upload_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.path.join(os.environ['ROOT_DIR'], rel_fp)

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(local_fp, s3_fp))
    else:
        run('aws s3 cp {0} {1}'.format(local_fp, s3_fp))


def runPipe(command):
    print('runPipe cmd=',command)
    p=Popen(command.split(),stdout=PIPE,stderr=PIPE)
    L=p.communicate()
    stdout=L[0].decode("utf-8").split('\n')
    stderr=L[1].decode("utf-8").split('\n')
    return stdout,stderr

def Last_Modified(file_name):
    try:
        mtime = getmtime(file_name)
    except OSError:
        mtime = 0
    return(mtime)


time_log=[]
def clock(message):
    print('%8.1f \t%s'%(time(),message))
    time_log.append((time(),message))

def printClock():
    t=time_log[0][0]
    for i in range(1,len(time_log)):
        print('%8.1f \t%s'%(time_log[i][0]-t,time_log[i][1]))
        t=time_log[i][0]

def list_s3_files(stack_directory):
    stdout,stderr=runPipe("aws s3 ls %s/ "%(stack_directory))
    filenames=[]
    for line in stdout:
        parts=line.strip().split()
        if len(parts)!=4:
            continue
        filenames.append(parts)
    return filenames

def read_files(s3_dir,_delete=False,data_dir='/dev/shm/data/'):
    s3files=list_s3_files(s3_dir)
    for filename in s3files:
        if not isfile(data_dir+'/'+filename):
            run('aws s3 cp %s/%s %s'%(s3_dir,filename,data_dir))
        D=fromfile(data_dir+'/'+filename,dtype=np.float16)
        pics=D.reshape([-1,_size,_size])
        if _delete:
            run('rm %s/%s'%(data_dir,filename))
        yield pics

def data_stream(s3_dir='s3://mousebraindata-open/MD657/permuted'):
    for pics in read_files(s3_dir):
        j=0
        for i in range(pics.shape[0]):
            if j%1000==0:
                print('\r examples read=%10d'%j,end='')
            j+=1    
            yield pics[i,:,:]


def Last_Modified(file_name):
    try:
        mtime = getmtime(file_name)
    except OSError:
        mtime = 0
    return(mtime)
            
def calc_err(pic,gaussian = None):
    if gaussian is None:
        gaussian=Gaussian2DKernel(1,x_size=7,y_size=7)
    factor=np.sum(gaussian)
    P=convolve(pic,gaussian)/factor
    #except:
    #    print('err in calc_err/convolve',pic.shape,gaussian.shape,factor)
    #    P=pic
    error=sqrt(mean(abs(pic-P)))
    sub=P[::2,::2]
    return error,sub

# def plot_patches(fig,data,h=15,w=15,_titles=[]):
#     for i in range(h*w):
#         if i>=data.shape[0]:
#             break
#         ax=fig.add_subplot(h,w,i+1);
#         pic=np.array(data[i,:,:],dtype=np.float32)

#         subfig=ax.imshow(pic,cmap='gray')
#         if(len(_titles)>i):
#             plt.title(_titles[i])
#         subfig.axes.get_xaxis().set_visible(False)
#         subfig.axes.get_yaxis().set_visible(False)

def pack_pics(Reps):
    size=Reps[0].shape[0]
    _len=len(Reps)
    Reps_mat=np.zeros([_len,size,size])
    for i in range(_len):
        Reps_mat[i,:,:]=Reps[i]
    return Reps_mat

def dist2(a,b):
    diff=(a-b)**2
    return np.sum(diff.flatten())

# def dist_hist(data):
#     D=[]
#     for i in range(1,data.shape[0]):
#         D.append(dist2(data[i,:,:],data[i-1,:,:]))
#         if i%1000==0:
#             print('\r',i,end='')
#     hist(D,bins=100);

# def find_threshold(image,percentile):
#     """find the threshold at the given percentile

#     :param image: grey-level image
#     :param percentile: threshold is chosen so that 
#                        percentile of the pixels are lower than it
#     :returns: threshold
#     :rtype: float

#     """
#     V=sorted(image.flatten())
#     l=len(V)
#     thr=V[int(l*percentile)] 
#     return thr

def mark_contours(D,tile):

    image = np.array(tile,dtype=np.uint8)
    kernel = np.ones((3,3),np.uint8)
    boundary=np.zeros(image.shape,np.uint8)
    repress=boundary.copy()
    #print('shape of boundary=',boundary.shape)
    _shape=D[0]['mask'].shape
    left=int(_shape[0]/2)
    right=_shape[0]-left
    left,right

    for R in D:
        #compute contour
        color=np.array([0,0,0],dtype=np.uint8)
        color[R['j'] % 2]=255
        mask=np.array(R['mask']*1,dtype=np.uint8)
        dilated = cv2.dilate(mask,kernel,iterations = 1)
        contour = dilated-mask
        #mark contour in ln image coordinates
        coor=[R['Y'],R['X']]

        boundary[coor[0]-left:coor[0]+right,coor[1]-left:coor[1]+right]\
        +=np.multiply.outer(contour,color)
        repress[coor[0]-left:coor[0]+right,coor[1]-left:coor[1]+right]\
        +=np.multiply.outer(contour,np.array([1,1,1],dtype=np.uint8))

    combined=image.copy()
    combined[repress==1]=255
    # combined +=boundary

    return combined
