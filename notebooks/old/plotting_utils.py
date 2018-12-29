import psutil
from os import system
from subprocess import Popen,PIPE
from time import time

import numpy as np
from numpy import sqrt, mean
from subprocess import Popen,PIPE
from os import system
from os.path import isfile,getmtime

import cv2

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel,convolve

def plot_patches(fig,data,h=15,w=15,_titles=[]):
    for i in range(h*w):
        if i>=data.shape[0]:
            break
        ax=fig.add_subplot(h,w,i+1);
        pic=np.array(data[i,:,:],dtype=np.float32)

        subfig=ax.imshow(pic,cmap='gray')
        if(len(_titles)>i):
            plt.title(_titles[i])
        subfig.axes.get_xaxis().set_visible(False)
        subfig.axes.get_yaxis().set_visible(False)

def pack_pics(Reps):
    size=Reps[0].shape[0]
    _len=len(Reps)
    Reps_mat=np.zeros([_len,size,size])
    for i in range(_len):
        Reps_mat[i,:,:]=Reps[i]
    return Reps_mat

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
