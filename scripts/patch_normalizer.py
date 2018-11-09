import cv2
from cv2 import moments,HuMoments
from skimage.measure import label
import pickle
import numpy as np
from lib.shape_utils import find_threshold

class normalizer:
    """ a class for normalizing patches """
    def __init__(self,cell_size=41,threshold=1):
        """
        Initialize a Normalizer
        create a circular mask of a given size, size must be an odd number.

        :param cell_size: the size of the mask.
        """
        cell_size=int(cell_size)
        if cell_size < 1 or cell_size % 2 !=1:
            raise Exception("normalizer.init, cell_size=",cell_size,"should be an odd positive integer")
        self.cell_size=cell_size
        self.threshold=threshold
        self.center=(cell_size-1)/2.
        self.icent=int(self.center)
        self.footprint=np.ones([cell_size,cell_size])>1
        
        for i in range(cell_size):
            for j in range(cell_size):
                self.footprint[i,j]=((i-self.center)**2+(j-self.center)**2)<=self.center**2
                
        self.mask = 1.* self.footprint
        self.ratio=sum(self.footprint.flatten())/(self.cell_size**2) # ratio of footprint area to square patch area
        #return self.cell_size,self.center,self.ratio,self.footprint

    def normalize_greyvals(self,ex):
        """normalize the grey-values of a patch so the the mean is zero and the std is 1.
        :param ex:  patch
        :returns: 
        * _m: the mean grey val before normalization
        * _m2: std of the grey val before normalization
        * patch: normalized pattch: numpy 2D array
        """
        ex=ex*self.mask
        _flat=ex.flatten()
        _m=np.mean(_flat)/self.ratio
        _m2=np.mean(_flat**2)/self.ratio
        if _m2>_m**2:
            _std=np.sqrt(_m2-_m**2)
        else:
            _std=1
            print('error in calc of _std',_m,_m2)
        ex_new=(ex-_m)/_std
        return _m,_std,ex_new * self.mask

    def segment_patch(self,patch):
        threshold=find_threshold(patch,percentile=0.8)
        binary=(patch<threshold)*1
        labels=label(binary,connectivity=2,background=int(binary[0,0]))
        mask=(labels==labels[self.icent,self.icent])
        masked_patch=patch*mask
        return masked_patch
    
    def angle(self,ex):
        """compute the rotation angle of a patch

        :param ex: 
        :returns: 
        :rtype: 

        """
        rows,self.size = ex.shape
        nex=np.array(ex)
        #nex[nex>0]=0  # remove background
        M=moments(-nex+self.mask)
        #print(M['m00'],M['m10'],M['m01'])
        x=M['m10']/M['m00']
        y=M['m01']/M['m00']
        nu20=(M['m20']/M['m00'])-x**2
        nu02=(M['m02']/M['m00'])-y**2
        nu11=(M['m11']/M['m00'])-y*x
        confidence=0
        if nu11!=0:
            confidence =np.abs((nu20-nu02)/nu11)  # the confidence number
                                              # is small if small
                                              # changes to the moments
                                              # will generate a
                                              # different angle
                                              # estimate

        ang_est=0
        if confidence > 0.1:
            ang_est=-np.arctan(2*nu11/(nu20-nu02))/np.pi+0.5

        if ang_est>0.5:
            ang_est-=1
        ang180=(ang_est+(np.sign(nu11))/2)*90

        if ang180>=180:
            ang180-=360
        if ang180<-180:
            ang180+=360
        return ang180,confidence

    def flipOrNot(self,ex):
        """
        decide using the moments whether to flip the image.
        :param ex: 
        :returns: 
        :rtype: 

        """
        self.size,self.size = ex.shape
        M=moments(ex)
        x=M['m10']/M['m00'] - self.center
        y=M['m01']/M['m00'] - self.center
        if abs(x)>abs(y):
            return x<0
        else:
            return y<0

    def normalize_angle(self,ex):
        """ place patch in a normalized angle

        :param ex: patch
        :returns: angle, angle and confidence
        :rtype: 

        """
        ang,conf=self.angle(ex)
        M = cv2.getRotationMatrix2D((self.center,self.center),-ang,1)
        dst= cv2.warpAffine(ex,M,(self.size,self.size))*self.mask
        if self.flipOrNot(dst):
            M180 = cv2.getRotationMatrix2D((self.center,self.center),180,1)
            dst = cv2.warpAffine(dst,M180,(self.size,self.size))

        return ang,conf,dst*self.mask

    # normalized_patch,rotation,confidence=Norm.
    def normalize_patch(self,ex):
        ex *= self.mask
        #normalize patch interms of grey values and in terms of rotation
        _m,_std,ex_grey_normed=self.normalize_greyvals(ex)
        segmented=self.segment_patch(ex_grey_normed)
        rot_angle1,conf1,ex_rotation_normed=self.normalize_angle(segmented)
        rot_angle2,conf2,ex_rotation_normed=self.normalize_angle(ex_rotation_normed)
        total_rotation = rot_angle1+rot_angle2
        confidence=conf2
        normalized_patch =  ex_rotation_normed*self.mask
        return normalized_patch,total_rotation,confidence
