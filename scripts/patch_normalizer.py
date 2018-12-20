import cv2
from cv2 import moments,HuMoments
from skimage.measure import label
import pickle
import numpy as np
#from lib.shape_utils import find_threshold

def calc_width(im):
    im=im/np.sum(im)
    _sum=im.sum(axis=0)
    x=np.arange(0,len(_sum))
    return np.sqrt(np.dot(x**2,_sum)-np.dot(x,_sum)**2)

class normalizer:
    """ a class for normalizing patches """
    def __init__(self,params):
        """        Initialize a Normalizer
        create a circular mask of a given size, size must be an odd number.

        :param params
        """
        self.params=params
        
    def circle_patch(self,radius):
        size=2*radius+1
        x=np.arange(-radius,radius+1)
        xx=np.array([x for i in x])
        yy=xx.T
        d=np.sqrt(xx**2+yy**2)
        return np.array((d<radius+0.1)*1,dtype=np.uint8)

    def set_mask(self,radius):
        #print('set mask, radius=',radius)
        self.mask = self.circle_patch(radius)
        self.center=int(radius)
        self.size=2*radius+1
        
    def normalize_greyvals(self,ex):
        """normalize the grey-values of a patch so the the mean is zero and the std is 1.
        :param ex:  patch
        :returns: 
        * patch: normalized pattch: numpy 2D array
        * dict: a dictionary of values associated with normalizing the patch
        """
        _flat=ex.flatten()
        _m=np.mean(_flat)
        _m2=np.mean(_flat**2)
        if _m2>_m**2:
            _std=np.sqrt(_m2-_m**2)
        else:
            _std=1
            print('error in calc of _std',_m,_m2)

        Dict={
            'mean':_m,
            'std':_std
        }

        ex_new=ex/_std
        return ex_new,Dict

    
    def angle(self,ex):
        """compute the rotation angle of a patch

        :param ex: 
        :returns: 
        :rtype: 

        """
        M=moments(ex)
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
    def put_in_circle(self,ex,prop):
        M=moments(ex)
        #print(M['m00'],M['m10'],M['m01'])
        x=M['m10']/M['m00']
        y=M['m01']/M['m00']

        x=round(x)
        y=round(y)
        w=prop['width']
        h=prop['height']
        x,y,h,w
        xc=w-x
        yc=h-y
        radius=int(np.ceil(np.sqrt(np.max([(x*x+y*y),(xc*xc+y*y),(x*x+yc*yc),(xc*xc+yc*yc)]))))
        self.set_mask(radius)
        size=2*radius+1
        circ_patch=np.zeros([size,size])
        x1=int(radius-x)
        x2=x1+w
        y1=int(radius-y)
        y2=y1+h
        circ_patch[y1:y2,x1:x2]=ex
        return circ_patch

    def normalize_patch(self,ex,props):
        ex_in_circle = self.put_in_circle(np.copy(ex),props)
        #normalize patch interms of grey values and in terms of rotation
        ex_grey_normed,grey_level_stats=self.normalize_greyvals(np.copy(ex_in_circle))

        #normalize angle
        rot_angle1,conf1,ex_rotation_normed=self.normalize_angle(np.copy(ex_grey_normed))
        rot_angle2,conf2,ex_rotation_normed=self.normalize_angle(np.copy(ex_rotation_normed))
        total_rotation = rot_angle1+rot_angle2
        confidence=conf2
        normalized_patch =  ex_rotation_normed*self.mask
        Dict={
            #'original_patch':ex,
            #'patch_in_circle':ex_in_circle,
            #'ex_grey_normed':ex_grey_normed,
            'normalized_patch':normalized_patch,
            'rotation':total_rotation,
            'rotation_confidence':confidence,
            'horiz_std':calc_width(normalized_patch),
            'vert_std':calc_width(normalized_patch.T)
        }
        Dict.update(grey_level_stats)

        return Dict
