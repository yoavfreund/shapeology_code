import cv2
from cv2 import moments,HuMoments
%pylab inline
import pickle

def create_footprint(cell_size=41):
    center=(cell_size-1)/2.
    footprint=np.ones([cell_size,cell_size])>1

    for i in range(cell_size):
        for j in range(cell_size):
            footprint[i,j]=((i-center)**2+(j-center)**2)<=center**2
            
    ratio=sum(footprint)/(cell_size**2) # ratio of footprint area to square patch area
    return cell_size,center,ratio,footprint

def normalize_greyvals(ex):
    ex=ex*mask
    _m=np.mean(ex)/ratio
    _m2=np.mean(ex**2)/ratio
    _std=np.sqrt(_m2-_m**2)
    #print('normalize_greyvals: mean=',_m,'std=',_std)
    ex_new=(ex-_m)/_std
    return _m,_std,ex_new * mask

def angle(ex):
    rows,cols = ex.shape
    M=moments(ex+mask)
    #print(M['m00'],M['m10'],M['m01'])
    x=M['m10']/M['m00']
    y=M['m01']/M['m00']
    nu20=(M['m20']/M['m00'])-x**2
    nu02=(M['m02']/M['m00'])-y**2
    nu11=(M['m11']/M['m00'])-y*x
    ang_est=-np.arctan(2*nu11/(nu20-nu02))/np.pi+0.5

    if ang_est>0.5:
        ang_est-=1
    ang180=(ang_est+(np.sign(nu11))/2)*90

    if ang180>=180:
        ang180-=360
    if ang180<-180:
        ang180+=360
    return ang180

def flipOrNot(ex):
    rows,cols = ex.shape
    M=moments(ex)
    x=M['m10']/M['m00'] - cols/2.
    y=M['m01']/M['m00'] - rows/2.
    if abs(x)>abs(y):
        return x<0
    else:
        return y<0


def normalize_angle(ex):
    rows,cols = ex.shape
    ang=angle(ex)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-ang,1)
    dst= cv2.warpAffine(ex,M,(cols,rows))*footprint*1
    if flipOrNot(dst):
        M180 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        dst = cv2.warpAffine(dst,M180,(cols,rows))

    return ang,dst*footprint*1.

from astropy.convolution import MexicanHat2DKernel,convolve
mexicanhat_2D_kernel = 10000*MexicanHat2DKernel(10)

def find_threshold(image,percentile=0.9):
    V=sorted(image.flatten())
    l=len(V)
    thr=V[int(l*percentile)] #consider only peaks in the top 5%
    return thr

def normalize(window,range=[0,1],dtype=np.float32):
    _max=max(window.flatten())
    _min=min(window.flatten())
    return np.array((window-_min)/(_max-_min),dtype=dtype)

from photutils.detection import find_peaks

def extract_patches(_mean,Peaks):
    markers=_mean*np.float32(0)
    stamp=footprint*np.float32(0.2)

    X=list(Peaks["x_peak"])
    Y=list(Peaks["y_peak"])

    extracted=[]
    for i in range(len(X)):
        corner_x=np.uint16(X[i]-center)
        corner_y=np.uint16(Y[i]-center)

        # ignore patches that extend outside of window
        if(corner_x<0 or corner_y<0 or \
           corner_x+cell_size>markers.shape[1] or corner_y+cell_size>markers.shape[0]):
            continue

        # mark location of extracted patches
        markers[corner_y:corner_y+cell_size,corner_x:corner_x+cell_size]=stamp
        # extract patch
        ex=np.array(_mean[corner_y:corner_y+cell_size,corner_x:corner_x+cell_size])
        ex *= mask

        #normalize patch interms of grey values and in terms of rotation
        _m,_std,ex_grey_normed=normalize_greyvals(ex)
        rot_angle1,ex_rotation_normed=normalize_angle(ex_grey_normed)
        rot_angle2,ex_rotation_normed=normalize_angle(ex_rotation_normed)
        extracted.append((_m,_std,rot_angle1+rot_angle2,ex_rotation_normed*mask))
    return extracted,markers
    

def preprocess(window):
    _mean=normalize(np.mean(window,axis=2))
    P=convolve(_mean,mexicanhat_2D_kernel)
    thr=find_threshold(P,0.9)
    Peaks=find_peaks(P,thr,footprint=footprint)
    return _mean,P,Peaks

window=pickle.load(open('../data/window.pkl','rb'))
cell_size,center,ratio,footprint=create_footprint(cell_size=41)
mask=1.*footprint
_mean,P,Peaks=preprocess(window)    
print('found',len(Peaks),'patches')
extracted,markers=extract_patches(_mean,Peaks)


pickle.dump(extracted,open('../data/extracted.pkl','wb'))