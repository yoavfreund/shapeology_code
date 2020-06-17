"""
map each patch into diffusion-map coordinates.
"""
import pickle as pk
import pydiffmap
import sys
import numpy as np
from lib.utils import calc_err

# Pre-process patch to reduce resolution and filter if too
# noisy. (have flag in the label for that)
class diffusionMap:
    def __init__(self,dmapFilename):
        """diffusionMap labels a patch using a diffusion map

        :param dmapFilename: name of pickle file containing the pydiffmap.dmap object 
        """
        self.DM=pk.load(open(dmapFilename,'rb'))
        # data=self.DM.data
        # self.DM = self.DM.from_sklearn(alpha=self.DM.alpha, k=self.DM.k, kernel_type=self.DM.kernel_type, epsilon=self.DM.epsilon, \
        #                            n_evecs=self.DM.n_evecs, neighbor_params=self.DM.neighbor_params)
        # self.DM.fit(data)
        #print(type(self.DM))
    

    def transform(self,data1D):
        """compute the dmap transformation for a list of square patches.

        :param data1D a two dimensional vector where each row is a vector corresponding to a flattened image
        :returns: an array of projected values (shape =  length of list * number of eigenvectors)
        :rtype: numpy array
        """
        lowD=self.DM.transform(data1D)
        return lowD

    def label_patch(self,Patches,smooth_threshold=0.4):
        """compute a label for each raw patch.

        preprocessing replicates the one used in the vector quantization:
        for each patch:
        compare image to smoothed image and if difference is large: reject (return None)
        otherwise reduce resolution by 2 and call transform.

        :param Patches: List of patches
        :param smooth_threshold: if err>smooth_threshold, return None for this patch
        :returns: returns a list of the same length as Patches where each element is either None or a 1d numpy array
        :rtype: list

        """
        _sub_size=51
        data1D=np.zeros((len(Patches),_sub_size*_sub_size))
        for i in range(len(Patches)):
            data1D[i,:] = Patches[i].flatten();
        dmap=self.transform(data1D)
        return(dmap)



