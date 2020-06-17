import numpy as np
from numpy.random import permutation
import os
from os import mkdir, remove
from glob import glob

class permutator:
    """ A class that creates a random permutation of equal-size bytearrays"""
    N=0 #permutor index (used to differentiate files) 
    def __init__(self,element,K=100,temp_root='/tmp'):
        '''
        Initialize a permutator.
        :param element: type=bytearray, bytearray of a padded patch, claiming the size of bytearrays
        :param K: number of files used for randomization
        :param temp_root: directory for temporary files without permutation
        '''
        assert type(element) == bytearray
        self.nbytes=len(element)
        print(self.nbytes,'bytes per element')
        if not os.path.exists(temp_root):
            os.makedirs(temp_root)
        self.tmpDir=temp_root+'/permuted-%d'%permutator.N
        self.K=K
        permutator.N+=1
        try:
            mkdir(self.tmpDir)
            print('made dir ',self.tmpDir)
        except:
            print(self.tmpDir,'already exists, removing contents')
            for file in glob(self.tmpDir+'/*'):
                remove(file)
        
        self.fp=[]
        for i in range(K):
            self.fp.append(open(self.tmpDir+'/bucket-'+str(i)+'.bin','bw'))

    def push(self,element):
        '''
        Store a patch into a random file.
        :param element: bytearray of a padded patch
        :return:
        '''
        assert type(element) == bytearray
        if len(element) != self.nbytes:
            print('Inconsistent element size: init=%d, current=%d'%(self.nbytes,element.nbytes))
        j=np.random.randint(self.K)
        #print(type(element),len(Bytes))
        self.fp[j].write(element)
    
    def combine(self,outfilename):
        '''
        Read and permute each file randomly collecting patches.
        :param outfilename: the directory to store the final results
        :return:
        '''
        for i in range(self.K):
            self.fp[i].close()
            self.fp[i]=self.tmpDir+'/bucket-'+str(i)+'.bin'
        order = permutation(len(self.fp))
        self.fp = np.array(self.fp)
        self.fp = self.fp[order[:20]]
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)

        for filename in self.fp:
            id = filename[filename.rfind('-')+1:filename.rfind('.bin')]
            outfile = open(outfilename + '/permuted' + id + '.bin', 'bw')
            file = open(filename, 'br')
            D = file.read()
            #print('buffer length',len(D))
            D=np.frombuffer(D,dtype=np.byte)
            #print(D.shape,type(D[0]))
            elements=D.reshape([-1,self.nbytes])
            #print(elements.shape)

            L=elements.shape[0]
            _order=permutation(L)
            permuted_elements=elements[_order,:]
            permuted_elements.tofile(outfile)
            error=elements.nbytes % self.nbytes
            if error!=0:
                print('incorrect file size:',filename,'shape=',elements.nbytes,'nbytes=',self.nbytes,'error=',error)
