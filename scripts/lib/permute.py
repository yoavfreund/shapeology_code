import numpy as np
from numpy.random import permutation
from os import mkdir, remove
from glob import glob

class permutator:
    """ A class that creates a random permutation of equal-size bytearrays"""
    N=0 #permutor index (used to differentiate files) 
    def __init__(self,element,K=100):  #number of files used for randomization
        assert type(element) == bytearray
        self.nbytes=len(element)
        print(self.nbytes,'bytes per element')
        self.tmpDir='/tmp/permuted-%d'%permutator.N
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
        assert type(element) == bytearray
        if len(element) != self.nbytes:
            throw('Inconsistent element size: init=%d, current=%d'%(self.nbytes,element.nbytes))
        j=np.random.randint(self.K)
        #print(type(element),len(Bytes))
        self.fp[j].write(element)
    
    def combine(self,outfilename):
        # read and permute each file
        for i in range(self.K):
            self.fp[i].close()
            self.fp[i]=open(self.tmpDir+'/bucket-'+str(i)+'.bin','br')
        outfile=open(outfilename,'bw')
        for filename in self.fp:
            D=filename.read()
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
                throw('incorrect file size:',filename,'shape=',elements.nbytes,'self.nbytes=',self.nbytes,'error=',error)
