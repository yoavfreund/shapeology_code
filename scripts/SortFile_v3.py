import os
import numpy as np
import pickle as pk
import numpy as np
from glob import glob
from time import time
import sys
sys.path.append('../scripts/')
from lib.utils import configuration, setup_download_from_s3
from lib.permute import permutator

time_log=[]
def clock(message):
    print('%8.1f \t%s'%(time(),message))
    time_log.append((time(),message))

class Sorter:
    """Process the patch files generated by extractPatches.py and prepare
    them for analysis by Kmeans and diffusion-maps"""
    def __init__(self, K=20, src_root='/tmp'):
        '''
        Initialize a files sorter.
        :param K: number of final files
        :param src_root: directory for source files without permutation
        '''
        self.src_root = src_root
        assert os.path.exists(self.src_root)
        self.K = K

    def sort_file(self, size, stem='permuted'):
        '''
        Create K files to collect cells from each section based on the proportion of the number of cells in each section to the total.
        One file contains 100000 cells.
        :param size: size of patches
        :param stem: name stem of directory to store the final results
        :return:
        '''
        # V = pk.load(open(pkl_file, 'rb'))
        self.size = size
        self.dir = self.src_root + 'cells-'+str(size)+'/'
        self.saveDir = stem + '-' + str(size) + '/'
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        total = sum([os.path.getsize(os.path.join(r, file)) for r, d, files in os.walk(self.dir) for file in files])
        files = [fn for fn in glob(self.dir + '*.bin')]
        prob = [os.path.getsize(fn)/total for fn in glob(self.dir + '*.bin')]
        self.count = []
        for i in range(len(files)):
            self.count.append(0)
        for i in range(self.K):
            self.fp = open(self.saveDir + '/permuted-' + str(i) + '.bin', 'bw')
            for choice in np.random.choice(len(files),100000,p=prob):
                fn = files[choice]
                V = np.fromfile(fn, np.float16)
                V = V.reshape([-1,self.size**2])
                self.fp.write(V[self.count[choice], :])
                self.count[choice] += 1
            clock(str(i) + ' files finished')



if __name__=='__main__':

    yamlfile = os.environ['REPO_DIR'] + 'shape_params-aws.yaml'
    params = configuration(yamlfile).getParams()
    stack = 'DK39'
    root_dir = os.environ['ROOT_DIR']

    clock('Process Begin')
    t0 = time()
    setup_download_from_s3(stack+'/cells/')
    clock('Download From S3')
    sorter = Sorter(src_root=root_dir + stack + '/cells/')
    size_thresholds = params['normalization']['size_thresholds']
    for size in size_thresholds:
        sorter.sort_file(size, stem=root_dir + 'permute/permuted')
        clock('Complete files of size '+str(size))
        print('Complete files of size '+str(size), time() - t0, 'seconds')
    log_fp = 'TimeLog/'
    if not os.path.exists(log_fp):
        os.mkdir(log_fp)
    pk.dump(time_log,open(log_fp+'Time_log_permute_v3.pkl','wb'))