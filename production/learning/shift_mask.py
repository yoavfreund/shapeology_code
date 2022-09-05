import numpy as np
import pickle
from time import time
import os
from matplotlib.path import Path
from shapely.geometry import Polygon

class mask_generator:
    def __init__(self, structure, annotation_file=os.environ['ROOT_DIR']+'DK43/DK43_rough_landmarks.pkl'):
        self.struc = structure
        assert os.path.exists(annotation_file)
        # assert os.path.exists(com_file)
        self.contours = pickle.load((open(annotation_file, 'rb')))
        # self.COMs = pickle.load(open(com_file,'rb'))

    def decentralization(self):
        C = {i:self.contours[i][self.struc] for i in self.contours if self.struc in self.contours[i]}
        concat = np.concatenate([C[i] for i in C])
        center = np.mean(concat, axis=0)
        self.max_x, self.max_y = np.max(concat,axis=0) - center
        self.min_x, self.min_y = np.min(concat,axis=0) - center
        for i in C:
            C[i] = C[i] - center
        self.C = C
        self.resol = 0.325
        self.margin = 200/self.resol
        self.step_size = round(20/self.resol)

    def fixed_mask(self,save_dir=os.environ['ROOT_DIR'] + 'Detection_preparation_v2/'):
        x_grid = np.arange(self.min_x - self.margin, self.max_x + self.margin, self.step_size)
        y_grid = np.arange(self.min_y - self.margin, self.max_y + self.margin, self.step_size)
        section_numbers = sorted(self.C.keys())
        grid3D = np.zeros([len(section_numbers), x_grid.shape[0], y_grid.shape[0]])

        section_index = 0
        total_shape_area = {}
        total_sur_area = {}
        len_max = 0
        for section in section_numbers:
            one_sect = self.C[section]
            path = Path(one_sect)

            inside_area = Polygon(one_sect).area
            total_shape_area[section - section_numbers[0]] = inside_area
            outside_area = Polygon(one_sect).buffer(self.margin, resolution=2).area - inside_area
            total_sur_area[section - section_numbers[0]] = outside_area

            length = one_sect[:, 0].max() - one_sect[:, 0].min()
            width = one_sect[:, 1].max() - one_sect[:, 1].min()
            if max(length, width) > len_max:
                len_max = max(length, width)

            grid = np.zeros([x_grid.shape[0] * y_grid.shape[0], 2])
            for i in range(x_grid.shape[0] * y_grid.shape[0]):
                grid[i, 0] = x_grid[i % x_grid.shape[0]]
                grid[i, 1] = y_grid[int(i / x_grid.shape[0])]

            in_out = path.contains_points(np.array(grid[:, :]))
            surround = Polygon(one_sect).buffer(self.margin, resolution=2)
            path = Path(list(surround.exterior.coords))
            outside = path.contains_points(np.array(grid[:, :]))

            grid3D[section_index, :, :] = 1 * in_out.reshape([y_grid.shape[0], -1]).T + 1 * outside.reshape(
                [y_grid.shape[0], -1]).T
            section_index += 1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fn = save_dir + self.struc + '.pkl'
        print(grid3D.shape, sum(list(total_shape_area.values())), self.min_x, self.min_y, len_max)
        pickle.dump([grid3D, total_shape_area, total_sur_area, self.min_x, self.min_y, len_max], open(fn, 'wb'))

    def shift_mask(self,mode='search',src_doot=os.environ['ROOT_DIR'] + 'Detection_preparation_v2/',\
                   save_dir=os.environ['ROOT_DIR'] + 'Detection_preparation_mask/'):
        fn = src_doot + self.struc + '.pkl'
        assert os.path.exists(fn)
        save_dir += mode + '/' + self.struc + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(fn, 'rb'))
        if mode=='search':
            shift_step_size = max(round(len_max / 20), round(30 / self.resol))
        elif mode=='refine':
            shift_step_size = max(round(len_max / 30), round(20 / self.resol))
        half = 15
        x_grid = np.arange(self.min_x-self.margin-shift_step_size*half, self.max_x+self.margin+shift_step_size*half, self.step_size)
        y_grid = np.arange(self.min_y-self.margin-shift_step_size*half, self.max_y+self.margin+shift_step_size*half, self.step_size)
        section_numbers = sorted(self.C.keys())

        grid = np.zeros([x_grid.shape[0] * y_grid.shape[0], 2])
        for i in range(x_grid.shape[0] * y_grid.shape[0]):
            grid[i, 0] = x_grid[i % x_grid.shape[0]]
            grid[i, 1] = y_grid[int(i / x_grid.shape[0])]

        for section in section_numbers:
            grid2D_inner = np.zeros([(2*half+1)**2, x_grid.shape[0], y_grid.shape[0]],dtype=np.int8)
            grid2D_outer = np.zeros([(2 * half + 1) ** 2, x_grid.shape[0], y_grid.shape[0]],dtype=np.int8)
            polygon = self.C[section]
            inside_area = Polygon(polygon).area
            outside_area = Polygon(polygon).buffer(self.margin, resolution=2).area - inside_area
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    region = polygon.copy()
                    region[:, 0] += i * shift_step_size
                    region[:, 1] += j * shift_step_size
                    path = Path(region)

                    inner = path.contains_points(np.array(grid[:, :]))
                    surround = Polygon(region).buffer(self.margin, resolution=2)
                    path = Path(list(surround.exterior.coords))
                    outside = path.contains_points(np.array(grid[:, :]))

                    grid2D_inner[(i+half)*(2*half+1)+j+half, :, :] = 1 * inner.reshape([y_grid.shape[0], -1]).T
                    grid2D_outer[(i+half)*(2*half+1)+j+half, :, :] = 1 * outside.reshape([y_grid.shape[0], -1]).T - 1 * inner.reshape([y_grid.shape[0], -1]).T

            fn = save_dir + str(section - section_numbers[0]) + '.npz'
            np.savez_compressed(fn, inner=grid2D_inner,outer= grid2D_outer)
            unique_combinations_inner, indices_inner = np.unique(grid2D_inner.reshape([grid2D_inner.shape[0], -1]).T, axis=0, return_inverse=True)
            unique_combinations_outer, indices_outer = np.unique(grid2D_outer.reshape([grid2D_outer.shape[0], -1]).T, axis=0,
                                                     return_inverse=True)
            indices_inner = indices_inner.reshape([grid2D_inner.shape[1],-1])
            indices_outer = indices_outer.reshape([grid2D_inner.shape[1], -1])
            fn = save_dir + str(section - section_numbers[0]) + '.pkl'
            pickle.dump([unique_combinations_inner,indices_inner,unique_combinations_outer,indices_outer],\
                        open(fn,'wb'))







