from scipy.io import loadmat
from scipy.io import savemat
from scipy.spatial.distance import cdist
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from scipy.io import savemat
from sklearn.preprocessing import scale, minmax_scale, normalize
import os
from sklearn.decomposition import PCA


if __name__ == '__main__':
    root = 'F:\\LinuxFile\\GSENet_final\\datasets\\'
    data_path = '../ERS/superpixel_result2'
    data_name = ['Indian_pines_corrected', 'Salinas_corrected', 'Houston', 'PaviaU']
    # data_name = ['Indian_pines_corrected']
    gt_name = ['Indian_pines_gt', 'Salinas_gt', 'Houston_gt', 'PaviaU_gt']
    superpixel_num = np.linspace(1000, 2000, 11, dtype=int)
    for i in range(len(data_name)):
        gt_path = root + gt_name[i] + '.mat'
        gt = loadmat(gt_path)
        gt = gt[list(gt.keys())[-1]]
        img_path = root + data_name[i] + '.mat'
        img = loadmat(img_path)
        img = img[list(img.keys())[-1]]
        n_row, n_column, n_band = img.shape
        img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)
        pca = PCA(n_components=0.99)
        img = pca.fit_transform(scale(img.reshape(-1, n_band))).reshape(n_row, n_column, -1)
        print('PCA shape', img.shape)
        folder = "{}_result".format(data_name[i])
        if not os.path.exists(folder):
            os.mkdir(folder)
        for inum in superpixel_num:
            filename = data_path + '/' + data_name[i] + '_sp_num_' + str(inum) + '.mat'
            superpixel_mat = loadmat(file_name=filename)['sp_map']

            color = (162 / 255, 169 / 255, 175 / 25)
            print(np.min(superpixel_mat))
            pos = np.where(gt == 0)
            superpixel_mat[pos] = 0
            x = minmax_scale(img.reshape(superpixel_mat.shape[0] * superpixel_mat.shape[1], -1))
            x = x.reshape(superpixel_mat.shape[0], superpixel_mat.shape[1], -1)
            print(x.shape)
            print(superpixel_mat.shape)
            mask = mark_boundaries(x[:, :, :3], superpixel_mat, color=(1, 1, 0), mode='subpixel')
            fig = plt.figure()
            plt.imshow(mask)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
