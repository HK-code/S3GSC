import pickle

from skimage.segmentation import slic, mark_boundaries, find_boundaries, felzenszwalb
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize
import os
import h5py
from sklearn.decomposition import PCA


def load_data(img_path, gt_path):
    if img_path[-3:] == 'mat':
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        gt_mat = sio.loadmat(gt_path)
        img_keys = img_mat.keys()
        gt_keys = gt_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        return img_mat.get(img_key[0]).astype('float64'), gt_mat.get(gt_key[0]).astype('int8')


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X  # 给数据加上一圈的0值
    return newX


def calNoZeroNumber(y):
    count = 0
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            if y[r, c] == 0:
                continue
            else:
                count += 1
    return count


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 统计非零标签像素的个数
    NoZeroNumber = calNoZeroNumber(y)
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # 以该数据为例创建一个（512*217,25,25,30）的四维矩阵用于存储每一个（25,25,30）的数据块
    # 并创建一个（512*127）的标签
    patchesData = np.zeros((NoZeroNumber, windowSize, windowSize, X.shape[2]))
    print("patchesData shape is", patchesData.shape)
    patchesLabels = np.zeros((NoZeroNumber))
    patchIndex = 0
    spatial_list = np.zeros((NoZeroNumber, 2))
    # 遍历数据，得到（25,25,30）的数据和数据标签
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] == 0:
                continue
            else:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                spatial_list[patchIndex, :] = [r, c]
                patchIndex = patchIndex + 1
    patchesLabels -= 1
    # 去除标签为0的无效数据,且标签减1，数据标签从0开始
    print("---遍历完成---")
    # if removeZeroLabels:
    #     patchesData = patchesData[patchesLabels > 0, :, :, :]
    #     patchesLabels = patchesLabels[patchesLabels > 0]
    #     patchesLabels -= 1
    return patchesData, patchesLabels, spatial_list


def standardize_label(y):
    """
    standardize the classes label into 0-k
    :param y:
    :return:
    """
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


if __name__ == '__main__':
    root = 'F:\\LinuxFile\\GSENet_final\\datasets\\'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'PaviaU', 'PaviaU_gt'
    # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'
    # im_, gt_ = 'Houston', 'Houston_gt'
    # data_name = ['Indian_pines_corrected', 'Salinas_corrected', 'Houston', 'PaviaU']
    # gt_name = ['Indian_pines_gt', 'Salinas_gt', 'Houston_gt', 'PaviaU_gt']
    # data_name = ['WHU_Hi_LongKou']
    # gt_name = ['WHU_Hi_LongKou_gt']
    data_name = ['HoustonU']
    gt_name = ['HoustonU_gt']
    for im_, gt_ in zip(data_name, gt_name):
        img_path = root + im_ + '.mat'
        gt_path = root + gt_ + '.mat'
        if im_ == 'xiongan':
            h5file_img = h5py.File(img_path, mode='r')
            img = np.array(h5file_img.get('XiongAn'))
            img = img.transpose(1, 2, 0)
            h5file_gt = h5py.File(gt_path, mode='r')
        elif im_ == 'HoustonU':
            print(img_path)
            h5file_img = h5py.File(img_path, mode='r')
            img = np.array(h5file_img.get('houstonU'))
            img = img.transpose(1, 2, 0)
            print(img.shape)
            # h5file_gt = h5py.File(gt_path, mode='r')
        else:
            img, gt = load_data(img_path, gt_path)

        NEIGHBORING_SIZE = 13
        nb_comps = 3

        #
        # 创建结果文件夹
        folder = "PCA_result"
        if not os.path.exists(folder):
            os.mkdir(folder)

        n_row, n_column, n_band = img.shape
        img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)
        # perform PCA
        pca = PCA(n_components=nb_comps)
        img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
        print(img.shape)
        print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
        pca_name = folder + '\\{}_pca.mat'.format(im_)
        savemat(pca_name, {'data': img})
