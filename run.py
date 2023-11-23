# encoding: utf-8

import pickle
import sys
import time
import h5py
from scipy.io import loadmat
from scipy.io import savemat
import os

from skimage.segmentation import felzenszwalb
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.neighbors import kneighbors_graph

sys.path.append('.')
from ClusteringEvaluator import cluster_accuracy
from model import DVSGSC
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from scipy.io import savemat
from yaml_config_hook import *
import argparse
sys.path.append('../')


def cal_center_feature(img, pos_x, pos_y):
    pos = np.stack((pos_x, pos_y), axis=-1)
    features = img[pos_x, pos_y, :]
    center_feature = np.mean(features, axis=0)
    center_pos = np.mean(pos, axis=0)
    return center_feature, center_pos


def cal_superpixel_feature(superpixel_mat, img):
    height, width, bands = img.shape
    sp_nums = np.unique(superpixel_mat).shape[0] - 1
    sp_features = np.zeros((sp_nums, bands))
    sp_center_pos = np.zeros((sp_nums, 2))  # superpixel center position
    sp_var = np.zeros(sp_nums)  # superpixel variance
    for i in range(1, len(np.unique(superpixel_mat))):  # remove label = 0
        sp_ilabel = np.unique(superpixel_mat)[i]
        x, y = np.where(superpixel_mat == sp_ilabel)
        center_feature, center_pos = cal_center_feature(img, x, y)
        sp_features[i - 1, :] = center_feature
        sp_center_pos[i - 1, :] = center_pos
    return sp_features, sp_center_pos


def creat_sp_graph(superpixel_mat, sp_center_pos, sp_feature, k1, k2):
    sp_nums = np.unique(superpixel_mat).shape[0] - 1
    norm_feature = normalize(sp_feature, norm='l2', axis=1)
    sp_graph1 = kneighbors_graph(sp_feature, n_neighbors=k2, include_self=True).toarray()
    sp_graph2 = np.dot(norm_feature, norm_feature.T)
    sp_graph = sp_graph1 * sp_graph2

    # resume time
    sp_distance = kneighbors_graph(sp_center_pos, n_neighbors=k1, include_self=True).toarray()
    pos_x, pos_y = np.where(sp_distance == 1)
    for x, y in zip(pos_x, pos_y):
        a_feature = sp_feature[x, :]
        b_feature = sp_feature[y, :]
        a2 = np.dot(a_feature, b_feature) / (np.linalg.norm(a_feature) * np.linalg.norm(b_feature))
        a1 = 1
        SSI = a1 * a2
        sp_distance[x, y] = sp_distance[x, y] * SSI
    return sp_distance, sp_graph


def order_sam_for_diag(x, y):
    """
    rearrange samples
    :param x: feature sets
    :param y: ground truth
    :return:
    """
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        start = stop
    return x_new, y_new


def standardize_label(y):
    """
    standardize the classes label into 0-k
    :param y:
    :return:
    """

    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


def sp_label_to_pixel_label(y_pre, sp_label):
    """
    superpixel label to pixel label
    :param y_pre: spectral clustering result [1, ... k]
    :param sp_label: ERS or Slic segmentation result
    :return:
    """
    sp_label_copy = copy.deepcopy(sp_label)
    for i in range(1, len(np.unique(sp_label))):
        i_sp_label = np.unique(sp_label)[i]
        pos = np.where(sp_label == i_sp_label)
        sp_label_copy[pos] = y_pre[i - 1]
    y_ = sp_label_copy[np.nonzero(sp_label_copy)]
    y_pre = standardize_label(y_)
    return y_pre


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/salinas.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    root, SpmapPath, data_name, gt_name = args.root, args.spmap, args.im_, args.gt_

    gt_path = root + gt_name + '.mat'
    gt = loadmat(gt_path)
    gt = gt[list(gt.keys())[-1]]
    img_path = root + data_name + '.mat'
    img = loadmat(img_path)
    img = img[list(img.keys())[-1]]
    superpixel_mat = loadmat(SpmapPath+data_name+"_superpixelmap.mat")['sp_map']

    k1, k2, D = args.k1, args.k2, args.D

    gt_zero_pos = np.where(gt == 0)
    superpixel_mat[gt_zero_pos] = 0  # remove nonlabeled pixel

    sp_feature, sp_center_pos = cal_superpixel_feature(superpixel_mat, img)
    sp_nums, n_feature = sp_feature.shape
    sp_feature = minmax_scale(sp_feature)
    # # reduce spectral bands using PCA
    pca = PCA(n_components=D)
    sp_feature = minmax_scale(pca.fit_transform(sp_feature))
    spatial_graph, spectral_graph = creat_sp_graph(superpixel_mat, sp_center_pos, sp_feature, k1, k2)
    for i in range(10):
        time_start = time.time()
        y_ = gt[np.nonzero(gt)]
        y = standardize_label(y_)
        print('sample number: {}, label number: {}'.format(sp_feature.shape, y.shape))
        N_CLASSES = np.unique(y).shape[0]  # Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

        x_patches_2d = np.transpose(normalize(sp_feature))
        N_CLASSES = np.unique(y).shape[0]  # Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8
        dvsgsc = DVSGSC(n_clusters=N_CLASSES, spatial_graph=spatial_graph,
                     spectral_graph=spectral_graph,
                     alpha=1, beta=1)
        y_pre_gcsc = dvsgsc.fit(x_patches_2d)
        y_pre_gcsc = sp_label_to_pixel_label(y_pre_gcsc, superpixel_mat)
        run_time = round(time.time() - time_start, 3)
        acc = cluster_accuracy(y, y_pre_gcsc)
        print('%10s %10s %10s' % ('OA', 'NMI', 'Kappa',))
        print('%10.5f %10.5f %10.5f' % (acc[0], acc[1], acc[2]))
        print(time.time()-time_start)
