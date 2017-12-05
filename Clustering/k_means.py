# coding: utf-8

import numpy as np


def k_means(X, k_kernels=None):
    '''
    :param X: np.array, shape:[samples_num, features_num]
    :param k_kernels: np.array, shape:[k_num, feature_num]
    :return: arg_min and k_kernels
    '''

    m, n = X.shape
    if k_kernels is None:
        k_kernels = np.random.random([3, n])
    while True:
        dist = np.array([np.sum(np.square(X - kernel), axis=1) for kernel in k_kernels])
        arg_min = np.argmin(np.transpose(dist, [1, 0]), axis=1)
        keep = k_kernels
        # 空间换时间
        k_kernels = np.zeros(k_kernels.shape, np.float32)
        k_sample_nums = np.zeros(k_kernels.shape[0], np.float32)
        for x, arg in zip(X, arg_min):
            k_kernels[arg] += x
            k_sample_nums[arg] += 1

        k_sample_nums += 1e-6
        for i in range(k_kernels.shape[0]):
            k_kernels[i] /= k_sample_nums[i]

        if np.sum(np.square(k_kernels - keep)) < 1e-4:
            return arg_min, k_kernels


if __name__ == "__main__":
    arg_min, k_kernels = k_means(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1.2, 1.3, 4.5], [6.5, 3.4, 5.5]]))
    print("arg:", arg_min)
    print("k_kernel", k_kernels)