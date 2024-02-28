#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def data_preprocessing(image):
    max = np.nanmean(image, axis=(0,1))
    min = np.nanstd(image, axis=(0,1))

    mask = np.isfinite(image)
    for i in range(9):
        image[:, :, i][~mask[:, :, i]] = max[i]

    return image

def print_matInfo(name, image):
    #print(np.isnan(image[:, :, 0]).sum(axis=1))

    if image.ndim >= 3 :
        image = image[0, 0,:]
    else:
        image = image[0,0]
    if image.dtype == 'uint8':     mat_type = "CV_8U"
    elif image.dtype == 'uint32':     mat_type = "CV_32U"
    elif image.dtype == 'int8':    mat_type = "CV_8S"
    elif image.dtype == 'uint16':  mat_type = "CV_16U"
    elif image.dtype == 'int16':   mat_type = "CV_16S"
    elif image.dtype == 'float32': mat_type = "CV_32F"
    elif image.dtype == 'float64': mat_type = "CV_64F"

    nchannel = 3 if image.ndim == 3 else 1

    ## depth, channel 출력
    print("%12s: depth(%s), channels(%s) -> mat_type(%sC%d)"
          % (name, image.dtype, nchannel, mat_type,  nchannel))
def grid(X):
    fig = plt.figure(figsize=(10,10))

    N = X.shape[-1]
    for i in range(N):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(X[:, :, i], cmap='plasma')

    plt.show()


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    # print(X.shape)
    # print(colors.shape)
    # fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9)) #  서브 플롯 새성
    #
    # for i in range(9):
    #     #plt.imsave("vis_%d.png" % i,X[:,:,i])
    #     ax.imshow(X[:,:,i], camp='gray')
    # plt.imshow()

    # X = data_preprocessing(X)
    #
    # x_mean, x_std =cv2.meanStdDev(X)
    # print_matInfo("X", X)
    #
    # print(f"평균값 {x_mean.reshape(-1)} \n"
    #       f" 분산값 : {x_std.reshape(-1)}")

    X = np.log1p(X)
    X = data_preprocessing(X)


    xt_mean, xt_std = cv2.meanStdDev(X)
    print_matInfo("X transform", X)
    print(f"평균값 {xt_mean.reshape(1, -1)} \n"
          f" 분산값 : {xt_std.reshape(1, -1)}")

    print()

    grid(X)

    return None


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata3.npy")

    colormapArray(data, colors)
    #pdb.set_trace()
