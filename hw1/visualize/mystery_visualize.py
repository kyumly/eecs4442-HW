#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb




def data_mapping2(X, colors):
    print(colors.shape)
    # 플로팅
    # get_spectrum(colors)

    channels = cv2.split(X)

    W, H, N = data.shape

    data_total = np.zeros((N, W, H, 3))

    for index, channel in enumerate(channels):
        merge_list = []
        ex = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        ex = channel
        for color in colors.T:
            """
            R, G , B 순서
            """
            colors_image = cv2.LUT(ex, color)
            merge_list.append(colors_image)

        image = np.stack(merge_list, axis=2)
        #image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
        data_total[index] = image


    return data_total


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
    if image.dtype == 'uint8':
        mat_type = "CV_8U"
    elif image.dtype == 'uint32':
        mat_type = "CV_32U"
    elif image.dtype == 'int8':
        mat_type = "CV_8S"
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

    #X = data_preprocessing(X)
    X = X.astype(np.uint8)
    images = data_mapping2(X, colors)
    images = (images * 255).astype(np.uint8)
    x_mean, x_std =cv2.meanStdDev(X)
    print_matInfo("X", X)

    print(f"평균값 {x_mean.reshape(-1)} \n"
          f" 분산값 : {x_std.reshape(-1)}")
    #grid(X)


    fig = plt.figure(figsize=(10,10))

    N = images.shape[0]

    for i in range(N):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(images[i], )

    plt.show()

    return None


def get_spectrum(colors):
    plt.imshow(colors[:, 0].reshape(1, -1), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Colormap Array')
    plt.xlabel('Value')
    plt.ylabel('Index')
    plt.show()




if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata4.npy")

    #pdb.set_trace()

    colormapArray(data, colors)
