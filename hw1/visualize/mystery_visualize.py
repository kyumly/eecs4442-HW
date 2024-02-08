#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

def griad(X):
    fig = plt.figure(figsize=(10, 10))
    #X=np.log(X+1)
    k = X[:, :, 8]
    for i in range(0, 9):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(X[:, :, i],cmap="plasma" )  # 이미지를 표시
    plt.show()


def grid_save(X):
    for i in range(9):
       print(i)
       plt.imsave(f"./result{name}/vis_{i}.png" ,X[:,:,i],cmap="plasma")

def colormapArray(X, colors, name=0):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    griad(X)
    #grid_save(X)

    return None


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata.npy")
    data = np.load("mysterydata/mysterydata2.npy")
    name = 1

    #pdb.set_trace()
    colormapArray(data, colors, name)
