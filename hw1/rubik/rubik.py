import cv2
import matplotlib.pyplot as plt
import numpy as np

def grid(X):
    fig = plt.figure(figsize=(10,10))

    N = len(X)
    for i in range(N):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(X[i, :, :], cmap='gray')

    plt.show()
if __name__ == '__main__':
    img = cv2.imread("indoor.png", cv2.COLOR_BGR2RGB)
    r,g,b = cv2.split(img)
    grid(np.stack([r,g,b]))

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b,g,r = cv2.split(img_lab)
    grid(np.stack([b,g,r]))

