import os

import cv2
import numpy as np
import scipy.ndimage
from common import read_img, save_img
import matplotlib.pyplot as plt



def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    #image = np.arange(0,81).reshape(9,9)

    h, w = image.shape
    #patch_size = (2,2)

    y_center, x_center = patch_size[0] // 2, patch_size[1] // 2
    num_h_even =   patch_size[0] % 2
    num_w_even =   patch_size[1] % 2

    for j in range(y_center, h - y_center, patch_size[0]):
        if num_h_even == 0:
            y1, y2 = j - y_center, j + y_center
        else:
            y1, y2 = j - y_center, j + y_center + 1

        for i in range(x_center, w - x_center, patch_size[1]):
            if num_w_even == 0:
                x1, x2 = i - x_center, i + x_center
            else:
                x1, x2 = i - x_center, i + x_center + 1
            a = image[y1:y2, x1:x2].astype("float32")
            a : np.ndarray

            mean = a.mean()
            std = a.std()
            a = (a-mean)/std
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            output.append(a)
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    output =np.zeros_like(image).astype(np.float32)
    kernel_H, kernel_W = kernel.shape

    H_padding = (kernel_H - 1) / 2
    W_padding = (kernel_W - 1) / 2

    H_padding_int = int(H_padding)
    W_padding_int = int(W_padding)

    H_mask = H_padding - H_padding_int
    W_mask = W_padding - W_padding_int

    if H_mask == 0 and W_mask== 0:
        img = np.pad(image, pad_width=((H_padding_int, H_padding_int), (W_padding_int, W_padding_int)))
    elif 0 < H_mask < 1 and 0 < W_mask < 1:
        img = np.pad(image, pad_width=((H_padding_int, H_padding_int+1), (W_padding_int+1, W_padding_int)))

    # 다시 설정
    image = img

    H, W = image.shape
    kernel_h_center, kernel_w_center, = kernel.shape[0] // 2, kernel.shape[1] // 2
    num_h_even = kernel.shape[0] % 2
    num_w_even = kernel.shape[1] % 2

    for start_h in range(kernel_h_center, H - kernel_h_center):
        if num_h_even == 0:
            y1, y2 = start_h - kernel_h_center, start_h + kernel_h_center
        else:
            y1, y2 = start_h - kernel_h_center, start_h + kernel_h_center + 1

        for start_w in range(kernel_w_center, W - kernel_w_center):
            if num_w_even == 0:
                x1, x2 = start_w - kernel_w_center, start_w + kernel_w_center
            else:
                x1, x2 = start_w - kernel_w_center, start_w + kernel_w_center + 1

            roi = np.flip(image[y1:y2, x1:x2].astype(np.float32))
            #roi = image[y1:y2, x1:x2].astype(np.float32)

            tmp = roi * kernel
            output[start_h-kernel_h_center, start_w-kernel_w_center] = cv2.sumElems(tmp)[0]

    return output
def convolve2(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    output = np.zeros_like(image)
    if len(kernel.shape) == 2:
        kernel = kernel[::-1, ::-1]
    elif len(kernel.shape) == 1:
        kernel = kernel[::-1]

    H, W = image.shape
    h, w = kernel.shape

    padding = [h // 2, w // 2]
    padded_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    print(padded_image)
    padded_image[padding[0]: H + padding[0], padding[1]: W + padding[1]] = image

    for y in range(H):
        for x in range(W):
            patch = padded_image[y: y + h, x: x + w]
            output[y, x] = np.sum(patch * kernel)
    return output
def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([-1,0,1]).reshape(1,3)  # 1 x 3
    ky = np.array([-1,0,1]).reshape(3,1)  # 3 x 1

    #st = np.array([1,1,1])
    #kx = st.reshape(3,1) * kx
    #ky = st.reshape(1,3) * ky

    # if len(kernel.shape) == 2:
    # kernel = kernel[ : :-1 , : :-1]
    # elif len(kernel.shape) == 1:
    #     kernel = kernel[ : :-1]

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)
    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude =cv2.magnitude(Ix, Iy)
    #grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    Ix = cv2.convertScaleAbs(Ix)
    Iy = cv2.convertScaleAbs(Iy)
    grad_magnitude = cv2.convertScaleAbs(grad_magnitude)
    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None
    gx_mask = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    gy_mask = np.array([[
        1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])
    Gx = convolve(image, gx_mask)
    Gy = convolve(image, gy_mask)
    grad_magnitude =cv2.magnitude(Gx, Gy)

    Gx = cv2.convertScaleAbs(Gx)
    Gy = cv2.convertScaleAbs(Gy)
    grad_magnitude = cv2.convertScaleAbs(grad_magnitude)


    return Gx, Gy, grad_magnitude




def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally

    idxs = [np.random.randint(0, len(patches)) for _ in range(3)]
    chosen_patches = np.array([patches[i] for i in idxs])
    chosen_patches = chosen_patches.reshape(16, -1)

    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code
    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code
    # (b): Complete convolve()
    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.

    kernel_size = 3
    kernel_sigma = 0.572
    # kernel_sigma = 2
    kernel_range = np.arange(-1*(kernel_size//2), (kernel_size //2) + 1)
    kernel_range = np.array([float(x) for x in kernel_range])

    # 커널 설명
    # arr = np.zeros((kernel_size, kernel_size))
    #
    # for x in range(kernel_size):
    #     for y in range(kernel_size):
    #         arr[x,y] = kernel_range[x]**2+kernel_range[y]**2
    # kernel_2d = np.zeros((kernel_size, kernel_size))
    #
    # for x in range(kernel_size):
    #     for y in range(kernel_size):
    #          kernel_2d[x,y] = np.exp(-arr[x,y]/(2*kernel_sigma**2))
    # kernel_2d /= kernel_2d.sum()

    x, y = np.meshgrid(kernel_range, kernel_range)
    sums = x**2 + y**2
    sums = np.exp(-sums/(2*kernel_sigma**2))
    sums /= sums.sum()
    kernel_gaussian = sums

    filtered_gaussian = convolve(img, kernel_gaussian)
    print(filtered_gaussian[0][0])
    print("결과 : ", scipy.ndimage.convolve(img, kernel_gaussian, mode='constant')[0][0])
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img,kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load("log1d.npz")
    plt.figure(1)
    plt.plot(data['log50'])
    plt.plot(data['gauss53'] - data['gauss50'])
    plt.legend(['Original', 'Approx'])
    plt.show()
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
