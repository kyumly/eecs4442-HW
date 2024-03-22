import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img
from hw2.filters import convolve


def corner_score2(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    output = np.zeros_like(image)
    H, W = image.shape
    h, w = window_size

    shifted_image = np.roll(image, (u, v), axis=(1, 0))

    padding = (h // 2, w // 2)

    padded_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_image[padding[0]: padding[0] + H, padding[1]: padding[1] + W] = image
    padded_shifted_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_shifted_image[padding[0]: padding[0] + H, padding[1]: padding[1] + W] = shifted_image

    for y in range(H):
        for x in range(W):
            e = np.sum((padded_shifted_image[y: y + h, x: x + w] - padded_image[y: y + h, x: x + h]) ** 2)
            output[y, x] = e

    return output

def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    output = np.zeros_like(image)

    window_size_H, window_size_W = window_size

    H_padding = (window_size_H - 1) / 2
    W_padding = (window_size_W - 1) / 2

    H_padding_int = int(H_padding)
    W_padding_int = int(W_padding)

    H_mask = H_padding - H_padding_int
    W_mask = W_padding - W_padding_int

    shift_image = np.roll(image, (u,v), axis=(1,0))

    if H_mask == 0 and W_mask == 0:
        img = np.pad(image, pad_width=((H_padding_int, H_padding_int), (W_padding_int, W_padding_int)))
        shift_image = np.pad(shift_image, pad_width=((H_padding_int, H_padding_int), (W_padding_int, W_padding_int)))

    elif 0 < H_mask < 1 and 0 < W_mask < 1:
        img = np.pad(image, pad_width=((H_padding_int, H_padding_int + 1), (W_padding_int + 1, W_padding_int)))
        shift_image = np.pad(shift_image, pad_width=((H_padding_int, H_padding_int + 1), (W_padding_int + 1, W_padding_int)))

    kernel_h_center, kernel_w_center, = window_size[0] // 2, window_size[1] // 2

    num_h_even = window_size[0] % 2
    num_w_even = window_size[1] % 2

    image = img
    H, W = image.shape

    for start_h in range(kernel_h_center, H -kernel_h_center):
        if num_h_even == 0:
            y1, y2 = start_h - kernel_h_center, start_h + kernel_h_center
        else:
            y1, y2 = start_h - kernel_h_center, start_h + kernel_h_center + 1

        for start_w in range(kernel_w_center, W - kernel_w_center):
            if num_w_even == 0:
                x1, x2 = start_w - kernel_w_center, start_w + kernel_w_center
            else:
                x1, x2 = start_w - kernel_w_center, start_w + kernel_w_center + 1
            value = np.power(shift_image[y1:y2, x1:x2] - img[y1:y2, x1:x2], 2)
            output[start_h-kernel_h_center, start_w-kernel_w_center] = np.sum(value)

    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    kx = np.array([-1,0,1]).reshape(1,-1)
    ky = np.array([-1,0,1]).reshape(-1,1)

    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0)
    # print("내값 : ", Ix[0][0])
    # Ix = convolve(image, kx)
    # print("내값 : ", Ix[0][0])
    # Iy = convolve(image, ky)


    Ixx = Ix ** 2
    Iyy = Iy ** 2

    Ixy = Ix * Iy

    M = np.zeros((3, image.shape[0], image.shape[1]))
    kernel = np.ones(window_size)

    M[0] = scipy.ndimage.convolve(Ixx, kernel, mode='constant', cval=0)
    M[1] = scipy.ndimage.convolve(Ixy, kernel, mode='constant', cval=0)
    M[2] = scipy.ndimage.convolve(Iyy, kernel, mode='constant', cval=0)

    alpha = 0.05

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    response = M[0] * M[2] - M[1] ** 2 - alpha * (M[0] + [2]) ** 2
    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    print(img.shape)
    W = (5, 5)
    tuples = ((0, 5), (0, -5), (5, 0), (-5, 0))
    for i, (u, v) in enumerate(tuples):
        score = corner_score(img, u, v, W)
        save_img(score, f"./feature_detection/corner_score_{i}.png")
    u, v, W = None, None, None

    # score = corner_score(img, u, v, W)
    # save_img(score, "./feature_detection/corner_score.png")
    #
    # # Computing the corner scores for various u, v values.
    # score = corner_score(img, 0, 5, W)
    # save_img(score, "./feature_detection/corner_score05.png")
    #
    # score = corner_score(img, 0, -5, W)
    # save_img(score, "./feature_detection/corner_score0-5.png")
    #
    # score = corner_score(img, 5, 0, W)
    # save_img(score, "./feature_detection/corner_score50.png")
    #
    # score = corner_score(img, -5, 0, W)
    # save_img(score, "./feature_detection/corner_score-50.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
