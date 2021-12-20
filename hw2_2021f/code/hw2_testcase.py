import cv2
import numpy as np
import time
import os
from my_filter2D import my_filter2D


def hw2_testcase():
    # This script has test cases to help you test your my_filter2D() function. You should verify here that your
    # output is reasonable before using your my_filter2D to construct a hybrid image in hw2.py. The outputs are all
    # saved and you can include them in your writeup. You can add calls to cv2.filter2D() if you want to check that
    # my_filter2D() is doing something similar.
    #
    # Revised by Dahyun Kang and originally written by James Hays.

    ## Setup
    test_image = cv2.imread('../data/cat.bmp') / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )

    result_dir = '../result/test'
    os.makedirs(result_dir, exist_ok=True)

    cv2.imshow('test_image', test_image)
    cv2.waitKey(10)

    ##################################
    ## Identify filter
    # This filter should do nothing regardless of the padding method you use.
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    identity_image = my_filter2D(test_image, identity_filter)

    cv2.imshow('identity_image', identity_image)
    cv2.imwrite(os.path.join(result_dir, 'identity_image.jpg'), identity_image * 255)

    ##################################
    ## Small blur with a box filter
    # This filter should remove some high frequencies
    blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    blur_filter = blur_filter / sum(sum(blur_filter))  # making the filter sum to 1

    blur_image = my_filter2D(test_image, blur_filter)

    cv2.imshow('blur_image', blur_image)
    cv2.imwrite(os.path.join(result_dir, 'blur_image.jpg'), blur_image * 255)

    ##################################
    ## Large blur
    # This blur would be slow to do directly, so we instead use the fact that Gaussian blurs are separable and blur
    # sequentially in each direction.
    large_1d_blur_filter = cv2.getGaussianKernel(25, 10)

    start_time = time.time()
    large_blur_image = my_filter2D(test_image, large_1d_blur_filter)
    large_blur_image = my_filter2D(large_blur_image, large_1d_blur_filter.T)  # notice the transpose operator
    print(f'[large_blur_image] time spent: {time.time() - start_time:.4} sec')

    cv2.imshow('large_blur_image', large_blur_image)
    cv2.imwrite(os.path.join(result_dir, 'large_blur_image.jpg'), large_blur_image * 255)

    # # If you want to see how slow this would be to do naively, try out this equivalent operation:
    # large_blur_filter_naive = large_1d_blur_filter * large_1d_blur_filter.T
    #
    # start_time = time.time()
    # large_blur_image_naive = my_filter2D(test_image, large_blur_filter_naive)
    # print(f'[large_blur_image_naive] time spent: {time.time() - start_time:.4} sec')
    #
    # cv2.imshow('large_blur_image_naive', large_blur_image_naive)
    # cv2.imwrite(os.path.join(result_dir, 'large_blur_image_naive.jpg'), large_blur_image_naive * 255)

    ##################################
    ## Oriented filter (Sobel Operator)
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # should respond to horizontal gradients

    sobel_image = my_filter2D(test_image, sobel_filter)

    # 0.5 added because the output image is centered around zero otherwise and mostly black
    cv2.imshow('sobel_image + 0.5', sobel_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'sobel_image.jpg'), (sobel_image + 0.5) * 255)

    ##################################
    ## High pass filter (Discrete Laplacian)
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    laplacian_image = my_filter2D(test_image, laplacian_filter)

    # 0.5 added because the output image is centered around zero otherwise and mostly black
    cv2.imshow('laplacian_image + 0.5', laplacian_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'laplacian_image.jpg'), (laplacian_image + 0.5) * 255)

    ##################################
    ## High pass "filter" alternative
    high_pass_image = test_image - blur_image  # simply subtract the low frequency content

    cv2.imshow('high_pass_image + 0.5', high_pass_image + 0.5)
    cv2.imwrite(os.path.join(result_dir, 'high_pass_image.jpg'), (high_pass_image + 0.5) * 255)

    ##################################
    ## Done
    print('Press any key ...')
    cv2.waitKey(0)


if __name__ == '__main__':
    hw2_testcase()
