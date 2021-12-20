# Python >= 3.7, OpenCV >= 4.5.0, Numpy >= 1.17.0
import os
import time
import cv2
import numpy as np
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

from bayer_to_rgb import bayer_to_rgb
from calculate_fundamental_matrix import calculate_fundamental_matrix
from rectify_stereo_images import rectify_stereo_images
from calculate_disparity_map import calculate_disparity_map


def hw3():
    result_dir = '../result'
    os.makedirs(result_dir, exist_ok=True)

    #=======================================================================================
    # Read bayer pattern image
    data_dir = '../data/scene0'
    img1_path = f'{data_dir}/bayer_cam1.png'
    img2_path = f'{data_dir}/bayer_cam2.png'
    bayer_img1 = cv2.imread(img1_path, -1)
    bayer_img2 = cv2.imread(img2_path, -1)

    #=======================================================================================
    # HW3-a
    img1 = bayer_to_rgb(bayer_img1, 'bilinear')
    img2 = bayer_to_rgb(bayer_img2, 'bilinear')

    # If you implement bicubic interpolation, you can use bicubic option as follows.
    # img1 = bayer_to_rgb(bayer_img1, 'bicubic')
    # img2 = bayer_to_rgb(bayer_img2, 'bicubic')

    #=======================================================================================
    # Read feature point
    points = np.loadtxt(f'{data_dir}/feature_points.txt', dtype=np.float32, delimiter=',')
    pts1 = points[:,:2]
    pts2 = points[:,2:]

    #=======================================================================================
    # Visualize image and feature points
    fig, ax = plt.subplots(1,2, figsize=(16,5))
    ax[0].imshow(img1)
    ax[0].scatter(pts1[:,0], pts1[:,1], c='#00ff00', s=80, marker='+')
    ax[0].set_title('Left image')
    ax[1].imshow(img2)
    ax[1].scatter(pts2[:,0], pts2[:,1], c='#00ff00', s=80, marker='+')
    ax[1].set_title('Right image')
    fig.tight_layout()
    plt.show(block=False)

    # Save image
    cv2.imwrite(f'{result_dir}/img_1.png', img1[:,:,::-1])
    cv2.imwrite(f'{result_dir}/img_2.png', img2[:,:,::-1])

    #=======================================================================================
    # HW3-b
    fundamental_matrix = calculate_fundamental_matrix(pts1, pts2)

    #=======================================================================================
    # Compute homography matrix for rectification.
    # As described in website, computing this matrix requires much more steps
    # than what we covered in the lecture, you don't need to work on this part.
    _, h1, h2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundamental_matrix, (img1.shape[1], img1.shape[0]))

    #=======================================================================================
    # HW3-c
    img1_rectified, img2_rectified = rectify_stereo_images(img1, img2, h1, h2)

    #=======================================================================================
    # Visualize rectified image
    fig, ax = plt.subplots(1,2, figsize=(16,5))
    ax[0].imshow(img1_rectified)
    ax[0].set_title('Left image rectified')
    ax[1].imshow(img2_rectified)
    ax[1].set_title('Right image rectified')
    fig.tight_layout()
    plt.show(block=False)

    # Save rectified image
    cv2.imwrite(f'{result_dir}/rectified_img_1.png', img1_rectified[:,:,::-1])
    cv2.imwrite(f'{result_dir}/rectified_img_2.png', img2_rectified[:,:,::-1])

    #=======================================================================================
    # HW3-d
    # You may change the window_size and max_disparity
    # WARNING: This may take some time depending your implementation.
    # Your code must be done in "180 seconds". Otherwise you will not get score.
    window_size = 7
    max_disparity = 40

    img1_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_RGB2GRAY)

    # Do not touch time measurement here. -------------------------------
    tic = time.time()
    # -------------------------------------------------------------------

    disparity_map = calculate_disparity_map(img1_gray, img2_gray, window_size, max_disparity)

    # Do not touch time measurement here. -------------------------------
    ex_time = time.time()-tic
    pf = (lambda x: 'PASS' if x<180.0 else 'FAIL')(ex_time)
    print(f'[ {pf} ] Disparity computation time: {ex_time:.2f} seconds')
    # -------------------------------------------------------------------

    #=======================================================================================
    # Visualize disparity map
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    img = ax.imshow(disparity_map, cmap='inferno')
    ax.set_title('Disparity map')
    fig.colorbar(img, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(f'{result_dir}/disparity_map', bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    hw3()


