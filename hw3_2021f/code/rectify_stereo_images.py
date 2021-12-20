import numpy as np
import cv2

# HW3-c
# Given two homography matrices for two images, generate the rectified image pair.
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # Hint: Care about alignment of image.
    # In order to superpose two rectified images, you need to create crtain amount of margin.
    # Which means you need to do additional things to get fully warped image (not cropped).
    ################################################
    src1 = np.array([[0, 0], [0, img1.shape[1]], [img1.shape[0], img1.shape[1]], [img1.shape[0], 0]], dtype=np.float32).reshape(-1, 1, 2)
    src2 = np.array([[0, 0], [0, img2.shape[1]], [img2.shape[0], img2.shape[1]], [img2.shape[0], 0]], dtype=np.float32).reshape(-1, 1, 2)
    dst1 = cv2.perspectiveTransform(src1, h1)
    dst2 = cv2.perspectiveTransform(src2, h2)
    dsts = np.concatenate((dst1, dst2), axis=0)
    mins = np.min(dsts, axis=0)
    maxs = np.max(dsts, axis=0)

    Ht_1 = np.array([[1, 0, -mins[0][0]], [0, 1, -mins[0][1]+20], [0, 0, 1]])
    img1_rectified = cv2.warpPerspective(img1, h1 @ Ht_1, (int(maxs[0][1] - mins[0][1] + 50), int(maxs[0][0] - mins[0][0] + 90)))
    Ht_2 = np.array([[1, 0, -mins[0][0]], [0, 1, -mins[0][1]+20], [0, 0, 1]])
    img2_rectified = cv2.warpPerspective(img2, h2 @ Ht_2, (int(maxs[0][1] - mins[0][1] + 50), int(maxs[0][0] - mins[0][0] + 90)))

    ################################################

    return img1_rectified, img2_rectified