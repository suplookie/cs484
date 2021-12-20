import cv2
import numpy as np


def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    alpha = 0.04
    threshold = 0.006

    edge_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edge_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    im_grad_x = cv2.filter2D(image, cv2.CV_32F, edge_x)
    im_grad_y = cv2.filter2D(image, cv2.CV_32F, edge_y)

    im_grad_xx = im_grad_x ** 2
    im_grad_yy = im_grad_y ** 2
    im_grad_xy = (im_grad_x * im_grad_y)

    det_M = im_grad_xx * im_grad_yy - im_grad_xy ** 2
    trace_M = im_grad_xx + im_grad_yy

    C = np.abs(det_M - alpha * trace_M ** 2)

    local_max_x = set()
    from scipy.signal import argrelextrema

    for x in range(C.shape[0]):
        for e in argrelextrema(C[x], np.greater)[0]:
            local_max_x.add((x, e))

    local_max_y = set()
    for x in range(C.T.shape[0]):
        for e in argrelextrema(C.T[x], np.greater)[0]:
            local_max_y.add((e, x))
    C_under_threshold = set([tuple(x) for x in np.argwhere(C>threshold)])
    inter = np.array([x for x in local_max_x & local_max_y & C_under_threshold], dtype=np.float32)
    return (inter.T)[1], (inter.T)[0]

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000



def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)
    padded_image = np.pad(image, (descriptor_window_image_width//2+1, descriptor_window_image_width//2+1), mode='reflect')
    padded_image = cv2.GaussianBlur(padded_image, (3, 3), 1)
    difference_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    difference_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    features = np.zeros((x.shape[0], 128))
    for index in range(x.shape[0]):
        j = int(x[index] + descriptor_window_image_width//2+1)
        i = int(y[index] + descriptor_window_image_width//2+1)
        raw_window = padded_image[i-descriptor_window_image_width//2 - 1:i+descriptor_window_image_width//2+1, j-descriptor_window_image_width//2-1:j+descriptor_window_image_width//2+1]
        window_difference_x = cv2.filter2D(raw_window, cv2.CV_32F, difference_x)[1:-1, 1:-1]
        window_difference_y = cv2.filter2D(raw_window, cv2.CV_32F, difference_y)[1:-1, 1:-1]
        magnitude = np.sqrt(window_difference_x ** 2 + window_difference_y ** 2)
        theta = np.arctan2(window_difference_x, window_difference_y) * 180 / np.pi
        magnitude = magnitude * (cv2.getGaussianKernel(descriptor_window_image_width, descriptor_window_image_width / 2) @ cv2.getGaussianKernel(descriptor_window_image_width, descriptor_window_image_width / 2).T)
        global_histogram = np.zeros(36)
        for a in range(descriptor_window_image_width):
            for b in range(descriptor_window_image_width):
                global_histogram[int(theta[a, b] // 10)] += magnitude[a, b]
        max_arg = np.argmax(global_histogram)
        theta -= max_arg * 10

        histogram = np.zeros((4, 4, 8))
        for a in range(descriptor_window_image_width):
            for b in range(descriptor_window_image_width):
                angle = int(theta[a, b])
                if angle < 0:
                    angle += 360
                histogram[a%4, b%4, angle // 45] += magnitude[a, b]

        reshaped_histogram = histogram.reshape(128)
        reshaped_histogram /= max(1e-6, np.linalg.norm(reshaped_histogram))
        reshaped_histogram[reshaped_histogram > 0.2] = 0.2
        reshaped_histogram /= max(1e-6, np.linalg.norm(reshaped_histogram))
        features[index, :] += reshaped_histogram

    return features

def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.
    # For extra credit you can implement spatial verification of matches.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    threshold = 0.75

    distance = 1 - np.dot(features1, features2.T)
    sorted_distance = np.sort(distance, axis=1)
    sorted_distance_index = np.argsort(distance, axis=1)
    NN1 = sorted_distance[:, 0]
    NN2 = sorted_distance[:, 1]
    ratio = NN1 / NN2
    matches = np.empty((0, 2))
    confidences = np.empty(0)

    for i in range(ratio.shape[0]):
        assert(ratio.all() <= 1)
        if ratio[i] < threshold:
            confidences = np.append(confidences, [1/ratio[i]])
            matches = np.append(matches, [[i, sorted_distance_index[i, 0]]], axis=0)
    return matches, confidences

