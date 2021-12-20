import os
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import time
from hw4_functions import *
from util_functions import *


# Original MATLAB codes are written by James Hays and James Tompkin for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
# Revised Python codes are written by Inseung Hwang at KAIST.
#
#
# This script
# - Loads and resizes images
# - Computes correspondence
# - Visualizes the matches
# - Evaluates the matches based on ground truth correspondences


# Visualize the results
# We will evaluate your code with this set to None (no vis).
# You can select visualization mode between 'dots' and 'arrows'
visualize = 'dots'

# Amount by which to resize images for speed.
# Feel free to experiment with this for debugging,
# but we will evaluate your code with this set to 0.5.
scale_factor = 0.5

# Width and height of the descriptor window around each local feature, in image pixels.
# In SIFT, this is 16 pixels.
# Feel free to experiment with this for debugging or extra credit,
# but we will evaluate your code with this set to 16.
descriptor_window_image_width = 16

# Number of points to evaluate for accuracy
# We will evaluate your code on the first 100 matches you return.
maxPtsEval = 100

# Whether to use the 'cheat' hand-picked interest points
cheatInterestPoints = False

def main():
    # Notre Dame de Paris
    # Easiest
    print('Notre Dame de Paris')
    image1 = cv2.imread('../data/NotreDame/921919841_a30df938f2_o.jpg')
    image2 = cv2.imread('../data/NotreDame/4191453057_c86028ce1f_o.jpg')
    image1 = cv2.normalize(image1.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image2 = cv2.normalize(image2.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    eval_file = '../data/NotreDame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat'
    reference_results = scipy.io.loadmat(eval_file)

    image1 = cv2.resize(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    image2 = cv2.resize(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    time_start = time.time()
    # Task: implement the following three fuctions
    # 1) Find distinctive interest points in each image. Szeliski 4.1.1
    if not cheatInterestPoints:
        x1, y1 = get_interest_points(image1, descriptor_window_image_width)
        x2, y2 = get_interest_points(image2, descriptor_window_image_width)
    else:
        # Use cheat_interest_points only for development and debugging!
        x1, y1, x2, y2 = cheat_interest_points(reference_results, scale_factor, image1, image2,
                                                 descriptor_window_image_width)


    # 2) Create feature descriptors at each interest point.Szeliski 4.1.2
    image1_features = get_descriptors(image1, x1, y1, descriptor_window_image_width)
    image2_features = get_descriptors(image2, x2, y2, descriptor_window_image_width)

    # 3) Match features.Szeliski 4.1.3
    matches, confidences = match_features(image1_features, image2_features)

    time_end = time.time()

    # Evaluate matches
    # matches = np.array([np.arange(0,100),np.arange(0,100)]).transpose()
    # confidences = np.arange(1,0,-0.01)
    evaluation_result = evaluate_correspondence(image1, image2, reference_results,
                                                scale_factor, x1, y1, x2, y2,
                                                matches, confidences,
                                                maxPtsEval, visualize, 'eval_ND.png')
    numGoodMatches, numBadMatches, accuracyAll, accuracyMaxEval = evaluation_result

    print("Elpased time: %.2fs"%(time_end-time_start))

    # Mount Rushmore
    # A little harder than Notre Dame
    print('Mount Rushmore')
    image1 = cv2.imread('../data/MountRushmore/9021235130_7c2acd9554_o.jpg')
    image2 = cv2.imread('../data/MountRushmore/9318872612_a255c874fb_o.jpg')
    image1 = cv2.normalize(image1.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image2 = cv2.normalize(image2.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    eval_file = '../data/MountRushmore/9021235130_7c2acd9554_o_to_9318872612_a255c874fb_o.mat'
    reference_results = scipy.io.loadmat(eval_file)

    image1 = cv2.resize(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    image2 = cv2.resize(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    time_start = time.time()
    # Task: implement the following three fuctions
    # 1) Find distinctive interest points in each image. Szeliski 4.1.1
    if not cheatInterestPoints:
        x1, y1 = get_interest_points(image1, descriptor_window_image_width)
        x2, y2 = get_interest_points(image2, descriptor_window_image_width)
    else:
        # Use cheat_interest_points only for development and debugging!
        x1, y1, x2, y2 = cheat_interest_points(reference_results, scale_factor, image1, image2,
                                                 descriptor_window_image_width)


    # 2) Create feature descriptors at each interest point.Szeliski 4.1.2
    image1_features = get_descriptors(image1, x1, y1, descriptor_window_image_width)
    image2_features = get_descriptors(image2, x2, y2, descriptor_window_image_width)

    # 3) Match features.Szeliski 4.1.3
    matches, confidences = match_features(image1_features, image2_features)

    time_end = time.time()

    # Evaluate matches
    # matches = np.array([np.arange(0,100),np.arange(0,100)]).transpose()
    # confidences = np.arange(1,0,-0.01)
    evaluation_result = evaluate_correspondence(image1, image2, reference_results,
                                                scale_factor, x1, y1, x2, y2,
                                                matches, confidences,
                                                maxPtsEval, visualize, 'eval_MR.png')
    numGoodMatches, numBadMatches, accuracyAll, accuracyMaxEval = evaluation_result

    print("Elpased time: %.2fs"%(time_end-time_start))

    # Gaudi's Episcopal Palace
    # This pair is difficult
    print('Gaudi\'s Episcopal Palace')
    image1 = cv2.imread('../data/EpiscopalGaudi/4386465943_8cf9776378_o.jpg')
    image2 = cv2.imread('../data/EpiscopalGaudi/3743214471_1b5bbfda98_o.jpg')
    image1 = cv2.normalize(image1.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image2 = cv2.normalize(image2.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    eval_file = '../data/EpiscopalGaudi/4386465943_8cf9776378_o_to_3743214471_1b5bbfda98_o.mat'
    reference_results = scipy.io.loadmat(eval_file)

    image1 = cv2.resize(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    image2 = cv2.resize(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    time_start = time.time()
    # Task: implement the following three fuctions
    # 1) Find distinctive interest points in each image. Szeliski 4.1.1
    if not cheatInterestPoints:
        x1, y1 = get_interest_points(image1, descriptor_window_image_width)
        x2, y2 = get_interest_points(image2, descriptor_window_image_width)
    else:
        # Use cheat_interest_points only for development and debugging!
        x1, y1, x2, y2 = cheat_interest_points(reference_results, scale_factor, image1, image2,
                                                 descriptor_window_image_width)


    # 2) Create feature descriptors at each interest point.Szeliski 4.1.2
    image1_features = get_descriptors(image1, x1, y1, descriptor_window_image_width)
    image2_features = get_descriptors(image2, x2, y2, descriptor_window_image_width)

    # 3) Match features.Szeliski 4.1.3
    matches, confidences = match_features(image1_features, image2_features)

    time_end = time.time()

    # Evaluate matches
    # matches = np.array([np.arange(0,100),np.arange(0,100)]).transpose()
    # confidences = np.arange(1,0,-0.01)
    evaluation_result = evaluate_correspondence(image1, image2, reference_results,
                                                scale_factor, x1, y1, x2, y2,
                                                matches, confidences,
                                                maxPtsEval, visualize, 'eval_EG.png')
    numGoodMatches, numBadMatches, accuracyAll, accuracyMaxEval = evaluation_result

    print("Elpased time: %.2fs"%(time_end-time_start))

    plt.show()

if __name__=='__main__':
    main()

