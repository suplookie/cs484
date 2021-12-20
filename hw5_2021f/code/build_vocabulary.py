import cv2
import numpy as np

from feature_extraction import feature_extraction


def build_vocabulary(image_paths, vocab_size, feature):
    """
    This function will sample feature descriptors from the training images,
    cluster them with kmeans, and the return the cluster centers.

    :param image_paths: a N array of string where each string is an image path
    :param vocab_size: the size of the vocabulary.
    :param feature: name of image feature representation.

    :return: a vocab_size x feature_size matrix. center positions of k-means clustering.
    """
    all_features = []

    print('Extracting features ...')
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]

        features = feature_extraction(img, feature)
        all_features.append(features)

    all_features = np.concatenate(all_features, 0)

    # k-means clustering
    print('K-means clustering ...')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(all_features, vocab_size, None, criteria, 10, flags)

    return centers
