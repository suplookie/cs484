from matplotlib.pyplot import axis
import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab_*.npy' exists and contains an N x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    all_hists = []

    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]

        features = feature_extraction(img, feature)
        hist, _ = np.histogram(np.argmin(pdist(features, vocab), axis=1), bins=[i for i in range(vocab.shape[0]+1)])
        hist = (hist - hist.mean()) / hist.std()
        all_hists.append(hist)

    return np.array(all_hists)
