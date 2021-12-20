import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.

    all_hists = []
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]
        img_hist = []
        w, h, _ = img.shape
        for i in range(max_level+1):
            for j in range(4**i):
                pyramid_img = img[w*(j//2**i)//2**i:w*(j//2**i+1)//2**i, h*(j%2**i)//2**i:h*(j%2**i+1)//2**i, :]
                features = feature_extraction(pyramid_img, feature)
                if features.shape == ():
                    hist = np.zeros(200)
                else:
                    hist, _ = np.histogram(np.argmin(pdist(features, vocab), axis=1), bins=[b for b in range(vocab.shape[0]+1)])
                    hist = (hist - hist.mean()) / hist.std()
                img_hist.append(hist) 
        img_hist = np.concatenate(img_hist, 0)
        all_hists.append(img_hist)

    return np.array(all_hists)
