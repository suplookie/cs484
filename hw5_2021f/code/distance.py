import numpy as np


def pdist(a, b):
    a_square = np.einsum('ij,ij->i', a, a)
    a_square = np.tile(np.reshape(a_square, [a.shape[0], 1]), [1, b.shape[0]])
    b_square = np.einsum('ij,ij->i', b, b)
    b_square = np.tile(np.reshape(b_square, [1, b.shape[0]]), [a.shape[0], 1])
    ab = np.dot(a, b.T)

    dist = a_square + b_square - 2 * ab
    dist = dist.clip(min=0)
    return np.sqrt(dist)
