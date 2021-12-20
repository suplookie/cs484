
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from get_features_from_pca import get_features_from_pca

def pca_visualize(pca_out_dim, feature, vocab_size):

    """
	This function assumes that 'vocab_*.npy' exists.
	This function visualizes the vocab after dimension reduction by
	'get_features_from_pca' function 2D or 3D specified by parameter 'pca_out_dim'.

    :param pca_out_dim: output dimension which will be used in PCA
	:param feature: feature extraction method 'HoG' or 'SIFT'
	:param vocab_size: number of center points which is used in k-means clustering

    :return: visualizes the vocab in 2D or 3D
    """

    reduced_vocab = get_features_from_pca(pca_out_dim, feature)

    if pca_out_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(vocab_size):
            ax.scatter(reduced_vocab[i, 0], reduced_vocab[i, 1], reduced_vocab[i, 2])
        plt.pause(0.1)

    else:

        reduced_vocab = get_features_from_pca(pca_out_dim, feature)
        plt.figure()
        for i in range(vocab_size):
            plt.scatter(reduced_vocab[i,0],reduced_vocab[i,1])
        plt.pause(0.1)

    return
