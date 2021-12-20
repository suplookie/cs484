import numpy as np

def get_features_from_pca(feat_num, feature):

    """
    This function loads 'vocab_*.npy' file and
	returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
	:param feature: 'HoG' or 'SIFT'

    :return: an N x feat_num matrix
    """

    vocab = np.load(f'vocab_{feature}.npy')

    # Your code here. You should also change the return value.

    vocab_normalized = (vocab - vocab.mean(axis=0)) / vocab.std(axis=0)
    cov_vocab = vocab_normalized.T @ vocab_normalized
    _, eig_vec = np.linalg.eig(cov_vocab)

    return vocab_normalized @ (eig_vec[:, :feat_num] @ eig_vec[:, :feat_num].T)


