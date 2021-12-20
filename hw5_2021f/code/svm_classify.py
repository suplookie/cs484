from matplotlib.pyplot import axis
import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    # Your code here. You should also change the return value.
    score = []
    for label in categories:
        model = svm.LinearSVC(C=1, max_iter=100000)
        model.fit(train_image_feats, train_labels==label)
        score.append(model.decision_function(test_image_feats))
    score = np.array(score)
    classified_label = np.argmax(np.array(score), axis=0)



    return np.array([categories[i] for i in classified_label])