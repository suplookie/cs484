from glob import glob

import numpy as np


def get_image_paths(data_path, categories, num_train_per_cat):
    """
    This function returns arrays containing the file path for each train
    and test image, as well as cell arrays with the label of each train and
    test image. By default all four of these arrays will be 1500x1 where each
    entry is a char array (or string).

    :param data_path: a path where containing dataset.
    :param categories: a N string array of categories.
    :param num_train_per_cat: a number of training images for each category.

    :return: [train_image_paths, test_image_paths, train_labels, test_labels]
    """
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []

    for i in range(num_categories):
        images = glob('%s/train/%s/*.jpg' % (data_path, categories[i]))
        for j in range(num_train_per_cat):
            train_image_paths.append(images[j])
            train_labels.append(categories[i])

        images = glob('%s/test/%s/*.jpg' % (data_path, categories[i]))
        for j in range(num_train_per_cat):
            test_image_paths.append(images[j])
            test_labels.append(categories[i])

    return np.array(train_image_paths), \
           np.array(test_image_paths), \
           np.array(train_labels), \
           np.array(test_labels)
