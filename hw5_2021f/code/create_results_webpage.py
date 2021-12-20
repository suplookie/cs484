import cv2
import os

import numpy as np
from matplotlib import pyplot as plt


def create_results_webpage(train_image_paths,
                           test_image_paths,
                           train_labels,
                           test_labels,
                           categories,
                           abbr_categories,
                           predicted_categories):
    print('Creating results_webpage/index.html, thumbnails, and confusion matrix')

    # number of examples of training examples, true positives, false positives,
    # and false negatives. This the table will be num_samples * 4 images wide
    # (unless there aren't enough images)
    num_samples = 2
    thumbnail_height = 75  # pixels

    os.makedirs('results_webpage', exist_ok=True)
    os.makedirs('results_webpage/thumbnails', exist_ok=True)
    if os.path.exists('results_webpage/thumbnails/*.jpg'):
        os.remove('results_webpage/thumbnails/*.jpg')

    # Generate confusion matrix
    num_categories = len(categories)

    confusion_matrix = np.zeros((num_categories, num_categories))

    for i in range(predicted_categories.size):
        row = categories.index(test_labels[i])
        column = categories.index(predicted_categories[i])
        confusion_matrix[row, column] += 1

    num_test_per_cat = test_labels.size / num_categories
    confusion_matrix = confusion_matrix / num_test_per_cat
    accuracy = np.mean(np.diag(confusion_matrix))
    print('Accuracy (mean of diagonal of confusion matrix) is %.3f' % accuracy)

    # Visualization of confusion matrix
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.xticks(range(15), abbr_categories)
    plt.yticks(range(15), categories)

    plt.savefig('results_webpage/confusion_matrix.png', bbox_inches='tight')

    # Result table rows
    row_htmls = []
    for i in range(num_categories):
        category = categories[i]
        cat_accuracy = confusion_matrix[i, i]

        train_exmples = train_image_paths[test_labels == category]
        true_positives = test_image_paths[(test_labels == category) &
                                          (predicted_categories == category)]
        false_positive_inds = (test_labels != category) & (predicted_categories == category)
        false_positives = test_image_paths[false_positive_inds]
        false_positive_labels = test_labels[false_positive_inds]

        false_negative_inds = (test_labels == category) & (predicted_categories != category)
        false_negatives = test_image_paths[false_negative_inds]
        false_negative_labels = predicted_categories[false_negative_inds]

        # Shuffle samples
        np.random.shuffle(train_exmples)
        np.random.shuffle(true_positives)
        false_positive_shuffle = np.random.permutation(false_positives.size)
        false_positives = false_positives[false_positive_shuffle]
        false_positive_labels = false_positive_labels[false_positive_shuffle]
        false_nagative_shuffle = np.random.permutation(false_negatives.size)
        false_negatives = false_negatives[false_nagative_shuffle]
        false_negative_labels = false_negative_labels[false_nagative_shuffle]

        train_exmples = train_exmples[:np.min([train_exmples.size, num_samples])]
        true_positives = true_positives[:np.min([true_positives.size, num_samples])]
        false_positives = false_positives[:np.min([false_positives.size, num_samples])]
        false_positive_labels = false_positive_labels[:np.min([false_positive_labels.size, num_samples])]
        false_negatives = false_negatives[:np.min([false_negatives.size, num_samples])]
        false_negative_labels = false_negative_labels[:np.min([false_negative_labels.size, num_samples])]

        # Write html rows
        def write_samples(classname, samples, labels=None):
            row_sample_html = ''
            for j in range(num_samples):
                if j < samples.size:
                    im = cv2.imread(samples[j])
                    height = im.shape[0]
                    rescale_factor = thumbnail_height / height
                    tmp = cv2.resize(im, None, None, rescale_factor, rescale_factor)
                    height = tmp.shape[0]
                    width = tmp.shape[1]

                    name = os.path.splitext(os.path.basename(samples[j]))[0]
                    thumbnail_path = 'thumbnails/%s_%s.jpg' % (category, name)
                    cv2.imwrite('results_webpage/%s' % thumbnail_path, tmp)
                    row_sample_html += '<td class="%s"><img src="%s" width=%d height=%d>' \
                                       % (classname, thumbnail_path, width, height)

                    if labels is not None:
                        row_sample_html += '<br><small>%s</small>' % labels[j]

                    row_sample_html += '</td>\n'
                else:
                    row_sample_html += '<td class="cell %s"></td>\n' % classname

            return row_sample_html

        row_html = '<tr class="category-row">\n'
        row_html += '<td>%s</td>\n' % category
        row_html += '<td><div>%.3f</div><div class="accuracy-bar" style="width:64px;">' \
                    '<div style="width:%.2fpx;"/></div></td>\n' \
                    % (cat_accuracy, cat_accuracy * 64)
        row_html += write_samples('training', train_exmples)
        row_html += write_samples('true-positive', true_positives)
        row_html += write_samples('false-positive', false_positives, false_positive_labels)
        row_html += write_samples('false-negative', false_negatives, false_negative_labels)
        row_html += '</tr>\n'

        row_htmls.append(row_html)

    # Write result html file
    result_html = '\n'.join(row_htmls)

    with open('template.html', 'r') as template_file:
        template_html = template_file.read()

    template_html = template_html.replace('%accuracy%', '%.3f' % accuracy)
    template_html = template_html.replace('%results%', result_html)

    with open('results_webpage/index.html', 'w') as result_file:
        result_file.write(template_html)
