import numpy as np
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


def cheat_interest_points(reference_results, scale_factor, image1, image2, feature_width):
    # Local Feature Stencil Code
    # Original MATLAB codes are written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # This function is provided for development and debugging but cannot be
    # used in the final handin. It 'cheats' by generating interest points from
    # known correspondences. It will only work for the three image pairs with
    # known correspondences.

    # NOTE: These feature points are _subpixel_ precise.
    # By default, we round these interest points, but in principle, you can
    # interpolate the image to extract descriptors at subpixel locations.

    # 'eval_file' is the file path to the list of known correspondences.
    # 'scale_factor' is needed to map from the original image coordinates to
    #   the resolution being used for the current experiment.

    # 'x1' and 'y1' are nx1 vectors of x and y coordinates of interest points
    #   in the first image.
    # 'x1' and 'y1' are mx1 vectors of x and y coordinates of interest points
    #   in the second image. For convenience, n will equal m but don't expect
    #   that to be the case when interest points are created independently per
    #   image.

    x1 = reference_results['x1']-1
    y1 = reference_results['y1']-1
    x2 = reference_results['x2']-1
    y2 = reference_results['y2']-1

    x1 = np.round( x1 * scale_factor )
    y1 = np.round( y1 * scale_factor )
    x2 = np.round( x2 * scale_factor )
    y2 = np.round( y2 * scale_factor )

    # Check bounds
    m1,n1 = image1.shape
    m2,n2 = image2.shape
    fw2 = feature_width/2

    ind1 = (x1 - fw2 < 0) | (x1 + fw2 >= m1) | (y1 - fw2 <0) | (y1 + fw2 >= n1)
    x1 = x1[~ind1]
    y1 = y1[~ind1]
    x2 = x2[~ind1]
    y2 = y2[~ind1]

    ind2 = (x2 - fw2 < 0) | (x2 + fw2 >= m1) | (y2 - fw2 <0) | (y2 + fw2 >= n1)
    x1 = x1[~ind2]
    y1 = y1[~ind2]
    x2 = x2[~ind2]
    y2 = y2[~ind2]

    return x1,y1,x2,y2

def show_correspondence(imgA, imgB, x1, y1, x2, y2, vismode, visfilename=None, good_matches=np.empty(0)):

    # Image Correspondence Visualization
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Edits by James Tompkin
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Visualizes corresponding points between two images, either as
    # arrows or dots.
    # mode='dots': Corresponding points will have the same random color
    # mode='arrows': Corresponding points will be joined by a line
    #
    # Writes out a png of the visualization if 'filename' is not empty ([]).
    #
    # Labels dots or arrows as correct or incorrect with green/red if 'goodMatch' is not empty ([]).

    H = max(imgA.shape[0], imgB.shape[0])
    W = imgA.shape[1]+ imgB.shape[1]
    newImg = np.zeros((H, W))
    newImg[:imgA.shape[0],:imgA.shape[1]] = imgA
    newImg[:imgB.shape[0],imgA.shape[1]:] = imgB
    plt.figure()
    plt.imshow(newImg,'gray')
    plt.axis('off')

    shiftX = imgA.shape[1]

    for i in range(x1.shape[0]):
        cur_color = np.random.rand(3)
        edgeColor = np.array([0, 0, 0])
        if good_matches.size > 0:
            if good_matches[i] == 0:
                edgeColor = [1, 0, 0]
            else:
                edgeColor = [0, 1, 0]

        if vismode == 'dots':
            plt.scatter(x1[i], y1[i], c=[cur_color], s=25, linewidths=1, edgecolors=edgeColor)
            plt.scatter(x2[i]+shiftX, y2[i], c=[cur_color], s=25, linewidths=1, edgecolors=edgeColor)
        elif vismode == 'arrows':
            plt.plot([x1[i], shiftX + x2[i]], [y1[i],y2[i]], 'o-', c=cur_color, markeredgecolor=edgeColor,linewidth=2, markersize=3)

    plt.show(block=False)
    plt.pause(0.001)
    if visfilename is not None:
        print('Saving visualization: %s' % (visfilename))
        plt.savefig(visfilename)

def evaluate_correspondence(imgA, imgB, ground_truth_correspondence, scale_factor, x1i, y1i, x2i, y2i, matches, confidences, maxPtsToEval, vismode, visfilename):

    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
    # Edited by James Tompkin
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Your feature points should be unique within a local region,
    # I.E., your detection non-maximal suppression function should work.
    #
    # Look at the 'uniqueness test' for how we enforce this.
    # It is intentionally simplistic and biased,
    # so make your detector _do the right thing_.

    ## Sort matches by confidence
    # Sort the matches so that the most confident onces are at the top of the
    # list. You should not delete this, so that the evaluation
    # functions can be run on the top matches easily.
    ind = np.argsort(-confidences)
    matches = matches[ind,:]

    x1_est = x1i[matches[:,0].astype('int').reshape((-1))]
    y1_est = y1i[matches[:,0].astype('int').reshape((-1))]
    x2_est = x2i[matches[:,1].astype('int').reshape((-1))]
    y2_est = y2i[matches[:,1].astype('int').reshape((-1))]

    x1_est = x1_est / scale_factor
    y1_est = y1_est / scale_factor
    x2_est = x2_est / scale_factor
    y2_est = y2_est / scale_factor

    good_matches = np.zeros((x1_est.size,1)) #indicator vector

    x1  = ground_truth_correspondence['x1']
    y1  = ground_truth_correspondence['y1']
    x2  = ground_truth_correspondence['x2']
    y2  = ground_truth_correspondence['y2']

    ##########################################################################
    # Uniqueness test
    #
    x1_est_tmp = x1_est
    y1_est_tmp = y1_est
    x2_est_tmp = x2_est
    y2_est_tmp = y2_est
    uniquenessDist = 5

    # For each ground truth point
    numPreMerge = x1_est.size
    for i in range(x1.size):
        # Compute distance of each estimated point to
        # the ground truth point
        x_dists = x1[i] - x1_est_tmp
        y_dists = y1[i] - y1_est_tmp
        dists = np.sqrt( x_dists**2 + y_dists**2 )
        toMerge = dists < uniquenessDist

        if np.any(toMerge):
            # Do something to remove duplicates. Let's
            # average the coordinates of all points
            # within 'uniquenessDist' pixels.
            # Also average the corresponded point (!)
            #
            # This part is simplistic, but a real-world
            # computer vision system would not know
            # which correspondences were good.
            avgX1 = np.mean( x1_est_tmp[toMerge] )
            avgY1 = np.mean( y1_est_tmp[toMerge] )
            avgX2 = np.mean( x2_est_tmp[toMerge] )
            avgY2 = np.mean( y2_est_tmp[toMerge] )

            x1_est_tmp = x1_est_tmp[~toMerge]
            y1_est_tmp = y1_est_tmp[~toMerge]
            x2_est_tmp = x2_est_tmp[~toMerge]
            y2_est_tmp = y2_est_tmp[~toMerge]

            # Add back point
            x1_est_tmp = np.append(x1_est_tmp,[avgX1])
            y1_est_tmp = np.append(y1_est_tmp,[avgY1])
            x2_est_tmp = np.append(x2_est_tmp,[avgX2])
            y2_est_tmp = np.append(y2_est_tmp,[avgY2])

    x1_est = x1_est_tmp
    y1_est = y1_est_tmp
    x2_est = x2_est_tmp
    y2_est = y2_est_tmp
    numPostMerge = x1_est.size
    #
    # Uniqueness test end
    ##################################################################

    ##################################################################
    # Distance test
    for i in range(x1_est.size):
        print('( %4.0f, %4.0f) to ( %4.0f, %4.0f)'%(x1_est[i], y1_est[i], x2_est[i], y2_est[i]), end='')

        # For each x1_est, find nearest ground truth point in x1
        x_dists = x1_est[i] - x1
        y_dists = y1_est[i] - y1
        dists = np.sqrt(  x_dists**2 + y_dists**2 )

        best_matches = np.argsort(dists.reshape((-1)))
        dists = dists[best_matches]

        current_offset = np.array([x1_est[i] - x2_est[i], y1_est[i] - y2_est[i]])
        most_similar_offset = np.array([x1[best_matches[0]] - x2[best_matches[0]], y1[best_matches[0]] - y2[best_matches[0]]])

        match_dist = np.sqrt( np.sum((current_offset.squeeze() - most_similar_offset.squeeze())**2))

        # A match is bad if there's no ground truth point within 150 pixels
        # or
        # If nearest ground truth correspondence offset isn't within 40 pixels
        # of the estimated correspondence offset.
        print(' g.t. point %4.0f px. Match error %4.0f px.'%(dists[0], match_dist), end='')

        if dists[0] > 150 or match_dist > 40:
            good_matches[i] = 0
            print('  incorrect')
        else:
            good_matches[i] = 1
            print('  correct')


    numGoodMatches = np.sum(good_matches)
    numBadMatches = x1_est.size - numGoodMatches
    print('Uniqueness: Pre-merge:    %d  Post-merge:  %d'% (numPreMerge,numPostMerge) )
    print('Total:      Good matches: %d  Bad matches: %d'% (numGoodMatches,numBadMatches) )

    # For evaluation, we're going to use the word 'accuracy'.
    # It's very difficult to count the number of actual correspondences in an
    # image, so ideas of recall are tricky to apply.
    # The second accuracy measure 'out of maxPtsToEval' captures some of that
    # idea: if you return less than maxPtsToEval, your accuracy will drop.
    #
    accuracyAll = (numGoodMatches / x1_est.size) * 100
    print('Accuracy:  %2.2f%% (on all %d submitted matches)'%(accuracyAll,x1_est.size))

    accuracyMaxEval = (np.sum(good_matches[:min(good_matches.size,maxPtsToEval)]) / maxPtsToEval) * 100
    print('Accuracy:  %2.2f%% (on first %d matches sorted by decreasing confidence)'%(accuracyMaxEval,maxPtsToEval))

    # Visualize the result
    if vismode is not None:
        # You may also switch to a different visualization method, by passing
        # 'arrows' into show_correspondence instead of 'dots'.
        show_correspondence(imgA, imgB, \
                            x1_est * scale_factor, y1_est * scale_factor, \
                            x2_est * scale_factor, y2_est * scale_factor, \
                            vismode, visfilename, good_matches)
    return numGoodMatches,numBadMatches,accuracyAll,accuracyMaxEval
