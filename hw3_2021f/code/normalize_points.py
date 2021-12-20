import numpy as np

def normalize_points(pts, numDims): 
    # strip off the homogeneous coordinate
    points = pts[:numDims,:]

    # compute centroid
    cent = np.mean(points, axis=1)

    # translate points so that the centroid is at [0,0]
    translatedPoints = np.transpose(points.T - cent)

    # compute the scale to make mean distance from centroid sqrt(2)
    meanDistanceFromCenter = np.mean(np.sqrt(np.sum(np.power(translatedPoints,2), axis=0)))
    if meanDistanceFromCenter > 0: # protect against division by 0
        scale = np.sqrt(numDims) / meanDistanceFromCenter
    else:
        scale = 1.0

    # compute the matrix to scale and translate the points
    # the matrix is of the size numDims+1-by-numDims+1 of the form
    # [scale   0     ... -scale*center(1)]
    # [  0   scale   ... -scale*center(2)]
    #           ...
    # [  0     0     ...       1         ]    
    T = np.diag(np.array([*np.ones(numDims) * scale, 1], dtype=np.float))
    T[0:-1, -1] = -scale * cent;

    if pts.shape[0] > numDims:
        normPoints = T @ pts;
    else:
        normPoints = translatedPoints * scale;

    # the following must be true:
    # np.mean(np.sqrt(np.sum(np.power(normPoints[0:2,:],2), axis=0))) == np.sqrt(2)

    return normPoints, T