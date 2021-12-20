import numpy as np

from normalize_points import normalize_points


# HW3-b
# Implement normalized 8-point algorithm
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################
    pts1 = np.append(pts1, np.array([[1, 1, 1, 1, 1, 1, 1, 1]]).T, axis=1)
    pts2 = np.append(pts2, np.array([[1, 1, 1, 1, 1, 1, 1, 1]]).T, axis=1)
    p1, T1 = normalize_points(pts1.T, 2)
    p2, T2 = normalize_points(pts2.T, 2)
    p1_t = p1.T
    p2_t = p2.T
    A = np.zeros((8, 9))
    for i in range(8):
        A[i] = (p1_t[i].reshape((3, 1)) @ p2_t[i].reshape((3, 1)).T).reshape(1, 9)
    f = np.linalg.eig(np.matmul(A.T, A))[1][:, -1]
    F = f.reshape((3, 3)).T
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    fundamental_matrix = T2.T @ F @ T1
    fundamental_matrix = fundamental_matrix / fundamental_matrix[2][2]



    ################################################

    return fundamental_matrix