import time
import cv2
import numpy as np
start_time = time.time()
A = cv2.imread('grizzlypeakg.png',0)
m1, n1 = A.shape
for i in range(m1):
    for j in range(n1):
        if A[i,j] <= 10:
            A[i,j] = 0
checkpoint1 = time.time()
_A = cv2.imread('grizzlypeakg.png',0)
B = _A <= 10
_A[B] = 0
checkpoint2 = time.time()

if np.array_equal(A, _A):
    print(checkpoint1 - start_time, checkpoint2 - checkpoint2)
    print(start_time, checkpoint1, checkpoint2)
else:
    print("array not equal!")

