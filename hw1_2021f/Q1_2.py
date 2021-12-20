import time
import cv2
import numpy as np

A = cv2.imread('grizzlypeakg.png',0)
start_1 = time.time()
for x in range(100):
    m1, n1 = A.shape
    for i in range(m1):
        for j in range(n1):
            if A[i,j] <= 10:
                A[i,j] = 0
end_1 = time.time()

A = cv2.imread('grizzlypeakg.png',0)
start_2 = time.time()
for x in range(100):
    B = A <= 10
    A[B] = 0
end_2 = time.time()

print("factor speedup:", (end_1 - start_1) / (end_2 - start_2))

print(end_1 - start_1, end_2 - start_2)

