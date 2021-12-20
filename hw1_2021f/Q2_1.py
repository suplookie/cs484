import cv2
import numpy as np
I = cv2.imread('gigi.jpg').astype(np.uint8)
B = I < 40
I = I - 40
I[B] = 0
cv2.imwrite('result.png', I)