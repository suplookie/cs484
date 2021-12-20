import numpy as np
import cv2
import sys

def main(filename):
    print(filename)
    img = cv2.imread(filename)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgray = np.float32(imgray)
    dst = cv2.cornerHarris(imgray, 2, 3, 0.04)

    img[dst>0.01*dst.max()] = [0, 0, 255]

    cv2.imwrite("q1_" + filename, img)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit()
    main(sys.argv[1])
