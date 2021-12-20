import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

def f(x, y):
    ret = np.zeros((7, 6))
    for i in range(7):
        for j in range(6):
            f = np.ones((y[i, j],y[i, j]))/y[i, j]**2
            src = cv2.imread('RISDance.jpg', 0)
            resize = cv2.resize(src, dsize=(0,0), fx = math.sqrt(x[i, j] / 8), fy = math.sqrt(x[i, j] / 8))
            start = time.time()
            dst = cv2.filter2D(resize, -1, f)
            end = time.time()
            ret[i,j] = end - start
    return ret

x = np.array([(2 ** i) * 0.25 for i in range(6)])
y = np.arange(3, 16, 2)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print(X)
print(Y)
print(Z)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('image size')
ax.set_ylabel('filter size')
ax.set_zlabel('computation time');
plt.savefig('result.png')
