import numpy as np
import cv2

# HW3-d
# Generate the disparity map from two rectified images.
# Use NCC for the matching cost function.
def calculate_disparity_map(img_left, img_right, window_size, max_disparity):
    # Your code here
    ################################################
    h, w = img_right.shape
    cost_volume = np.zeros((h-window_size, w-window_size, max_disparity))
    slice_left = np.zeros((h - window_size, w - window_size, window_size, window_size))
    slice_right = np.zeros((h - window_size, w - window_size, window_size, window_size))
    left_ss = np.zeros((h - window_size, w - window_size, max_disparity))
    right_ss = np.zeros((h - window_size, w - window_size, max_disparity))
    corr =  np.zeros((h - window_size, w - window_size, max_disparity))

    for j in range(w - window_size):
        for i in range(h - window_size):
            slice_left[i, j] = img_left[i:i+window_size, j:j+window_size] - np.mean(img_left[i:i+window_size, j:j+window_size])
            left_ss[i, j, :] = np.linalg.norm(slice_left[i, j])
            slice_right[i, j] = img_right[i:i+window_size, j:j+window_size] - np.mean(img_right[i:i+window_size, j:j+window_size])
            right_ss[i, j, 0] = np.linalg.norm(slice_right[i, j])
    for i in range(h - window_size):
        for j in range(w - window_size):
            for k in range(min(w, j + window_size + max_disparity) - j - window_size):
                right_ss[i, j, k] = right_ss[i, j + k, 0]
                corr[i, j, k] = np.sum(slice_left[i, j] * slice_right[i, j + k])
    cost_volume = corr / (left_ss * right_ss)
    disparity_map = np.argmax(cost_volume, axis=2)




    ################################################

    return disparity_map
