import cv2
import numpy as np


def vis_hybrid_image(hybrid_image, scales=5, scale_factor=0.5, padding=5):
    # scales: how many downsampled versions to create
    # scale_factor: how much to downsample each time
    # padding: how many pixels to pad

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]  # counting how many color channels the input has

    output = [hybrid_image]
    cur_image = hybrid_image
    for i in range(2, scales):
        # Add padding
        output.append(np.ones((original_height, padding, num_colors)))

        # Downsample image
        cur_image = cv2.resize(cur_image, dsize=None, fx=scale_factor, fy=scale_factor)

        # Pad the top and append to the output
        tmp = np.vstack([np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image])
        output.append(tmp)

    return np.hstack(output)
