# Before trying to construct hybrid images,
# it is suggested that you implement my_filter2D.py and then debug it using hw2_testcase.py

import cv2
import os

from gen_hybrid_image import gen_hybrid_image
from vis_hybrid_image import vis_hybrid_image


def hw2():
    ## Setup
    # Read images and convert to floating point format
    image1 = cv2.imread('../data/dog.bmp') / 255.0
    image2 = cv2.imread('../data/cat.bmp') / 255.0

    # For your write up, there are several additional test cases in 'data'. Feel free to make your own, too (you'll
    # need to align the images in a photo editor such as Photoshop). The hybrid images will differ depending on which
    # image you assign as image1 (which will provide the low frequencies) and which image you assign as image2 (which
    # will provide the high frequencies)

    ## Hybrid Image Construction
    # cutoff_frequency is the standard deviation, in pixels, of the Gaussian blur that will remove high frequencies.
    # You may tune this per image pair to achieve better results.

    cutoff_frequency = 7
    hybrid_image, low_frequencies, high_frequencies = gen_hybrid_image(image1, image2, cutoff_frequency)
    vis = vis_hybrid_image(hybrid_image)

    cv2.imshow('low_frequencies', low_frequencies)
    cv2.imshow('high_frequencies + 0.5', high_frequencies + 0.5)
    cv2.imshow('hybrid_image', hybrid_image)
    cv2.imshow('hybrid_image_scales', vis)

    result_dir = '../result'
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(os.path.join(result_dir, 'low_frequencies.jpg'), low_frequencies * 255)
    cv2.imwrite(os.path.join(result_dir, 'high_frequencies.jpg'), (high_frequencies + 0.5) * 255)
    cv2.imwrite(os.path.join(result_dir, 'hybrid_image.jpg'), hybrid_image * 255)
    cv2.imwrite(os.path.join(result_dir, 'hybrid_image_scales.jpg'), vis * 255)

    print('Press any key ...')
    cv2.waitKey(0)


if __name__ == '__main__':
    hw2()
