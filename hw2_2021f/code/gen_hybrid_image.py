import cv2
import numpy as np
from my_filter2D import my_filter2D


def gen_hybrid_image(image1, image2, cutoff_frequency):
    # Inputs:
    # - image1 -> The image from which to take the low frequencies.
    # - image2 -> The image from which to take the high frequencies.
    # - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian blur that will remove high frequencies.
    #
    # Task:
    # - Use my_filter2D to create 'low_frequencies' and 'high_frequencies'.
    # - Combine them to create 'hybrid_image'.

    ########################################################################
    # Remove the high frequencies from image1 by blurring it.
    # The amount of blur that works best will vary with different image pairs.
    ########################################################################
    # 1-D Gaussian kernel
    kernel = cv2.getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency)
    # 2-D Guassian kernel
    kernel = np.matmul(kernel, kernel.T)
    low_frequencies = my_filter2D(image1, kernel)      # Your code here

    ########################################################################
    # Remove the low frequencies from image2.
    # The easiest way to do this is to subtract a blurred version of image2 from the original version of image2.
    # This will give you an image centered at zero with negative values.
    ########################################################################
    high_frequencies = image2 - my_filter2D(image2, kernel)     # Your code here

    ########################################################################
    # Combine the high frequencies and low frequencies
    ########################################################################
    hybrid_image = low_frequencies + high_frequencies         # Your code here

    return hybrid_image, low_frequencies, high_frequencies
