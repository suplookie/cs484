import cv2
import numpy as np


# HW3-a
# Generate rgb image from bayer pattern image
def bayer_to_rgb(bayer_img, mode):
    assert mode=='bilinear' or mode=='bicubic'
    if mode == 'bilinear':
        # Implement demosaicing using bilinear interpolation.
        # Your code here
        ################################################
        rgb_img = np.zeros((bayer_img.shape[0], bayer_img.shape[1], 3), dtype=np.uint8)
        for i in range(bayer_img.shape[0]):
            for j in range(bayer_img.shape[1]):
                if i % 2 and j % 2:
                    count_red = 1
                    red_val = int(bayer_img[i - 1][j - 1])
                    count_green = 2
                    green_val = int(bayer_img[i][j - 1]) + int(bayer_img[i - 1][j])
                    if i + 1 < bayer_img.shape[0]:
                        count_red += 1
                        red_val += bayer_img[i + 1][j - 1]
                        count_green += 1
                        green_val += bayer_img[i + 1][j]
                        if j + 1 < bayer_img.shape[1]:
                            count_red += 2
                            red_val += int(bayer_img[i - 1][j + 1]) + int(bayer_img[i + 1][j + 1])
                            count_green += 1
                            green_val += bayer_img[i][j + 1]
                    elif j + 1 < bayer_img.shape[1]:
                        count_red + 1
                        red_val += bayer_img[i - 1][j + 1]
                        count_green += 1
                        green_val += bayer_img[i][j + 1]
                    rgb_img[i][j][0] = red_val // count_red
                    rgb_img[i][j][1] = green_val // count_green
                    rgb_img[i][j][2] = bayer_img[i][j]

                elif not (i % 2 or j % 2):
                    count_green = 2
                    green_val = int(bayer_img[i + 1][j]) + int(bayer_img[i][j + 1])
                    count_blue = 1
                    blue_val = int(bayer_img[i + 1][j + 1])
                    if i - 1 >= 0:
                        count_green += 1
                        green_val += bayer_img[i - 1][j]
                        count_blue += 1
                        blue_val += bayer_img[i - 1][j + 1]
                        if j - 1 >= 0:
                            count_green += 1
                            green_val += bayer_img[i][j - 1]
                            count_blue += 2
                            blue_val += int(bayer_img[i - 1][j - 1]) + int(bayer_img[i + 1][j - 1])
                    elif j - 1 >= 0:
                        count_green += 1
                        green_val += bayer_img[i][j -1]
                        count_blue += 1
                        blue_val += bayer_img[i + 1][j - 1]
                    rgb_img[i][j][0] = bayer_img[i][j]
                    rgb_img[i][j][1] = green_val // count_green
                    rgb_img[i][j][2] = blue_val // count_blue
                else:
                    count_red = 1
                    count_blue = 1
                    if i % 2:
                        red_val = int(bayer_img[i - 1][j])
                        blue_val = int(bayer_img[i][j + 1])
                        if i + 1 < bayer_img.shape[0]:
                            count_red += 1
                            red_val += bayer_img[i + 1][j]
                            if j - 1 >= 0:
                                count_blue += 1
                                blue_val += bayer_img[i][j - 1]
                        elif j - 1 >= 0:
                            count_blue += 1
                            blue_val += bayer_img[i][j - 1]
                    else:
                        red_val = int(bayer_img[i][j - 1])
                        blue_val = int(bayer_img[i + 1][j])
                        if i - 1 >= 0:
                            count_blue += 1
                            blue_val += bayer_img[i - 1][j]
                            if j + 1 < bayer_img.shape[1]:
                                count_red += 1
                                red_val += bayer_img[i][j + 1]
                        elif j + 1 < bayer_img.shape[1]:
                            count_red += 1
                            red_val += bayer_img[i][j + 1]
                    rgb_img[i][j][0] = red_val // count_red
                    rgb_img[i][j][1] = bayer_img[i][j]
                    rgb_img[i][j][2] = blue_val // count_blue


        ################################################
    elif mode == 'bicubic':
        # Optional: Implement demosaicing using bicubic interpolation.
        # Your code here
        ################################################
        rgb_img = np.zeros((bayer_img.shape[0], bayer_img.shape[1], 3), dtype=np.uint8)


        ################################################
    
    return rgb_img