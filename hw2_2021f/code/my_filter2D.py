import cv2


def my_filter2D(image, kernel, zero_pad=False, fft=True):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    # Your code here
    ################
    import numpy as np
    def testShape(size):
        if not size % 2:
            raise KernelShapeError

    kernel_shape = kernel.shape
    try:
        testShape(kernel_shape[0])
        testShape(kernel_shape[1])
    except KernelShapeError as e:
        print(e)

    img_shape = image.shape
    out = image.copy()
    npad = ((kernel_shape[0] // 2, kernel_shape[0] // 2), (kernel_shape[1] // 2, kernel_shape[1] // 2))
    if len(img_shape) == 3 and img_shape[2] == 3:
        #color
        color = True
        r_, g_, b_ = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if zero_pad:
            rb = np.pad(r_, npad, mode='constant', constant_values=0)
            gb = np.pad(g_, npad, mode='constant', constant_values=0)
            bb = np.pad(b_, npad, mode='constant', constant_values=0)
        else:
            rb = np.pad(r_, npad, mode='reflect')
            gb = np.pad(g_, npad, mode='reflect')
            bb = np.pad(b_, npad, mode='reflect')
        padded = np.dstack(tup=(rb, gb, bb))

    else:
        #gray
        color = False
        if zero_pad:
            padded = np.pad(img, npad, mode='constant', constant_values=(0))
        else:
            padded = np.pad(img, npad, mode='reflect')
    if fft:
        k_pad = ((image.shape[0] - kernel.shape[0], 0), (image.shape[1] - kernel.shape[1], 0))
        p_kernel = np.pad(kernel, k_pad, mode='constant', constant_values=0)
        p_kernel = np.roll(p_kernel, kernel.shape[0]//2 + 1, axis=0)
        p_kernel = np.roll(p_kernel, kernel.shape[1]//2 + 1, axis=1)

        if color:
            out_r = np.fft.ifft2(np.fft.fft2(image[:, :, 0])*np.fft.fft2(p_kernel))
            out_g = np.fft.ifft2(np.fft.fft2(image[:, :, 1])*np.fft.fft2(p_kernel))
            out_b = np.fft.ifft2(np.fft.fft2(image[:, :, 2])*np.fft.fft2(p_kernel))
            out = np.dstack(tup=(out_r.real, out_g.real, out_b.real))
        else:
            out = (np.fft.ifft2(np.fft.fft2(image)*np.fft.fft2(p_kernel))).real

    else:
        for m in range(img_shape[0]):
            for n in range(img_shape[1]):
                val = 0
                for k in range(kernel_shape[0]):
                    for l in range(kernel_shape[1]):
                        val += kernel[k][l] * padded[m - k + kernel_shape[0]-1][n - l + kernel_shape[1]-1]
                out[m][n] = val
    return out


class KernelShapeError(Exception):
    def __str__(self):
        return "The filter is not odd-dimension. "
