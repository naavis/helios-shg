from serfilesreader import Serfile
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from numpy.polynomial import Polynomial
from numba import njit


def print_headers(input_file):
    header = input_file.getHeader()
    print('Found the following headers:')
    for header_key in header:
        print(f'\t{header_key}: {header[header_key]}')


def fit_poly_to_dark_line(data: np.ndarray) -> Polynomial:
    x = np.arange(0, data.shape[1])
    y = np.argmin(data, axis=0)
    poly = np.polynomial.Polynomial.fit(x, y, 2)
    return poly


def warp_function(cr: np.ndarray, offset: np.ndarray) -> np.ndarray:
    output = cr.copy()
    output[:, 1] += offset
    return output


def fix_image_curvature(image: np.ndarray, curve: np.ndarray) -> np.ndarray:
    # Explicitly defining the warped shape ensures all the input data fits in the new distortion-corrected image
    warped_shape = (image.shape[0] + int(curve[0] - curve.min()), image.shape[1])
    # The offset curve must be repeated to match the warp_function input data shape, and curve[0] is subtracted to keep
    # the start of the emission line on the same row both in the input data and output data
    offset = np.repeat(curve, warped_shape[0]) - curve[0]
    warped_image = skimage.transform.warp(image, warp_function, {'offset': offset}, output_shape=warped_shape)
    return warped_image


@njit
def get_emission_line(image: np.ndarray, poly_curve: np.ndarray) -> np.ndarray:
    output = np.zeros_like(poly_curve)
    for i in range(0, poly_curve.size):
        x = poly_curve[i]
        lower = int(np.floor(x))
        delta = x - lower
        output[i] = (1.0 - delta) * image[lower, i] + delta * image[lower + 1, i]
    return output


def scale_correction(image: np.ndarray) -> np.ndarray:
    scaling_ratio = image.sum(axis=1).std() / image.sum(axis=0).std()
    return skimage.transform.resize(image, (int(image.shape[0] * scaling_ratio), int(image.shape[1])))


def main(args):
    input_file = Serfile(args[0])
    print_headers(input_file)

    # Pick reference frame from middle of video
    ref_frame = input_file.readFrameAtPos(1000)

    # Fit 2nd degree polynomial to dark emission line in reference frame
    poly = fit_poly_to_dark_line(ref_frame)
    _, poly_curve = poly.linspace(ref_frame.shape[1])
    #emission_row = int(poly_curve[0])

    output_frame = np.ndarray((input_file.getLength(), ref_frame.shape[1]))
    for i in range(0, input_file.getLength()):
        image = input_file.readFrameAtPos(i)
        #warped_image = fix_image_curvature(image, poly_curve)
        #output_frame[i, :] = warped_image[emission_row, :]
        output_frame[i, :] = get_emission_line(image, poly_curve)

    final_output = scale_correction(output_frame)
    plt.imshow(final_output.T, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
