from serfilesreader import Serfile
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from numpy.polynomial import Polynomial
from numba import njit


def print_headers(header: dict):
    print('SER file headers:')
    for header_key in header:
        print(f'\t{header_key}: {header[header_key]}')


def fit_poly_to_dark_line(data: np.ndarray) -> Polynomial:
    # Limit polynomial fitting only to region with proper signal
    columns_with_signal = data.max(axis=0) > (0.3 * data.max())
    first_index = np.nonzero(columns_with_signal)[0][0]
    last_index = np.nonzero(columns_with_signal)[0][-1]

    x = np.arange(first_index, last_index)
    y = np.argmin(data[:, first_index:last_index], axis=0)
    poly = Polynomial.fit(x, y, 2)
    print(f"Distortion coefficients: {poly.coef}")
    return poly


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
    sun_pixels = np.nonzero(1.0 * (image > 0.3 * image.max()))
    vertical_range = sun_pixels[0].max() - sun_pixels[0].min()
    horizontal_range = sun_pixels[1].max() - sun_pixels[1].min()
    scaling_ratio = horizontal_range / vertical_range
    print(f"Scale correction: {scaling_ratio}")
    return skimage.transform.resize(image, (int(image.shape[0] * scaling_ratio), int(image.shape[1])))


def tilt_correction(image: np.ndarray) -> np.ndarray:
    # Limit fitting to areas with proper signal
    rows_with_signal = image.max(axis=1) > (0.3 * image.max())
    first_index = np.nonzero(rows_with_signal)[0][0]
    last_index = np.nonzero(rows_with_signal)[0][-1]

    # Calculate tilt angle by fitting a line through the center of mass of each row of data
    x = np.arange(first_index, last_index)
    sum_masses = image[first_index:last_index, :].sum(axis=1)
    horizontal_coordinates = np.tile(np.arange(0, image.shape[1]), (last_index - first_index, 1))
    y = (image[first_index:last_index, :] * horizontal_coordinates).sum(axis=1) / sum_masses

    poly = Polynomial.fit(x, y, 1)
    shift = poly.convert().coef[1]
    print(f"Tilt: {np.rad2deg(np.arctan(shift))} degrees")

    return skimage.transform.warp(image, skimage.transform.AffineTransform(shear=-np.arctan(shift)))


def plot_intermediary_images(ref_frame, raw_output_frame, tilt_corrected, final_output, curve_x, curve_y):
    fig, ax = plt.subplots(2, 2, tight_layout=True, frameon=False)
    ax[0, 0].imshow(ref_frame, cmap='gray')
    ax[0, 0].plot(curve_x, curve_y)
    ax[0, 0].set_title('Reference frame')
    ax[0, 1].imshow(raw_output_frame.T, cmap='gray')
    ax[0, 1].set_title('Before correction')
    ax[1, 0].imshow(tilt_corrected.T, cmap='gray')
    ax[1, 0].set_title('After tilt correction')
    ax[1, 1].imshow(final_output.T, cmap='gray')
    ax[1, 1].set_title('After scale correction')
    plt.show()


def main(args):
    input_file = Serfile(args[0])
    ser_header = input_file.getHeader()
    print_headers(ser_header)

    # Pick reference frame from middle of video
    ref_frame_index = int(ser_header['FrameCount'] / 2)
    print(f"Using frame {ref_frame_index} as reference")
    ref_frame = input_file.readFrameAtPos(ref_frame_index)

    # Fit 2nd degree polynomial to dark emission line in reference frame
    poly = fit_poly_to_dark_line(ref_frame)
    poly_curve_x, poly_curve = poly.linspace(ref_frame.shape[1], domain=[0, ref_frame.shape[1]])

    output_frame = np.ndarray((input_file.getLength(), ref_frame.shape[1]))
    for i in range(0, input_file.getLength()):
        image = input_file.readFrameAtPos(i)
        output_frame[i, :] = get_emission_line(image, poly_curve)

    tilt_corrected = tilt_correction(output_frame)
    final_output = scale_correction(tilt_corrected)

    plot_intermediary_images(ref_frame, output_frame, tilt_corrected, final_output, poly_curve_x, poly_curve)

    plt.imshow(final_output.T, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
