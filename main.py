from serfilesreader import Serfile
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from numpy.polynomial import Polynomial


def fit_poly_to_dark_line(data: np.ndarray) -> Polynomial:
    x = np.arange(0, data.shape[1])
    y = np.argmin(data, axis=0)
    poly = np.polynomial.Polynomial.fit(x, y, 2)
    return poly


def plot_image_with_fit(data: np.ndarray, poly: Polynomial):
    plt.imshow(data, cmap='gray')
    plt.plot(*poly.linspace(data.shape[1]))
    plt.show()


def print_headers(input_file):
    header = input_file.getHeader()
    print('Found the following headers:')
    for header_key in header:
        print(f'\t{header_key}: {header[header_key]}')


def warp_function(cr: np.ndarray, curve: np.ndarray) -> np.ndarray:
    output = cr.copy()
    resized_curve = np.repeat(curve, int(output.shape[0] / curve.shape[0]))
    output[:, 1] = cr[:, 1] + resized_curve - curve[0]
    return output


def fix_image_curvature(image: np.ndarray, curve: np.ndarray) -> np.ndarray:
    warped_shape = (image.shape[0] + int(curve[0] - curve.min()), image.shape[1])
    warped_image = skimage.transform.warp(image, warp_function, {'curve': curve}, output_shape=warped_shape)
    return warped_image


def main():
    input_file = Serfile(r'C:\Users\samul\Desktop\12_08_34\12_08_34.ser')
    print_headers(input_file)

    # Pick reference frame from middle of video
    ref_frame = input_file.readFrameAtPos(1000)

    # Fit 2nd degree polynomial to dark emission line in reference frame
    poly = fit_poly_to_dark_line(ref_frame)
    _, poly_curve = poly.linspace(ref_frame.shape[1])
    emission_row = int(poly_curve[0])

    output_frame = np.ndarray((input_file.getLength(), ref_frame.shape[1]))
    for i in range(0, input_file.getLength()):
        image = input_file.readFrameAtPos(i)
        warped_image = fix_image_curvature(image, poly_curve)
        output_frame[i, :] = warped_image[emission_row, :]
    plt.imshow(output_frame, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
