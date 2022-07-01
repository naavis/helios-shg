from serfilesreader import Serfile
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms
import skimage.transform
import skimage.feature
import skimage.draw
import skimage.morphology
import skimage.measure
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
    print(f"Absorption line distortion coefficients: {poly.coef}")
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


# EllipseModel from scikit-image does not give consistent results
# for the ellipse parameters. This function corrects them, so that
# the `a` axis is always the longest one, and the theta is the clockwise
# angle between the `a` axis and the positive horizontal axis.
def correct_ellipse_model_params(a: float, b: float, theta: float) -> tuple:
    if a < b:
        if theta < np.pi / 2:
            return b, a, theta + np.pi / 2
        else:
            return b, a, theta - np.pi / 2
    else:
        if theta < 0:
            return a, b, np.pi + theta
        else:
            return a, b, theta


def geometric_correction(image: np.ndarray) -> np.ndarray:
    xc, yc, a, b, theta = fit_ellipse(image)
    shear_angle = rot_to_shear(a, b, theta)
    # The shear angle needs some manipulation to be in the correct
    # range for our purposes. This ensures it is always centered
    # around 0 degrees with solar scans, and not -90 or 90 degrees.
    corrected_shear_angle = -(shear_angle + np.pi / 2 if shear_angle < 0 else shear_angle - np.pi / 2)
    print(f'Tilt: {np.rad2deg(corrected_shear_angle):.2f} degrees')

    # This is a bit of a hack to determine whether to squish or stretch the image,
    # i.e. whether the horizontal or vertical axis of the ellipse is the longer one
    if np.abs(theta - np.pi / 2) < np.pi / 4:
        scale = a / b
    else:
        scale = b / a
    print(f'Scale: {scale:.2f}')

    # Shearing with skimage contains a bug, so using matplotlib instead:
    # https://github.com/scikit-image/scikit-image/issues/3239

    # Correcting for shearing changes the scale a bit, making the scale correction a bit off.
    # It is so insignificant (on the order of half a pixel or less) that we don't care about it here, though.
    transform = matplotlib.transforms.Affine2D()
    transform.scale(1, scale)
    transform.skew(corrected_shear_angle, 0.0)
    output_shape = (int(image.shape[0] / scale), image.shape[1])
    corrected_image = skimage.transform.warp(image, transform.get_matrix(), order=3, output_shape=output_shape)
    return corrected_image


def show_image(image: np.ndarray):
    plt.imshow(image, cmap='gray', vmin=0.0)
    plt.show()


def fit_ellipse(image: np.ndarray) -> tuple:
    edges = skimage.feature.canny(image > 0.1 * image.max())
    edge_points = np.fliplr(np.argwhere(edges))

    # The condition below ensures that there are a sane number
    # of points on the ellipse. Scikit-image's EllipseModel
    # seems to get confused with long scans and lots of points
    if edge_points.shape[0] > 1000:
        edge_points = edge_points[::int(np.ceil(edge_points.shape[0] / 1000)), :]

    # plot_edge_points_on_image(image, edge_points)

    ellipse = skimage.measure.EllipseModel()
    estimation_successful = ellipse.estimate(edge_points)
    if not estimation_successful:
        raise Exception("Could not fit ellipse to image")
    xc, yc, a, b, theta = ellipse.params
    a, b, theta = correct_ellipse_model_params(a, b, theta)
    print(f'Found ellipse at: ({xc:.2f}, {yc:.2f}) with a: {a:.2f}, b: {b:.2f} and rotation {np.rad2deg(theta):.2f}°')

    # plot_ellipse_on_image(image, xc, yc, a, b, theta)

    return xc, yc, a, b, theta


def plot_ellipse_on_image(image: np.ndarray, xc: float, yc: float, a: float, b: float, theta: float):
    points = skimage.measure.EllipseModel().predict_xy(np.linspace(0.0, np.pi * 2, 100), [xc, yc, a, b, theta])
    plt.scatter(points[:, 0], points[:, 1])
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_edge_points_on_image(image, edge_points):
    plt.scatter(edge_points[:, 0], edge_points[:, 1])
    plt.imshow(image, cmap='gray', vmin=0.0)
    plt.show()


def rot_to_shear(a: float, b: float, theta: float):
    # Stolen from here: https://math.stackexchange.com/a/2510239
    slope = (a * a * np.tan(theta) + b * b / np.tan(theta)) / (a * a - b * b)
    return np.arctan(slope)


def process_video(filename: str) -> np.ndarray:
    input_file = Serfile(filename)
    ser_header = input_file.getHeader()
    print_headers(ser_header)
    # Pick reference frame from middle of video
    ref_frame_index = int(ser_header['FrameCount'] / 2)
    print(f"Using frame {ref_frame_index} as reference")
    ref_frame = input_file.readFrameAtPos(ref_frame_index)
    # Fit 2nd degree polynomial to dark emission line in reference frame
    poly = fit_poly_to_dark_line(ref_frame)
    _, poly_curve = poly.linspace(ref_frame.shape[1], domain=[0, ref_frame.shape[1]])
    output_frame = np.ndarray((input_file.getLength(), ref_frame.shape[1]))
    for i in range(0, input_file.getLength()):
        image = input_file.readFrameAtPos(i)
        output_frame[i, :] = get_emission_line(image, poly_curve)

    final_output = geometric_correction(output_frame).T
    return final_output


def main(args: [str]):
    final_output = process_video(args[0])
    show_image(final_output)


if __name__ == '__main__':
    main(sys.argv[1:])
