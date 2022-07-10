import click
import matplotlib.transforms
import numpy as np
import scipy
import skimage.draw
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.transform
from numba import njit


def geometric_correction(image: np.ndarray) -> (np.ndarray, (float, float), float):
    xc, yc, a, b, theta = fit_ellipse(image)
    click.echo(f'Found sun ellipse at ({xc:.2f}, {yc:.2f}), with '
               f'a: {a:.2f}, b: {b:.2f} and rotation {np.rad2deg(theta):.2f}°')
    shear_angle = rot_to_shear(a, b, theta)
    # The shear angle needs some manipulation to be in the correct
    # range for our purposes. This ensures it is always centered
    # around 0 degrees with solar scans, and not -90 or 90 degrees.
    corrected_shear_angle = shear_angle + np.pi / 2 if shear_angle < 0 else shear_angle - np.pi / 2
    click.echo(f'Tilt: {np.rad2deg(corrected_shear_angle):.2f} degrees')
    if np.abs(np.rad2deg(corrected_shear_angle)) > 5.0:
        click.secho('Significant tilt detected! '
                    'Consider aligning your instrument or mount better.', fg='red', bold=True)

    # This is a bit of a hack to determine whether to squish or stretch the image,
    # i.e. whether the horizontal or vertical axis of the ellipse is the longer one
    if np.abs(theta - np.pi / 2) < np.pi / 4:
        scale = a / b
    else:
        scale = b / a
    click.echo(f'Scale: {scale:.2f}')

    if scale < 1.0:
        click.secho('Data has insufficient number of frames compared to resolution! '
                    'Consider using a slower scan speed or increasing frame rate.', fg='red', bold=True)

    # Shearing with skimage contains a bug, so using matplotlib instead:
    # https://github.com/scikit-image/scikit-image/issues/3239

    # Correcting for shearing changes the scale a bit, making the scale correction a bit off.
    # It is so insignificant (on the order of half a pixel or less) that we don't care about it here, though.

    transform = matplotlib.transforms.Affine2D()
    transform.skew(corrected_shear_angle, 0.0)
    transform.scale(1, 1.0 / scale)
    transform_matrix = transform.get_matrix()

    # Calculate center and diameter after geometry correction
    new_xc, new_yc = np.dot(transform_matrix[:2, :2], np.array([xc, yc]))
    if scale < 1.0:
        diameter = 2 * a
    else:
        diameter = 2 * b

    output_shape_x = int(image.shape[0] / scale)
    output_shape_y = int(np.ceil(image.shape[1] + image.shape[0] * np.tan(corrected_shear_angle)))
    output_shape = (output_shape_x, output_shape_y)
    inv_matrix = np.linalg.inv(transform_matrix)
    corrected_image = skimage.transform.warp(image, inv_matrix, order=3, output_shape=output_shape)

    return corrected_image, (new_xc, new_yc), diameter


@njit(cache=True)
def rot_to_shear(a: float, b: float, theta: float) -> float:
    # Stolen from here: https://math.stackexchange.com/a/2510239
    slope = (a * a * np.tan(theta) + b * b / np.tan(theta)) / (a * a - b * b)
    return np.arctan(slope)


def fit_ellipse(image: np.ndarray) -> tuple:
    edges = skimage.feature.canny(image > 0.1 * image.max())
    edge_points = np.fliplr(np.argwhere(edges))

    # The condition below ensures that there are a sane number
    # of points on the ellipse. Scikit-image's EllipseModel
    # seems to get confused with long scans and lots of points
    if edge_points.shape[0] > 100:
        edge_points = edge_points[::int(np.ceil(edge_points.shape[0] / 100)), :]

    ellipse = skimage.measure.EllipseModel()
    estimation_successful = ellipse.estimate(edge_points)
    if not estimation_successful:
        raise RuntimeError("Could not fit ellipse to image")
    xc, yc, a, b, theta = ellipse.params
    a, b, theta = correct_ellipse_model_params(a, b, theta)

    return xc, yc, a, b, theta


# EllipseModel from scikit-image does not give consistent results
# for the ellipse parameters. This function corrects them, so that
# the `a` axis is always the longest one, and the theta is the clockwise
# angle between the `a` axis and the positive horizontal axis.
@njit(cache=True)
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


def extract_transversallium(image: np.ndarray) -> np.ndarray:
    masked_image = np.ma.masked_less(image, 0.2 * image.max())
    median = np.ma.median(masked_image, axis=0)
    smooth_median = scipy.signal.medfilt(median, 201)
    highpass_median = median / smooth_median
    correction_factors = np.ma.filled(highpass_median, 1.0)
    return correction_factors


def transversallium_correction(image: np.ndarray, transversallium_factors: np.ndarray) -> np.ndarray:
    return image / transversallium_factors
