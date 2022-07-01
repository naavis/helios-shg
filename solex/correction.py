import click
import matplotlib.transforms
import numpy as np
import skimage.draw
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.transform
from numba import njit


def geometric_correction(image: np.ndarray) -> np.ndarray:
    xc, yc, a, b, theta = fit_ellipse(image)
    click.echo(f'Found sun ellipse at ({xc:.2f}, {yc:.2f}), with '
               f'a: {a:.2f}, b: {b:.2f} and rotation {np.rad2deg(theta):.2f}°')
    shear_angle = rot_to_shear(a, b, theta)
    # The shear angle needs some manipulation to be in the correct
    # range for our purposes. This ensures it is always centered
    # around 0 degrees with solar scans, and not -90 or 90 degrees.
    corrected_shear_angle = -(shear_angle + np.pi / 2 if shear_angle < 0 else shear_angle - np.pi / 2)
    click.echo(f'Tilt: {np.rad2deg(corrected_shear_angle):.2f} degrees')
    if np.abs(np.rad2deg(corrected_shear_angle)) > 4.0:
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
    transform.scale(1, scale)
    transform.skew(corrected_shear_angle, 0.0)
    output_shape = (int(image.shape[0] / scale), image.shape[1])
    corrected_image = skimage.transform.warp(image, transform.get_matrix(), order=3, output_shape=output_shape)
    return corrected_image


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
