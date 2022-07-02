import matplotlib.pyplot as plt
import numpy as np

from solex.correction import fit_ellipse


def show_image(image: np.ndarray):
    plt.imshow(image, cmap='gray', vmin=0.0)
    plt.show()


def crop_image(image: np.ndarray, desired_side_length: int = None) -> np.ndarray:
    xc, yc, a, b, theta = fit_ellipse(image)
    if desired_side_length is None:
        desired_side_length = int(max(a, b) * 2 * 1.2)

    x_min = int(np.floor(xc - desired_side_length / 2))
    x_max = int(np.ceil(xc + desired_side_length / 2))
    x_padding = max(max(-x_min, 0), max(x_max - image.shape[1], 0))

    y_min = int(np.floor(yc - desired_side_length / 2))
    y_max = int(np.ceil(yc + desired_side_length / 2))
    y_padding = max(max(-y_min, 0), max(y_max - image.shape[0], 0))

    padding = max(x_padding, y_padding)
    padded_image = np.pad(image, padding)

    xc_padding = int(xc + padding)
    yc_padding = int(yc + padding)
    cropped_image = padded_image[(yc_padding - desired_side_length // 2):(yc_padding + desired_side_length // 2),
                                 (xc_padding - desired_side_length // 2):(xc_padding + desired_side_length // 2)]
    return cropped_image
