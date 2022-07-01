import numpy as np
from numba import njit
from numpy.polynomial.polynomial import Polynomial


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


@njit(cache=True)
def get_absorption_line(image: np.ndarray, poly_curve: np.ndarray) -> np.ndarray:
    output = np.zeros_like(poly_curve)
    for i in range(0, poly_curve.size):
        x = poly_curve[i]
        lower = int(np.floor(x))
        delta = x - lower
        output[i] = (1.0 - delta) * image[lower, i] + delta * image[lower + 1, i]
    return output
