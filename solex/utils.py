import matplotlib.pyplot as plt
import numpy as np


def show_image(image: np.ndarray):
    plt.imshow(image, cmap='gray', vmin=0.0)
    plt.show()
