import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.ndarray):
    plt.imshow(image, cmap='gray', vmin=0.0)
    plt.show()
