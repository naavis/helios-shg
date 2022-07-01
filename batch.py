import imageio
import numpy as np
import glob
import pathlib
import sys

import solex.correction
from main import process_video


def batch_process(args):
    glob_pattern = args[0]
    files = glob.glob(glob_pattern)
    files.sort()
    print(f'Found files: {files}')
    for f in files:
        print(f'Processing: {f}')
        new_file_path = pathlib.Path(f).with_suffix('.png')
        result_image = process_video(f)
        cropped_image = crop_image(result_image)
        imageio.imwrite(new_file_path, cropped_image.astype(np.uint16))


def crop_image(image, desired_side_length=None):
    xc, yc, a, b, theta = solex.correction.fit_ellipse(image)
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


if __name__ == '__main__':
    batch_process(sys.argv[1:])
