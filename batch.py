import imageio
import scipy.ndimage
import numpy as np
import glob
import pathlib
import sys

from main import process_video


def batch_process(args):
    glob_pattern = args[0]
    files = glob.glob(glob_pattern)
    print(f'Found files: {files}')
    for f in files:
        new_file_path = pathlib.Path(f).with_suffix('.png')
        result_image = process_video(f)
        center_of_mass = scipy.ndimage.center_of_mass(result_image)
        desired_side_length = 1700
        padding = 1000
        com_x = padding + int(np.floor(center_of_mass[0]))
        com_y = padding + int(np.floor(center_of_mass[1]))
        padded_image = np.pad(result_image, padding)
        cropped_image = padded_image[(com_x - desired_side_length // 2):(com_x + desired_side_length // 2), (com_y - desired_side_length // 2):(com_y + desired_side_length // 2)]
        imageio.imwrite(new_file_path, cropped_image.astype(np.uint16))


if __name__ == '__main__':
    batch_process(sys.argv[1:])
