import glob
import pathlib
import sys

import imageio
import numpy as np

from solex_read import process_video
from solex.utils import crop_image


def batch_process(args):
    glob_pattern = args[0]
    files = glob.glob(glob_pattern)
    files.sort()
    print(f'Found files: {files}')
    for f in files:
        print(f'Processing: {f}')
        result_image = process_video(f)
        cropped_image = crop_image(result_image)
        new_file_path = pathlib.Path(f).with_suffix('.png')
        imageio.imwrite(new_file_path, cropped_image.astype(np.uint16))


if __name__ == '__main__':
    batch_process(sys.argv[1:])
