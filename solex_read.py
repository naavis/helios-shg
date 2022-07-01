import glob
import os
import pathlib

import click
import imageio
import numpy as np

from solex.utils import show_image, crop_image
from solex.processor import process_video


@click.command()
@click.option('--save', is_flag=True, default=False, help='Save image to PNG file')
@click.option('--no-show', is_flag=True, default=False, help='Do not show image after processing')
@click.option('--no-crop', is_flag=True, default=False, help='Do not do automatic cropping')
@click.option('--ref-frame', type=int, help='Reference frame for finding absorption line, defaults to middle frame')
@click.argument('files', nargs=-1)
def solex_read(files, save, no_show, no_crop, ref_frame):
    """Processes Sol'Ex spectroheliograph videos into narrowband still images."""
    if os.name == 'nt':
        # Windows does not automatically expand wildcards, so that has to be done with the glob module.
        # Note that technically Windows allows square brackets in filenames, so this solution
        # is not perfect. They will be interpreted as wildcard characters in this case.
        files = [glob.glob(f) if any(c in f for c in ['*', '?', '[', ']']) else f for f in files]
    for f in files:
        click.echo(f'Processing: {f}')
        result = process_video(f, ref_frame)
        if not no_crop:
            result = crop_image(result)
        click.echo(f'Result image size: {result.shape[0]}x{result.shape[1]} pixels')
        if save:
            new_file_path = pathlib.Path(f).with_suffix('.png')
            click.echo(f'Saving result to: {new_file_path}')
            imageio.imwrite(new_file_path, result.astype(np.uint16))
        if not no_show:
            show_image(result)
        click.echo('')


if __name__ == '__main__':
    solex_read()
