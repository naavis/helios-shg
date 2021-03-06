import glob
import os
import pathlib
import time
from collections.abc import Iterable

import click
import imageio
import numpy as np

from helios.processor import process_video
from helios.utils import show_image, crop_image


@click.command()
@click.option('--save', is_flag=True, default=False, help='Save image to PNG file.')
@click.option('--no-show', is_flag=True, default=False, help='Do not show image after processing.')
@click.option('--no-crop', is_flag=True, default=False, help='Do not do automatic cropping. Overrides --output-size.')
@click.option('--output-size', type=int, help='Desired image side length in pixels. Handy for matching the size of '
                                              'several images for stacking!')
@click.option('--ref-frame', type=int, help='Reference frame for finding absorption line, defaults to middle frame.')
@click.option('--flipv', is_flag=True, help='Flip result vertically.')
@click.option('--fliph', is_flag=True, help='Flip result horizontally.')
@click.option('--no-transversallium', is_flag=True, default=False, help='Do not apply transversallium correction.')
@click.option('--continuum-offset', type=int, default=15, help='Offset in pixels from absorption line used to measure '
                                                               'continuum signal for transversallium correction.')
@click.argument('files', nargs=-1)
def helios(files,
           save,
           no_show,
           no_crop,
           ref_frame,
           output_size,
           flipv,
           fliph,
           no_transversallium,
           continuum_offset):
    """
    Processes spectroheliograph scans into narrowband still images.

    FILES must be one or more SER raw video files.
    """
    # Windows does not automatically expand wildcards, so that has to be done with the glob module.
    # Note that technically Windows allows square brackets in filenames, so this solution
    # is not perfect. They will be interpreted as wildcard characters in this case.
    if os.name == 'nt':
        files = [glob.glob(f) if any(c in f for c in ['*', '?', '[', ']']) else f for f in files]
        files = list(flatten(files))
    for f in files:
        start_time = time.time()
        click.echo(f'Started processing: {click.format_filename(f)}')
        result, (xc, yc), diameter = process_video(f, ref_frame, not no_transversallium, continuum_offset)
        if not no_crop:
            result = crop_image(result, output_size, xc, yc, diameter)
        click.echo(f'Result image size: {result.shape[0]}x{result.shape[1]} pixels')
        if flipv:
            result = np.flipud(result)
        if fliph:
            result = np.fliplr(result)
        end_time = time.time()
        click.echo(f'Processed file in {end_time - start_time:.2f} seconds')
        if save:
            new_file_path = pathlib.Path(f).with_suffix('.png')
            click.echo(f'Saving result to: {click.format_filename(str(new_file_path))}')
            imageio.imwrite(new_file_path, result.astype(np.uint16))
        if not no_show:
            show_image(result)
        click.echo('')


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


if __name__ == '__main__':
    helios()
