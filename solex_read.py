import glob
import os
import pathlib

import click
import imageio
import numpy as np

from solex.correction import geometric_correction
from solex.linefitting import fit_poly_to_dark_line, get_absorption_line
from solex.ser_reader import SerFile
from solex.utils import show_image, crop_image


def process_video(filename: str, ref_frame_index: int = None) -> np.ndarray:
    input_file = SerFile(filename)
    input_file.print_headers()

    # Pick reference frame from middle of video
    if ref_frame_index is None:
        ref_frame_index = int(input_file.frame_count / 2)
    click.echo(f"Using frame {ref_frame_index} as reference")
    ref_frame = input_file.read_frame(ref_frame_index)

    # Fit polynomial to dark absorption line in reference frame
    poly_curve = fit_poly_to_dark_line(ref_frame)
    output_frame = np.ndarray((input_file.frame_count, ref_frame.shape[1]))
    for i in range(0, input_file.frame_count):
        image = input_file.read_frame(i)
        output_frame[i, :] = get_absorption_line(image, poly_curve)

    final_output = geometric_correction(output_frame).T
    return final_output


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
