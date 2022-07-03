import click
import numpy as np

from solex.correction import geometric_correction, extract_transversallium, transversallium_correction
from solex.linefitting import fit_poly_to_dark_line, get_absorption_line
from solex.ser_reader import SerFile


def process_video(filename: str, ref_frame_index: int, apply_transversallium_correction: bool, continuum_offset: int) -> np.ndarray:
    input_file = SerFile(filename)
    input_file.print_headers()

    # Pick reference frame from middle of video
    if ref_frame_index is None:
        ref_frame_index = int(input_file.frame_count / 2)
    elif ref_frame_index < 0 or ref_frame_index >= input_file.frame_count:
        raise RuntimeError('Invalid reference frame index supplied')
    click.echo(f"Using frame {ref_frame_index} as reference")
    ref_frame = input_file.read_frame(ref_frame_index)

    # Fit polynomial to dark absorption line in reference frame
    poly_curve = fit_poly_to_dark_line(ref_frame)
    output_frame = np.ndarray((input_file.frame_count, ref_frame.shape[1]))
    continuum_frame = np.zeros_like(output_frame)
    for i in range(0, input_file.frame_count):
        image = input_file.read_frame(i)
        output_frame[i, :] = get_absorption_line(image, poly_curve)
        if apply_transversallium_correction:
            continuum_frame[i, :] = get_absorption_line(image, continuum_offset + poly_curve)

    if apply_transversallium_correction:
        transversallium = extract_transversallium(continuum_frame)
        output_frame = transversallium_correction(output_frame, transversallium)
    final_output = geometric_correction(output_frame).T
    return final_output
