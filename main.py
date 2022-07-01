from solex.correction import geometric_correction
from solex.ser_reader import SerFile
import sys
import numpy as np
from solex.linefitting import fit_poly_to_dark_line, get_absorption_line
from solex.utils import show_image


def process_video(filename: str) -> np.ndarray:
    input_file = SerFile(filename)
    input_file.print_headers()
    # Pick reference frame from middle of video
    ref_frame_index = int(input_file.frame_count / 2)
    print(f"Using frame {ref_frame_index} as reference")
    ref_frame = input_file.read_frame(ref_frame_index)
    # Fit 2nd degree polynomial to dark absorption line in reference frame
    poly = fit_poly_to_dark_line(ref_frame)
    _, poly_curve = poly.linspace(ref_frame.shape[1], domain=[0, ref_frame.shape[1]])
    output_frame = np.ndarray((input_file.frame_count, ref_frame.shape[1]))
    for i in range(0, input_file.frame_count):
        image = input_file.read_frame(i)
        output_frame[i, :] = get_absorption_line(image, poly_curve)

    final_output = geometric_correction(output_frame).T
    return final_output


def main(args: [str]):
    final_output = process_video(args[0])
    show_image(final_output)


if __name__ == '__main__':
    main(sys.argv[1:])
