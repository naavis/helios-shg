import mmap

import click
import numpy as np


class SerFile:
    def __init__(self, file_path):
        self._file_handle = open(file_path, 'rb')
        self._file_mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        self.header = self._read_header()
        self.width = self.header['Width']
        self.height = self.header['Height']
        self.frame_count = self.header['FrameCount']
        self.bytes_per_pixel = self._get_bytes_per_pixel()
        self._little_endian = self.header['LittleEndian'] == 0

    def __del__(self):
        self._file_mmap.close()
        self._file_handle.close()

    def read_frame(self, frame_index):
        bytes_per_frame = self.width * self.height * self.bytes_per_pixel
        offset = 178 + frame_index * bytes_per_frame
        self._file_mmap.seek(offset)

        data_type = np \
            .dtype('uint16' if self.bytes_per_pixel == 2 else 'uint8') \
            .newbyteorder('<' if self._little_endian else '>')
        raw_bytes = self._file_mmap.read(bytes_per_frame)
        image_1d = np.frombuffer(raw_bytes, data_type)
        return image_1d.reshape((self.height, self.width))

    def print_headers(self):
        click.echo('SER file headers:')
        for header_key in self.header:
            click.echo(f'\t{header_key}: {self.header[header_key]}')

    def _read_header(self):
        header = {}
        offset = 0
        self._file_mmap.seek(offset)

        header['FileId'] = self._file_mmap.read(14).decode('utf-8')
        if header['FileId'].strip() != 'LUCAM-RECORDER':
            raise RuntimeError('Input file is not a valid SER file')
        offset += 14

        self._file_mmap.seek(offset)
        header['LuId'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['ColorId'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['LittleEndian'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['Width'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['Height'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['PixelDepthPerPlane'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        self._file_mmap.seek(offset)
        header['FrameCount'] = int.from_bytes(self._file_mmap.read(4), 'little', signed=False)
        offset += 4

        return header

    def _get_bytes_per_pixel(self):
        if self.header['ColorId'] <= 19:
            number_of_planes = 1
        else:
            number_of_planes = 3

        if self.header['PixelDepthPerPlane'] <= 8:
            bytes_per_pixel = number_of_planes
        else:
            bytes_per_pixel = 2 * number_of_planes

        return bytes_per_pixel
