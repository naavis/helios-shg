import numpy as np


class SerFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = self._read_header()
        self.width = self.header['Width']
        self.height = self.header['Height']
        self.frame_count = self.header['FrameCount']
        self.little_endian = self.header['LittleEndian'] == 0
        self.bytes_per_pixel = self._get_bytes_per_pixel()

    def _read_header(self):
        header = {}
        with open(self.file_path, 'rb') as f:
            offset = 0
            f.seek(offset)

            header['FileId'] = f.read(14).decode('utf-8')
            offset += 14

            f.seek(offset)
            header['LuId'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['ColorId'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['LittleEndian'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['Width'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['Height'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['PixelDepthPerPlane'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

            f.seek(offset)
            header['FrameCount'] = int.from_bytes(f.read(4), 'little', signed=False)
            offset += 4

        return header

    def read_frame(self, frame_index):
        with open(self.file_path, 'rb') as f:
            offset = 178 + frame_index * self.width * self.height * self.bytes_per_pixel
            f.seek(offset)

            data_type = np\
                .dtype('uint16' if self.bytes_per_pixel == 2 else 'uint8')\
                .newbyteorder('<' if self.little_endian else '>')
            raw_data = np.fromfile(f, dtype=data_type, count=self.width * self.height)
            return raw_data.reshape((self.height, self.width))

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
