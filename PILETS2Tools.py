import numpy as np
import rawutil
from PIL import Image, ImageFile


class ETS2DepthDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        min_val, max_val = self.args
        denorm = pow(2, 24) - 1
        format = f"<{(self.state.xsize * self.state.ysize)}U"

        self.fd.seek(26)  # skip header size
        data = np.array(rawutil.unpack(format, self.fd.read()))
        data = ((data / denorm) - min_val) / (max_val - min_val)
        data = np.uint8((1 - data) * 255)
        self.set_as_raw(data)
        return 0, 0


Image.register_decoder('ets2-depth', ETS2DepthDecoder)

"""Image format"""


def _accept(header):
    return header[:2] == b"DP"


class ETS2DepthImageFile(ImageFile.ImageFile):
    """ PIL custom image format for ETS Dataset Depth"""
    format = "ETS2D"
    format_description = "ETS2 dataset depth"

    def _open(self):
        self.magic = bytes(rawutil.unpack('bb', self.fp.read(2))).decode('utf-8')
        self.file_size, = rawutil.unpack('<l', self.fp.read(4))
        width, height = rawutil.unpack('<ll', self.fp.read(8))
        self._size = (width, height)
        self.min_val, self.max_val, self.offset = rawutil.unpack('<ffl', self.fp.read(12))
        self.mode = 'L'

        # data descriptor
        self.tile = [("ets2-depth", (0, 0) + self.size, 26, (self.min_val, self.max_val))]


Image.register_open(ETS2DepthImageFile.format, ETS2DepthImageFile, _accept)
Image.register_extensions(ETS2DepthImageFile.format, ['.depth.raw'])

if __name__ == "__main__":
    im = Image.open('test-data/capture-0000001673.depth.raw')
    im.convert('RGBA').save('test-pil.png')