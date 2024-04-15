from enum import Enum, auto

supportedExtensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".jp2", ".exr", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"}
supportedExtensions.update({ext.upper() for ext in supportedExtensions})


class NNOpt(Enum):
    FILL_DOWNSAMPLE = auto()
    FILL_UPSAMPLE = auto()
    MISSING_DATA_FLAG = auto()
    AUTO_INTERPRET_RGB_MASK = auto()
    CACHE_DISK = auto()
    CACHE_MEMORY = auto()
    
