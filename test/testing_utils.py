from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import cv2
import numpy as np


class ImageExtensions(Enum):
    PNG = ".png"
    WEBP = ".webp"
    JPG = ".jpg"
    JPEG = ".jpeg"


class CompressedExtensions(Enum):
    CBZ = ".cbz"
    ZIP = ".zip"

def generate_image(x: int, y: int) -> np.ndarray:
    # Create an image with random values
    shape = (x, y, 3)
    data = np.ndarray(shape=shape, dtype=np.uint8)
    data[:] = np.random.randint(0, 255, size=shape)

    return data


def encode_image(data: np.ndarray, ext: str):
    return cv2.imencode(ext, data)[1]

def save_image(data: np.ndarray, path: str | Path, ext: str):
    encode_image(data, ext).tofile(path)


def read_image(path: str | Path):
    return cv2.imread(path)



def create_random_compressed_comic(path: str | PathLike, img_count: int, img_size: (int, int), img_ext: ImageExtensions,
                             check: bool = True):
    path = Path(path)
    assert not check or path.suffix in CompressedExtensions
    with ZipFile(path, "w") as zf:
        for i in range(img_count):
            zf.writestr(f"{i}{img_ext.value}", encode_image(generate_image(*img_size), img_ext.value))


def create_library(path: str | PathLike, dir_count: int, create_comic_func: Callable[[PathLike], None],
                   dir_comic_count: int, comic_ext: CompressedExtensions):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for i in range(dir_count):
        sub_path = path / f"test_{i}"
        sub_path.mkdir()

        for j in range(dir_comic_count):
            create_comic_func(sub_path / f"comic_{j}{comic_ext.value}")
