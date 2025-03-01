import pytest
import numpy as np
import cv2
import random
from upscaler.upscaling import containers

ZIP_COUNT=3

def create_image(path: str, x: int, y: int, ext: str) -> np.ndarray:
    # Create an image with random values
    shape=(x,y,3)
    data = np.ndarray(shape=shape, dtype=np.uint8)
    data[:] = np.random.randint(0, 255, size=shape)

    cv2.imencode(ext, data)[1].tofile(path)
    return data


def test_compression():
    for i in range(ZIP_COUNT):
        pass


def test_upscale():
    pass
