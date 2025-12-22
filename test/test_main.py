from pathlib import Path

import pytest

from test.testing_utils import create_library, create_random_compressed_comic, ImageExtensions, \
    CompressedExtensions
from upscaler.__main__ import main

DIR_COUNT = 2
DIR_COMIC_COUNT = 1
IMG_COUNT = 2
IMG_SIZE = (30, 50)


@pytest.mark.timeout(300)
def test_library(tmp_path):
    comic_func = lambda path: create_random_compressed_comic(path, IMG_COUNT, IMG_SIZE, ImageExtensions.WEBP)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    output_dir.mkdir()

    create_library(input_dir, DIR_COUNT, comic_func, DIR_COMIC_COUNT, CompressedExtensions.CBZ)
    args = ["upscaler", input_dir.as_posix(), output_dir.as_posix(), "--suffix", "", "--stop-on-failures"]
    main(args)

    input_files = sorted(file.relative_to(input_dir) for file in input_dir.glob("**"))
    output_files = sorted(file.relative_to(output_dir) for file in output_dir.glob("**"))

    assert input_files == output_files
