import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

import test
from test.testing_utils import create_library, create_random_compressed_comic, ImageExtensions, \
    CompressedExtensions

DIR_COUNT = 2
DIR_COMIC_COUNT = 1
IMG_COUNT = 2
IMG_SIZE = (30, 50)


@pytest.fixture(scope="session")
def temp_dir():
    temp_dir = Path(__file__).parent / "temp"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir


def test_library(temp_dir):
    date = datetime.fromtimestamp(time.time()).strftime("%d_%m_%Y-%H_%M_%S")

    comic_func = lambda path: create_random_compressed_comic(path, IMG_COUNT, IMG_SIZE, ImageExtensions.WEBP)

    input_dir = temp_dir / f"{date}" / "input"
    output_dir = temp_dir / f"{date}" / "output"

    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    create_library(input_dir, DIR_COUNT, comic_func, DIR_COMIC_COUNT, CompressedExtensions.CBZ)
    res = subprocess.run(
        args=[sys.executable, "-m", "upscaler", input_dir.as_posix(), output_dir.as_posix(), "--suffix", "", "--stop-on-failures"],
        cwd=Path(test.__file__).parent.parent,
        shell=True,
        timeout=30
    )

    input_files = sorted(file.relative_to(input_dir) for file in input_dir.glob("**"))
    output_files = sorted(file.relative_to(output_dir) for file in output_dir.glob("**"))

    assert input_files == output_files
