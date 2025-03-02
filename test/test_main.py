import importlib.util
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from test.testing_utils import create_library, create_random_compressed_comic, ImageExtensions, \
    CompressedExtensions

DIR_COUNT = 3
DIR_COMIC_COUNT = 1
IMG_COUNT = 2
IMG_SIZE = (800, 600)


@pytest.fixture(scope="session")
def temp_dir():
    temp_dir = Path(__file__).parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    shutil.rmtree(temp_dir)


def test_library(temp_dir):
    date = datetime.fromtimestamp(time.time()).strftime("%d_%m_%Y-%H_%M_%S")

    comic_func = lambda path: create_random_compressed_comic(path, IMG_COUNT, IMG_SIZE, ImageExtensions.WEBP)

    input_dir = temp_dir / f"{date}" / "input"
    output_dir = temp_dir / f"{date}" / "output"

    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    create_library(input_dir, DIR_COUNT, comic_func, DIR_COMIC_COUNT, CompressedExtensions.CBZ)
    import upscaler.__main__
    upscaler.__main__.main(["test", str(input_dir), str(output_dir)], continue_on_failure=False)
    # ret = subprocess.run([
    #     sys.executable,
    #     Path(importlib.util.find_spec("upscaler").origin).parent.parent / "main.py",
    #     input_dir,
    #     output_dir,
    # ], capture_output=True)
    # print(ret.stdout)
    # print(ret.stderr, file=sys.stderr)
    #
    # ret.check_returncode()

    input_files = sorted(file.relative_to(input_dir) for file in input_dir.glob("**"))
    output_files = sorted(file.relative_to(output_dir) for file in output_dir.glob("**"))

    assert input_files == output_files
