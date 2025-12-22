import os
from pathlib import Path
import pytest
from PIL import Image
from upscaler.utils.files import get_size, prune_empty_folders, get_closest_dir, ls_dir
from upscaler.utils.paths import cleanup_path, get_program_path
from upscaler.utils.measure import measure_ns, nanoseconds_pretty_str, Measurement
from upscaler.utils.image import divide, fit_to_width
import numpy as np


def test_get_size(temp_directory: Path):
    tmpdir = temp_directory
    file_data = "hello"
    file_size = len(file_data.encode("utf-8"))
    file_count = 5
    for i in range(file_count):
        file = tmpdir / f"file_{i}.txt"
        file.write_text(file_data)
    assert get_size(str(tmpdir)) == file_size * file_count


def test_prune_empty_folders(temp_directory: Path):
    empty_dir = temp_directory / "empty"
    empty_dir.mkdir()
    prune_empty_folders(str(temp_directory))
    assert not empty_dir.exists()


def test_get_closest_dir(temp_directory: Path):
    test_file = temp_directory / "test.txt"
    test_file.write_text("test")
    assert get_closest_dir(str(test_file)) == str(temp_directory.resolve())
    assert get_closest_dir(str(temp_directory)) == str(temp_directory.resolve())


def test_ls_dir(temp_directory: Path):
    (temp_directory / "file.txt").write_text("test")
    (temp_directory / "subdir").mkdir()
    files, dirs = ls_dir(str(temp_directory))
    assert len(files) == 1
    assert len(dirs) == 1


def test_cleanup_path():
    path = "./test/../file.txt"
    result = cleanup_path(path)
    assert os.path.isabs(result)


def test_get_program_path():
    result = get_program_path("python")
    assert result is None or ("python" in result.lower() and os.path.exists(result))


def test_measure_ns():
    elapsed, result = measure_ns(lambda: 42)
    assert result == 42
    assert elapsed > 0


def test_nanoseconds_pretty_str():
    result = nanoseconds_pretty_str(1_500_000_000)
    assert "1h" in result or "0h" in result


def test_measurement():
    m = Measurement(lambda: 42, log_call=False, log_stats=False)
    result = m.call()
    assert result == 42
    assert len(m.measurements) == 1
    assert m.average() > 0
    assert m.median() > 0


def test_divide(temp_directory: Path):
    input_dir = temp_directory / "input"
    output_dir = temp_directory / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    img = Image.new('RGB', (100, 100), color='red')
    img.save(input_dir / "test.png", format="PNG")
    img.close()
    
    divide(str(input_dir), str(output_dir), 2, format="png")
    
    output_img = Image.open(output_dir / "test.png")
    assert output_img.width == 50
    assert output_img.height == 50
    output_img.close()


def test_fit_to_width(temp_directory: Path):
    img = Image.new('RGB', (200, 100), color='blue')
    img_path = temp_directory / "test.png"
    img.save(img_path, format="PNG")
    img.close()
    
    fit_to_width(str(temp_directory), 100, format="png")
    
    result_img = Image.open(img_path)
    assert result_img.width == 100
    assert result_img.height == 50
    result_img.close()