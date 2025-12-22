from pathlib import Path
import pytest
from upscaler.paths import OutputPathGenerator, OutputPathGeneratorConfig


def test_output_path_generator_basic(temp_directory: Path):
    input_file = temp_directory / "test.cbz"
    input_file.touch()
    
    config = OutputPathGeneratorConfig(
        input=str(input_file),
        output=str(temp_directory),
        suffix="_upscaled",
        compress=True,
        rename=False,
        remove_root_folders=0
    )
    
    gen = OutputPathGenerator(str(input_file), config)
    assert gen.output_path.endswith("_upscaled.cbz")
    assert not gen.exists()


def test_output_path_generator_no_suffix(temp_directory: Path):
    input_file = temp_directory / "test.cbz"
    input_file.touch()
    
    config = OutputPathGeneratorConfig(
        input=str(input_file),
        output=str(temp_directory),
        suffix="",
        compress=True,
        rename=False,
        remove_root_folders=0
    )
    
    gen = OutputPathGenerator(str(input_file), config)
    assert gen.output_path.endswith(".cbz")


def test_output_path_generator_rename(temp_directory: Path):
    subdir = temp_directory / "parent"
    subdir.mkdir()
    input_file = subdir / "test.cbz"
    input_file.touch()
    
    config = OutputPathGeneratorConfig(
        input=str(input_file),
        output=str(temp_directory),
        suffix="",
        compress=True,
        rename=True,
        remove_root_folders=0
    )
    
    gen = OutputPathGenerator(str(input_file), config)
    assert " - " in gen.output_path


def test_output_path_generator_remove_invalid_chars(temp_directory: Path):
    input_file = temp_directory / "test.cbz"
    input_file.touch()
    
    config = OutputPathGeneratorConfig(
        input=str(input_file),
        output=str(temp_directory),
        suffix="",
        compress=True,
        rename=False,
        remove_root_folders=0
    )
    
    gen = OutputPathGenerator(str(input_file), config)
    cleaned = gen.remove_invalid_characters("test  file")
    assert cleaned == "test file"


def test_output_path_generator_exists(temp_directory: Path):
    input_file = temp_directory / "test.cbz"
    input_file.touch()
    
    config = OutputPathGeneratorConfig(
        input=str(input_file),
        output=str(temp_directory),
        suffix="",
        compress=True,
        rename=False,
        remove_root_folders=0
    )
    
    gen = OutputPathGenerator(str(input_file), config)
    Path(gen.output_path).touch()
    assert gen.exists()


def test_output_path_generator_from_args():
    class Args:
        input = "/test/input"
        output = "/test/output"
        suffix = "_up"
        compress = True
        rename = False
        remove_root_folders = 0
    
    config = OutputPathGeneratorConfig.from_args(Args())
    assert config.input == "/test/input"
    assert config.suffix == "_up"


def test_output_path_generator_from_dict():
    d = {
        "input": "/test/input",
        "output": "/test/output",
        "suffix": "_up",
        "compress": True,
        "rename": False,
        "remove_root_folders": 0
    }
    config = OutputPathGeneratorConfig.from_dict(d)
    assert config.input == "/test/input"
    assert config.suffix == "_up"
