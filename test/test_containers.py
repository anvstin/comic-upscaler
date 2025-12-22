import tempfile
from pathlib import Path
import pytest
import numpy as np
from upscaler.upscaling.containers import (
    ImageContainer, FileInterface, DirInterface, ZipInterface,
    IoData, Extensions
)
from test.testing_utils import generate_image, save_image, ImageExtensions


def test_extensions():
    assert ".cbz" in Extensions.ZIP
    assert ".png" in Extensions.IMG


def test_io_data():
    data = IoData(IoData.Types.IMAGE, "test.png", lambda: b"data")
    assert data.type == IoData.Types.IMAGE
    assert data.filepath == "test.png"
    assert data.get() == b"data"


def test_file_interface_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.png"
        img = generate_image(10, 10)
        save_image(img, test_file, ".png")
        
        interface = FileInterface(test_file, write=False)
        interface.open()
        
        items = list(interface.iterate())
        assert len(items) == 1
        assert items[0].type == IoData.Types.IMAGE
        
        interface.close()


def test_file_interface_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "output.png"
        
        interface = FileInterface(test_file, write=True)
        interface.open()
        interface.add_file("./", b"test data")
        interface.close()
        
        assert test_file.exists()


def test_dir_interface():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        
        img = generate_image(10, 10)
        save_image(img, input_dir / "test.png", ".png")
        
        interface = DirInterface(input_dir, write=False)
        interface.open()
        
        items = list(interface.iterate())
        assert len(items) >= 1
        
        interface.close()


def test_dir_interface_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        
        interface = DirInterface(output_dir, write=True)
        interface.open()
        interface.add_file("test.txt", b"test data")
        # Wait for thread to complete
        for thread in interface.threads:
            thread.join()
        interface.close()
        
        assert (output_dir / "test.txt").exists()


def test_zip_interface_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        from test.testing_utils import create_random_compressed_comic, CompressedExtensions
        
        zip_file = Path(tmpdir) / "test.cbz"
        create_random_compressed_comic(zip_file, 2, (10, 10), ImageExtensions.PNG)
        
        interface = ZipInterface(zip_file, write=False)
        interface.open()
        
        items = list(interface.iterate())
        assert len(items) == 2
        
        interface.close()


def test_zip_interface_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_file = Path(tmpdir) / "output.cbz"
        
        interface = ZipInterface(zip_file, write=True)
        interface.open()
        interface.add_file("test.txt", b"test data")
        interface.close()
        
        assert zip_file.exists()


def test_image_container_zip():
    with tempfile.TemporaryDirectory() as tmpdir:
        from test.testing_utils import create_random_compressed_comic, CompressedExtensions
        
        zip_file = Path(tmpdir) / "test.cbz"
        create_random_compressed_comic(zip_file, 2, (10, 10), ImageExtensions.PNG)
        
        container = ImageContainer(zip_file)
        assert isinstance(container.interface, ZipInterface)
        
        images = list(container.iterate_images())
        assert len(images) == 2


def test_image_container_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        
        img = generate_image(10, 10)
        save_image(img, input_dir / "test.png", ".png")
        
        container = ImageContainer(input_dir)
        assert isinstance(container.interface, DirInterface)


def test_image_container_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.png"
        img = generate_image(10, 10)
        save_image(img, test_file, ".png")
        
        container = ImageContainer(test_file)
        assert isinstance(container.interface, FileInterface)
