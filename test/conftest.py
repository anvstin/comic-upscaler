import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)
