import pytest
import tempfile
from pathlib import Path
from upscaler.__main__ import parse_args


def test_parse_args_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir])
        assert args.input == str(Path(tmpdir).resolve())
        assert args.output == str(Path(tmpdir).resolve())


def test_parse_args_with_suffix():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--suffix", "_test"])
        assert args.suffix == "_test"


def test_parse_args_no_suffix():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--suffix"])
        assert args.suffix == ""


def test_parse_args_daemon():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "-d"])
        assert args.daemon is True


def test_parse_args_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "-f", "png"])
        assert args.format == "png"


def test_parse_args_width():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "-w", "1024"])
        assert args.width == 1024


def test_parse_args_remove_root_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--remove_root_folders", "2"])
        assert args.remove_root_folders == 2


def test_parse_args_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--sync"])
        assert args.sync is True


def test_parse_args_fp32():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--fp32"])
        assert args.fp32 is True


def test_parse_args_rename():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--rename"])
        assert args.rename is True


def test_parse_args_stop_on_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--stop-on-failures"])
        assert args.stop_on_failures is True


def test_parse_args_max_workers():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir, "--max_workers", "4"])
        assert args.max_workers == 4


def test_parse_args_invalid_width():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit):
            parse_args(["prog", tmpdir, tmpdir, "-w", "-1"])


def test_parse_args_compress_always_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(["prog", tmpdir, tmpdir])
        assert args.compress is True
