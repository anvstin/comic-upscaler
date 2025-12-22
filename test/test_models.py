import tempfile
from pathlib import Path
import pytest
from upscaler.upscaling.models import UpscaleData, get_realesrgan_model_from_name, get_realesrgan_model
from upscaler.upscaling.upscaler_config import UpscaleConfig, ModelDtypes


def test_upscale_data_init():
    data = UpscaleData(
        name="test",
        path="test.pth",
        scale=4
    )
    assert data.name == "test"
    assert data.scale == 4
    assert not data.is_url()


def test_upscale_data_url():
    data = UpscaleData(
        name="test",
        path="https://example.com/model.pth",
        scale=4
    )
    assert data.is_url()


def test_upscale_data_str():
    data = UpscaleData(
        name="test",
        path="test.pth",
        scale=4
    )
    assert "test" in str(data)
    assert "4" in str(data)


def test_upscale_data_hash():
    data1 = UpscaleData(name="test", path="test.pth", scale=4)
    data2 = UpscaleData(name="test", path="test.pth", scale=4)
    assert hash(data1) == hash(data2)


def test_upscale_data_invalid_extension():
    with pytest.raises(AssertionError):
        UpscaleData(name="test", path="test.txt", scale=4)


def test_upscale_data_configure():
    data = UpscaleData(name="test", path="test.pth", scale=4)
    config = UpscaleConfig(device="cpu")
    data.configure(config)
    assert data.config.device == "cpu"


def test_get_realesrgan_model_from_name():
    data = get_realesrgan_model_from_name("R-ESRGAN AnimeVideo")
    assert data.name == "R-ESRGAN AnimeVideo"
    assert data.scale == 4


def test_get_realesrgan_model_from_name_invalid():
    with pytest.raises(ValueError):
        get_realesrgan_model_from_name("Invalid Model")


def test_get_realesrgan_model():
    config = UpscaleConfig(
        model_name="R-ESRGAN AnimeVideo",
        device="cpu"
    )
    data = get_realesrgan_model(config)
    assert data.name == "R-ESRGAN AnimeVideo"
    assert data.config.device == "cpu"


def test_upscale_data_fetch_model_not_found():
    data = UpscaleData(
        name="test",
        path="nonexistent.pth",
        scale=4
    )
    with pytest.raises(FileNotFoundError):
        data.fetch_model()


def test_upscale_data_real_path_url():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = UpscaleData(
            name="test",
            path="https://example.com/model.pth",
            scale=4,
            download_path=tmpdir
        )
        assert str(data.real_path) == tmpdir


def test_all_realesrgan_models():
    model_names = [
        "R-ESRGAN General 4xV3",
        "R-ESRGAN General WDN 4xV3",
        "R-ESRGAN AnimeVideo",
        "R-ESRGAN 4x+",
        "R-ESRGAN 4x+ Anime6B",
        "R-ESRGAN 2x+"
    ]
    for name in model_names:
        data = get_realesrgan_model_from_name(name)
        assert data.name == name
        assert data.is_url()
