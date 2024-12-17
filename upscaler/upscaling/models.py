import logging
import os
from pathlib import Path

from upscaling import UpscaleConfig
from upscaling.upscaler_config import ModelDtypes

log = logging.getLogger(__name__)


class UpscaleData:
    log = logging.getLogger("UpscalerData")

    def __init__(self, name: str, path: str, scale: int, download_path: str | None = None, config=UpscaleConfig()):
        self.name = name
        self.path = path
        self.scale = scale
        self.config = config
        self.model = None

        file_extension = Path(self.path).suffix
        assert file_extension == ".pth" or file_extension == ".safetensor", f"{self.path} must be a .pth or .safetensor file"

        self.real_path = Path(self.path)
        if self.is_url():
            self.real_path = Path(
                download_path) if download_path else Path.cwd() / "models" / f"{self.name}-x{self.scale}{file_extension}"

    def __str__(self):
        return f"{self.name} x{self.scale} ({self.path})"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_url(self) -> bool:
        return self.path.startswith("https://") or self.path.startswith("http://")

    def fetch_model(self) -> Path:
        self.log.debug("fetch_model {self}")
        if self.is_url():
            return self.download_model()
        if not self.real_path.is_file():
            raise FileNotFoundError(f"{self.name} x{self.scale} model not found at {self.real_path}")

        return Path(self.path)

    def load_model(self):
        self.log.debug(f"load_model {self.name}")

        if not os.path.isfile(self.real_path):
            raise RuntimeError(f"Could not find model {self.name} at {self.real_path}")
        self.log.debug(f"load_model {self.name}")
        from spandrel import ImageModelDescriptor, ModelLoader

        model = ModelLoader().load_from_file(self.real_path)

        assert isinstance(model, ImageModelDescriptor), f"{model} is not an ImageModelDescriptor"
        assert model.scale == self.scale, f"{model} has a scale of {model.scale}, but expected {self.scale}"

        if self.config.model_dtype == ModelDtypes.HALF:
            model = model.half()
        self.model = model.eval().to(self.config.device)

    def unload_model(self):
        self.log.debug(f"unload_model {self.name}")
        del self.model
        self.model = None

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model

    def download_model(self) -> Path:
        self.log.debug(f"download_model {self.name}")
        from tqdm import tqdm
        import requests

        # Create download directory if it does not exist
        os.makedirs(self.real_path.parent, exist_ok=True)

        # Download model
        with requests.get(self.path, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            file_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            if self.real_path.is_file() and self.real_path.stat().st_size == file_size:
                self.log.debug(f"model already downloaded at {self.real_path} with size {file_size}")
                return self.real_path

            with tqdm(total=file_size, unit='iB', unit_scale=True) as pbar, open(self.real_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        pbar.update(len(chunk))
                        f.write(chunk)
                    else:
                        f.flush()

        self.log.debug("model downloaded at {self.real_path} with size {file_size}")
        return self.real_path

    def configure(self, upscale_config: UpscaleConfig) -> None:
        self.config = upscale_config


def get_realesrgan_model_from_name(model_name: str) -> UpscaleData:
    models = (
        UpscaleData(
            name="R-ESRGAN General 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4,
        ),
        UpscaleData(
            name="R-ESRGAN General WDN 4xV3",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            scale=4,
        ),
        UpscaleData(
            name="R-ESRGAN AnimeVideo",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            scale=4,
        ),
        UpscaleData(
            name="R-ESRGAN 4x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4,
        ),
        UpscaleData(
            name="R-ESRGAN 4x+ Anime6B",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4,
        ),
        UpscaleData(
            name="R-ESRGAN 2x+",
            path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            scale=2,
        ),
    )

    compatible_models = [model for model in models if model.name == model_name]
    if not compatible_models:
        raise ValueError(f"{model_name} is not a valid RealESRGAN model")
    if len(compatible_models) > 1:
        raise RuntimeError(f"{model_name} is not a unique RealESRGAN model")

    return compatible_models[0]


def get_realesrgan_model(upscale_config: UpscaleConfig) -> UpscaleData:
    data = get_realesrgan_model_from_name(upscale_config.model_name)
    data.configure(upscale_config)
    return data
