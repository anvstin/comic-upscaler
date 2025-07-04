from dataclasses import dataclass
from enum import Enum


class ModelDtypes(Enum):
    HALF = 'half'
    FLOAT = 'float'

    def get(self):
        import torch
        types = {
            'half': torch.half,
            'float': torch.float
        }
        return types[self.value]


@dataclass
class UpscaleConfig:
    device: str = 'cpu'

    model_name: str = "R-ESRGAN AnimeVideo"
    model_dtype: ModelDtypes = ModelDtypes.FLOAT

    output_format: str = "webp"
    output_max_width: int = 1600
    output_max_height: int = 0
    output_preserve_upscale_ratio: bool = False

    upscale_workers: int = 4

    # chunk_enabled: bool = False
    # chunk_size: int = 4000
    # chunk_padding: int = 100
