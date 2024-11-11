import logging
import math
import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from upscaling.containers import FileInterface, ImageContainer
from . import UpscaleData, ImageConverter

log = logging.getLogger(__file__)


def create_chunks(image: Tensor, patch_size: int = 1024, overlap: int = 100) -> Tensor:
    height, width, _ = image.shape
    step = patch_size - overlap
    padding_width = (step - (width - overlap) % step) % step
    padding_height = (step - (height - overlap) % step) % step
    image = F.pad(image, (0, 0, 0, padding_width, 0, padding_height))
    return (
        image
        .unfold(0, patch_size, step)
        .unfold(1, patch_size, step)
        .moveaxis(2, -1)
    )


def upscale_images(upscale_data: UpscaleData, img_list: Iterator[tuple[str, Tensor]]) -> \
        Iterator[tuple[str, Tensor]]:
    model = upscale_data.get_model()
    with torch.no_grad():
        for path, img in img_list:
            log.info(f"Upscaling {path}")

            res = path, model(img)
            del img
            log.debug(f"Done upscaling")
            yield res


def upscale_container(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface):
    device = torch.device(upscale_data.config.device)
    img_dtype = upscale_data.config.model_dtype.get()

    log.info(f"Using device {device} with type {img_dtype}")
    tensor_iterator = (
        (path, torch.tensor(d / 255.0).permute(2, 0, 1)[None].to(img_dtype).to(device))
        for path, d in data.iterate_images()
    )

    for path, img in upscale_images(upscale_data, tensor_iterator):
        # Save the image to file
        output_path = (Path("./") / Path(path).name).with_suffix("." + upscale_data.config.output_format)
        log.debug(f"Squeezing image ({img.shape})")
        img_np: np.ndarray = (img.squeeze(0).clip(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        del img
        torch.cuda.empty_cache()
        resized = cv2.resize(img_np, dsize=(upscale_data.config.output_max_width, int(np.round(
            img_np.shape[0] * upscale_data.config.output_max_width / img_np.shape[1]))),
                             interpolation=cv2.INTER_LANCZOS4)
        del img_np
        log.debug(f"Converting image output to {output_path.suffix} {path} ({resized.shape})")
        res = ImageConverter(resized).to(output_path.suffix)
        del resized

        log.debug(f"Done converting")
        new_output_paths = (output_path,)
        if len(res) > 1:
            digits = int(math.log10(len(res))) + 1
            new_output_paths = (
                output_path.with_suffix(f".{i:0{digits}}{output_path.suffix}")
                for i in range(len(res))
            )

        log.info(f"Saving to {tuple(i.as_posix() for i in new_output_paths)}")
        for img_data, img_path in zip(res, new_output_paths):
            log.debug(f"Saving to {img_path} (shape: {img_data.shape})")
            output_interface.add_file(img_data, img_path)
        del res
        log.debug(f"Done saving images")
