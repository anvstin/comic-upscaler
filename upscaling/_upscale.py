import logging
import multiprocessing
from pathlib import Path
import threading
from threading import Thread
import time
from typing import Callable, Iterator
import concurrent.futures
from torch import Tensor
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

from upscaling.containers import FileInterface, ImageContainer, IoData
from . import UpscaleData, UpscaleConfig
import io
import cv2
import torchvision
from torchvision.transforms import InterpolationMode

from .upscaler_config import ModelDtypes

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
            log.info(f"Done upscaling")
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
        output_path = Path("./") / Path(path).name
        log.info(f"Squeezing image ({img.shape})")
        img_np: np.ndarray = measure(
            lambda: (img.squeeze(0).clip(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())
        del img
        torch.cuda.empty_cache()
        resized = cv2.resize(img_np, dsize=(img_np.shape[0] // 2, img_np.shape[1] // 2),
                             interpolation=cv2.INTER_LANCZOS4)
        log.info(f"Converting image output to PNG {path} ({resized.shape})")
        res = cv2.imencode(".png", resized)[1].tobytes()
        del img_np
        # Image.fromarray((np.clip(img_np, 0, 1) * 255).astype("uint8")).save("buf.tmp", format="png")
        log.info(f"Done converting")
        log.info(f"Saving image to {output_path}")

        output_interface.add_file(res, output_path)
        del res
        log.info(f"Done saving")

        log.info(f"Average: {np.average(times)}")


times = []


def measure(func: Callable):
    start = time.perf_counter()
    res = func()
    end = time.perf_counter()
    total = end - start
    times.append(total)
    log.info(f"func done in {total} ms")
    return res
