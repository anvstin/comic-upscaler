import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Generator

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
import gc

from . import UpscaleData, ImageConverter, FileInterface, ImageContainer

log = logging.getLogger(__file__)

MAX_INPUT_SIZE = 10240 # 8192 + 2048

def create_chunks(image: Tensor, patch_size: int = 1024, overlap: int = 100) -> Tensor:
    height, width, _ = image.shape
    step = patch_size - overlap
    padding_width = (step - (width - overlap) % step) % step
    padding_height = (step - (height - overlap) % step) % step
    image = functional.pad(image, (0, 0, 0, padding_width, 0, padding_height))
    return (
        image
        .unfold(0, patch_size, step)
        .unfold(1, patch_size, step)
        .moveaxis(2, -1)
    )


def upscale_images(upscale_data: UpscaleData, img_list: Iterator[tuple[Path, Tensor]], stop_on_failures: bool) -> \
        Iterator[tuple[Path, Tensor]]:
    model = upscale_data.get_model()
    with torch.no_grad():
        for path, img in img_list:
            log.info(f"Upscaling {path}")
            log.debug(f"Input shape: {img.shape}")
            try:
                res_img = model(img)
            except Exception as e:
                log.error(f"Failed to upscale {path}: {e}")
                if stop_on_failures:
                    raise e
                yield path, img
            log.debug(f"Done upscaling")
            yield path, res_img

def get_horizontal_chunks(data: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    """

    :param data:
    :param chunk_size:
    :return:
    """
    size = math.ceil(data.shape[0] / chunk_size)
    for i in range(size):
        yield data[i * chunk_size:(i + 1) * chunk_size, :]


def get_prepared_images_for_upscale(upscale_data: UpscaleData, data: ImageContainer) -> Iterator[tuple[Path, Tensor]]:
    """

    :param upscale_data:
    :param data:
    :return:
    """
    device = torch.device(upscale_data.config.device)
    img_dtype = upscale_data.config.model_dtype.get()
    log.info(f"Using device {device} with type {img_dtype}")

    # to_tensor = lambda input_img: (torch.tensor(input_img, device=device) / 255).permute(2, 0, 1)[None].to(img_dtype)

    for path, img in data.iterate_images():
        if img.shape[1] > MAX_INPUT_SIZE:
            raise RuntimeError(f"width is too large for image of shape {img.shape}")
        if img.shape[0] > MAX_INPUT_SIZE:
            log.info(f"Splitting {path} of shape {img.shape}")
            chunks = list(get_horizontal_chunks(img, MAX_INPUT_SIZE))
            digits = int(math.log10(len(chunks))) + 1
            chunk_names = [
                (Path("/") / Path(path).name).with_suffix(f".{i:0{digits}}.{upscale_data.config.output_format}")
                           for i in range(len(chunks))
            ]
            # yield from zip(chunk_names, map(to_tensor, chunks))
            yield from zip(chunk_names, chunks)
        else:
            output_path = (Path("/") / Path(path).name).with_suffix(f".{upscale_data.config.output_format}")
            # yield output_path, to_tensor(img)
            yield output_path, img

def post_process_images(upscale_data: UpscaleData, img_list: Iterator[tuple[Path, Tensor]]) -> Iterator[tuple[Path, np.ndarray]]:
    for path, img in img_list:
        yield post_process_image(upscale_data, (path, img))

def post_process_image(upscale_data: UpscaleData, input_data: tuple[Path, Tensor]) -> tuple[Path, np.ndarray]:
    path, img = input_data
    log.debug(f"Squeezing image ({img.shape})")
    img_np: np.ndarray = (img.squeeze(0).permute(1, 2, 0).clip(0, 1) * 255).to(torch.uint8).cpu().numpy()
    del img
    torch.cuda.empty_cache()

    resized = cv2.resize(img_np, dsize=(upscale_data.config.output_max_width, int(np.round(
        img_np.shape[0] * upscale_data.config.output_max_width / img_np.shape[1]))),
                            interpolation=cv2.INTER_LANCZOS4)
    del img_np
    return path, resized

def save_images(img_list: Iterator[tuple[Path, np.ndarray]], output_interface: FileInterface):
    for output_path, resized in img_list:
        log.debug(f"Converting image output to {output_path.suffix} {output_path} ({resized.shape})")
        res = ImageConverter(resized).to(output_path.suffix)
        del resized

        log.debug(f"Done converting")
        new_output_paths = [output_path]
        if len(res) > 1:
            digits = int(math.log10(len(res))) + 1
            new_output_paths = [
                output_path.with_suffix(f".{i:0{digits}}{output_path.suffix}")
                for i in range(len(res))
            ]

        log.info(f"Saving to {tuple(i.as_posix() for i in new_output_paths)}")
        for img_data, img_path in zip(res, new_output_paths):
            log.debug(f"Saving to {img_path} (shape: {img_data.shape})")
            output_interface.add_file(img_path, img_data)
        del res
        log.debug(f"Done saving images")


def preprocess_image(upscale_data: UpscaleData, img_data: tuple[Path, np.ndarray], device: torch.device) -> tuple[Path, Tensor]:
    path, img_np = img_data
    img_dtype = upscale_data.config.model_dtype.get()
    return path, (torch.tensor(img_np, device=device) / 255).permute(2, 0, 1)[None].to(img_dtype)

def save_image(data: tuple[Path, Tensor], output_interface: FileInterface):
    output_path, resized = data
    log.debug(f"Converting image output to {output_path.suffix} {output_path} ({resized.shape})")
    res = ImageConverter(resized).to(output_path.suffix)
    del resized

    log.debug(f"Done converting")
    new_output_paths = [output_path]
    if len(res) > 1:
        digits = int(math.log10(len(res))) + 1
        new_output_paths = [
            output_path.with_suffix(f".{i:0{digits}}{output_path.suffix}")
            for i in range(len(res))
        ]

    log.info(f"Saving to {tuple(i.as_posix() for i in new_output_paths)}")
    for img_data, img_path in zip(res, new_output_paths):
        log.debug(f"Saving to {img_path} (shape: {img_data.shape})")
        output_interface.add_file(img_path, img_data)
    del res
    log.debug(f"Done saving images")

def upscale_image(upscale_data: UpscaleData, data: tuple[Path, Tensor], stop_on_failures: bool) -> tuple[Path, Tensor]:
    path, img = data
    model = upscale_data.get_model()
    log.info(f"Upscaling {path}")
    log.debug(f"Input shape: {img.shape}")
    with torch.no_grad():
        try:
            res_img = model(img)
        except Exception as e:
            log.error(f"Failed to upscale {path}: {e}")
            if stop_on_failures:
                raise e
            return path, img

    log.debug(f"Done upscaling")
    return path, res_img

def upscale_container(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface, stop_on_failures: bool):
    device = torch.device(upscale_data.config.device)
    log.info(f"Using device {device} for upscaling")

    def upscale_pipeline(img_data: tuple[Path, Tensor]):
        data = img_data
        data = preprocess_image(upscale_data, data, device)
        data = upscale_image(upscale_data, data, stop_on_failures)
        data = post_process_image(upscale_data, data)
        save_image(data, output_interface)

    input_images = get_prepared_images_for_upscale(upscale_data, data)

    if upscale_data.config.upscale_workers > 1:
        with ThreadPoolExecutor(max_workers=upscale_data.config.upscale_workers) as executor:
            executor.map(upscale_pipeline, input_images)
    else:
        map(upscale_pipeline, input_images)

def upscale_file(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface, stop_on_failures: bool):
    output_interface.open()
    try:
        upscale_container(upscale_data, data, output_interface, stop_on_failures)
        for iodata in data.iterate_non_images():
            path = Path(iodata.filepath)
            log.info(f"Copying {path.name}")
            output_interface.add_file(path, iodata.get())
    except Exception as e:
        raise e
    finally:
        output_interface.close()
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
