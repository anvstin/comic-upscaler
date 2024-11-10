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
from upscaling.containers import FileInterface, ImageContainer, IoData
from . import UpscalerData
import io
import cv2
import torchvision
from torchvision.transforms import InterpolationMode

log = logging.getLogger(__file__)
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
def upscale_imgs(upscaler_data: UpscalerData, img_list: Iterator[tuple[str, Tensor]], device: torch.device | str) -> Iterator[tuple[str, Tensor]]:
    model = upscaler_data.load_model().cuda().half().eval()
    with torch.no_grad():
        for path, img in img_list:
            log.info(f"Upscaling {path}")

            res= path, model(img)
            del img
            log.info(f"Done upscaling")
            yield res

def upscale_container(upscaler_data: UpscalerData, data: ImageContainer, outputInterface: FileInterface):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    log.info(f"Using device {device}")
    img_dtype = torch.float32
    img_dtype = torch.half
    tensor_iterator = (
        (path, torch.tensor(d / 255.0).permute(2, 0, 1)[None].to(img_dtype).to(device))
        for path, d in data.iterate_images()
        )

    for path, img in upscale_imgs(upscaler_data, tensor_iterator, device=device):
        # Save the image to file
        output_path = Path("./") / Path(path).name
        log.info(f"Squeezing image ({img.shape})")
        img_np: np.ndarray = measure(lambda: (img.squeeze(0).clip(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())
        del img
        torch.cuda.empty_cache()
        resized = cv2.resize(img_np, dsize=(img_np.shape[0] //2, img_np.shape[1] //2), interpolation=cv2.INTER_LANCZOS4)
        log.info(f"Converting image output to PNG {path} ({resized.shape})")
        res = cv2.imencode(".png", resized)[1].tobytes()
        del img_np
        # Image.fromarray((np.clip(img_np, 0, 1) * 255).astype("uint8")).save("buf.tmp", format="png")
        log.info(f"Done converting")
        log.info(f"Saving image to {output_path}")
        # with open("buf.tmp", "rb") as buf:
        # thread = multiprocessing.Process(target=outputInterface.add_file, args=(res, output_path))
        # threads.append(thread)
        # thread.start()

        outputInterface.add_file(res, output_path)
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
