from upscaling.containers import FileInterface
from . import UpscalerData, ImageContainer
from pathlib import Path
import multiprocessing as mp

def upscale_func(upscaler_data: UpscalerData, data: ImageContainer, outputInterface: FileInterface):
    from ._upscale import upscale_container
    data.open()
    outputInterface.open()
    upscale_container(upscaler_data, data, outputInterface)
    data.close()
    outputInterface.close()

def upscale_file(upscaler_data: UpscalerData, data: ImageContainer, outputInterface: FileInterface):
    # Upscale the image in another process
    upscale_func(upscaler_data, data, outputInterface)
    # p = mp.Process(target=upscale_func, args=(upscaler_data, data, outputInterface))
    # p.start()
    # p.join()



