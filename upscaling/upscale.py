from upscaling.containers import FileInterface
from . import UpscaleData, ImageContainer
from pathlib import Path
import multiprocessing as mp

def upscale_func(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface):
    from ._upscale import upscale_container
    data.open()
    output_interface.open()
    upscale_container(upscale_data, data, output_interface)
    data.close()
    output_interface.close()

def upscale_file(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface):
    # Upscale the image in another process
    upscale_func(upscale_data, data, output_interface)
    # p = mp.Process(target=upscale_func, args=(upscaler_data, data, outputInterface))
    # p.start()
    # p.join()



