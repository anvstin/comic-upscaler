import logging
from pathlib import Path

from upscaling.containers import FileInterface
from . import UpscaleData, ImageContainer

log = logging.getLogger(__name__)

def upscale_file(upscale_data: UpscaleData, data: ImageContainer, output_interface: FileInterface):
    from ._upscale import upscale_container
    output_interface.open()
    upscale_container(upscale_data, data, output_interface)
    for iodata in data.iterate_non_images():
        path = Path(iodata.filepath)
        log.info(f"Copying {path.name}")
        output_interface.add_file(iodata.io.read(), path)
    output_interface.close()
