import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_WEBP_SIZE = 16383
MAX_PNG_SIZE = 65500
MAX_JPEG_SIZE = 65500


def _encode_split(extension: str, image: np.ndarray, params: list, max_size) -> list[np.ndarray]:
    nb_split = int(np.ceil(image.shape[0] / max_size))
    split_size = int(np.ceil(image.shape[0] / nb_split))
    log.debug(f"nb_split: {nb_split}, shape: {image.shape}")
    res = []
    for i in range(nb_split):
        split = image[i * split_size:(i + 1) * split_size, :, :]
        status, img = cv2.imencode(ext=extension, img=split, params=params)
        res.append(img)
        if not status:
            raise ValueError("Could not encode image")

    return res


class ImageConverter:
    def __init__(self, image: np.ndarray):
        self.images: list[np.ndarray] = [image]

    def add_image(self, image):
        pass

    def to(self, extension: str, **kwargs) -> np.ndarray:
        stripped_extension = extension.lstrip(".").lower()
        save_map = {
            "png": self.to_png,
            "jpeg": self.to_jpeg,
            "jpg": self.to_jpeg,
            "webp": self.to_webp,
            "gif": self.to_gif,
        }
        return save_map[stripped_extension](**kwargs)

    def to_webp(self, quality: int = 90) -> list[np.ndarray]:
        if len(self.images) == 0:
            raise ValueError("No images to convert")
        if len(self.images) > 1:
            log.warning(f"The current WEBP implementation only supports one image for now, using first image")
        return _encode_split(".webp", self.images[0], [cv2.IMWRITE_WEBP_QUALITY, quality], MAX_WEBP_SIZE)

    def to_png(self, compression: int = 9) -> list[np.ndarray]:
        return self._encode_single_image_container("png", [cv2.IMWRITE_PNG_COMPRESSION, compression], MAX_PNG_SIZE)

    def to_jpeg(self, quality: int = 90) -> list[np.ndarray]:
        return self._encode_single_image_container("jpg", [cv2.IMWRITE_JPEG_QUALITY, quality], MAX_JPEG_SIZE)

    def to_gif(self, quality: int = 90) -> list[np.ndarray]:
        if len(self.images) == 0:
            raise ValueError("No images to convert")
        if len(self.images) > 1:
            log.warning(f"The current GIF implementation only supports one image for now, using first image")
        return _encode_split(".webp", self.images[0], [cv2.IMWRITE_WEBP_QUALITY, quality], MAX_WEBP_SIZE)

    def _encode_single_image_container(self, container: str, params: list, max_size: int) -> list[np.ndarray]:
        if len(self.images) == 0:
            raise ValueError("No images to convert")
        if len(self.images) != 1:
            raise ValueError(f"The {container.upper()} container only supports one image")
        return _encode_split(f".{container.lower()}", self.images[0], params, max_size)
