import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import IO, BinaryIO, Iterator
from zipfile import ZipFile

import cv2
import natsort
import numpy as np

from utils.files import rm_tree

log = logging.getLogger(__file__)


class Extensions:
    ZIP = {".zip", ".cbz", ".cbr"}
    IMG = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class IoData:
    class Types(Enum):
        IMAGE = "image"
        OTHER = "other"

    type: Types
    filepath: str
    io: IO[bytes]


class ImageContainer:
    def __init__(self, container_path: Path):

        self.container_path = container_path

        if container_path.is_dir():
            self.interface = DirInterface(container_path)
        elif container_path.suffix in Extensions.ZIP:
            self.interface = ZipInterface(container_path)
        else:
            self.interface = FileInterface(container_path)

    def open(self):
        log.debug(f"opening file {self.container_path} with container {self.interface}")
        self.interface.open()

    def close(self):
        log.debug(f"closing file {self.container_path} with container {self.interface}")
        self.interface.close()

    def iterate_images(self, split_width: int = 0, split_height: int = 0) -> Iterator[tuple[str, np.ndarray]]:
        self.interface.open()
        for data in self.interface.iterate():
            log.debug(f"found subfile {data.filepath}")
            if data.type == IoData.Types.IMAGE:
                log.debug(f"Reading image {data.filepath}")
                byte_data = np.asarray(bytearray(data.io.read()), dtype=np.uint8)
                log.debug(f"Done reading")
                decoded = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
                yield data.filepath, decoded
                del byte_data, decoded
        self.interface.close()

    def iterate_images_async(self, split_width: int = 0, split_height: int = 0, cache_count: int = 4) -> Iterator[
        tuple[str, np.ndarray]]:
        values = list(self.interface.iterate())
        threads: list[Future | None] = [None] * len(values)

        if cache_count < 0:
            raise ValueError(f"cache_count < 0: {cache_count}")

        def read_val(data: IoData) -> tuple:
            if data.type == IoData.Types.IMAGE:
                byte_data = np.asarray(bytearray(data.io.read()), dtype=np.uint8)
                decoded = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
                return data, decoded
            return data, None

        self.interface.open()
        with ThreadPoolExecutor() as executor:
            for i in range(cache_count):
                threads[i] = executor.submit(lambda: read_val(values[i]))

            for i in range(len(values)):
                if i + cache_count < len(values):
                    threads[i + cache_count] = executor.submit(lambda: read_val(values[i + cache_count]))
                data, decoded = threads[i].result()  # type: ignore
                if data.type == IoData.Types.IMAGE:
                    log.debug(f"Reading image {data.filepath}")
                    log.debug(f"Done reading")
                    yield data.filepath, decoded
                    del decoded
        self.interface.close()

    def iterate_non_images(self) -> Iterator[IoData]:
        self.interface.open()
        for data in self.interface.iterate():
            if data.type != IoData.Types.IMAGE:
                yield data
        self.interface.close()


class FileInterface:
    ALREADY_OPENED_MSG = "Interface has already been opened, close it beforehand"
    ALREADY_CLOSED_MSG = "Interface is already closed, open it beforehand"

    def __init__(self, file: Path, write: bool = False) -> None:
        self.file = file
        self.write = write
        self._io = None

    def open(self):
        if self._io is not None:
            raise RuntimeError(self.ALREADY_OPENED_MSG)

        params = "wb" if self.write else "rb"
        self._io = open(self.file, params)
        return self._io

    def close(self):
        if self._io is None:
            raise RuntimeError(self.ALREADY_CLOSED_MSG)
        self._io.close()
        self._io = None

    def get_write(self):
        if not self.write:
            raise RuntimeError("Interface is not in write mode")
        return self._get()

    def get_read(self):
        if self.write:
            raise RuntimeError("Interface is not in read mode")
        return self._get()

    def _get(self):
        if self._io is not None:
            return self._io
        raise RuntimeError("Interface has not been opened")

    def add_file(self, data: bytes, relative_path: str | Path = "./") -> None:
        if not self.write:
            raise RuntimeError("Iterator is not in write mode")
        if relative_path != "./":
            raise ValueError("Unsupported relative path in file only mode")
        f = self.get_write()
        f.write(data)

    @staticmethod
    def one_file_iterator(file: str | Path, io: BinaryIO) -> Iterator[IoData]:
        file_path = Path(file)
        file_str = file_path.as_posix()

        if file_path.suffix in Extensions.IMG:
            yield IoData(IoData.Types.IMAGE, file_str, io)
        else:
            yield IoData(IoData.Types.OTHER, file_str, io)

    def iterate(self) -> Iterator[IoData]:
        if self.write:
            raise RuntimeError("Iterator is not in read mode")
        return self.one_file_iterator(self.file, self.get_read())


class DirInterface(FileInterface):
    def __init__(self, file: Path, write: bool = False) -> None:
        super().__init__(file, write)
        self._sub_io_list: list[BinaryIO] = []
        self.threads = []

    @staticmethod
    def write_file(path: Path, data: bytes):
        with open(path, "wb") as f:
            f.write(data)

    def add_file(self, data: bytes, relative_path: str | Path = "./") -> None:
        log.debug(f"Adding file: {relative_path} to {self.file}")
        path = self.file / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.threads.append(threading.Thread(target=self.write_file, args=(path.as_posix(), data)))
        self.threads[-1].start()

    def iterate(self) -> Iterator[IoData]:
        for file in self.file.glob("**"):
            if file.is_file():
                relative_path = file.relative_to(self.file)
                self._sub_io_list.append(open(relative_path, "rb"))
                yield next(self.one_file_iterator(relative_path, self._sub_io_list[-1]))

    def open(self):
        if self.file.is_file():
            raise IOError(f"file is not a dir: {self.file}")
        if self.write:
            rm_tree(str(self.file))
        self.file.mkdir(parents=True, exist_ok=True)

    def close(self):
        for io in self._sub_io_list:
            io.close()
        self._sub_io_list = []


class ZipInterface(FileInterface):
    def __init__(self, file: Path, write: bool = False) -> None:
        super().__init__(file, write)
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)

    def open(self):
        log.debug(f"Opening {self.file}")
        if self._io is not None:
            raise RuntimeError(self.ALREADY_OPENED_MSG)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self._io = ZipFile(self.file, "w" if self.write else "r")
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def close(self):
        log.debug(f"Closing {self.file}")
        if self._io is None:
            raise RuntimeError(self.ALREADY_CLOSED_MSG)
        self.queue.empty()
        self.queue.put(None)
        self.thread.join()

        self._io.close()
        self._io = None

    def _process_queue(self):
        while (item := self.queue.get()) is not None:
            data, relative_path = item
            log.debug(f"Adding file: {relative_path} to {self.file}")
            path = Path(relative_path)
            zf: ZipFile = self.get_write()  # type: ignore

            parent = path.parent.as_posix()
            if path.parent != Path("/") and (not parent in zf.namelist() or not zf.getinfo(parent).is_dir()):
                zf.mkdir(parent)
            zf.writestr(path.as_posix(), data)

    def add_file(self, data: bytes, relative_path: str | Path = "./") -> None:
        self.queue.put((data, relative_path))

    def iterate(self) -> Iterator[IoData]:
        zf: ZipFile = self.get_read()  # type: ignore
        for name in natsort.natsorted(zf.namelist()):
            if Path(name).suffix in Extensions.IMG:
                yield IoData(IoData.Types.IMAGE, name, zf.open(name))
            else:
                yield IoData(IoData.Types.OTHER, name, zf.open(name))
