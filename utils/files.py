import concurrent
import logging
import multiprocessing
import shutil
import os
import glob
import threading
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from os import DirEntry
from natsort import natsorted
from rich.progress import track, Progress
import argparse
import re
import queue
import natsort
from typing import Generator, Iterator, Iterable, Type

log = logging.getLogger(__name__)

def get_size(path: str) -> int:
    """
    Get the size of a path

    Args:
        path (str): The path to the folder

    Returns:
        int: The size of the folder
    """
    size = 0
    for file in glob.glob(path + '/**', recursive=True):
        if os.path.isdir(file):
            continue
        size += os.path.getsize(file)
    return size


def rm_tree(path: str) -> None:
    """
    Remove a directory and all its contents

    Args:
        path (str): The path to the directory to remove
    """
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

def prune_empty_folders(path: str) -> None:
    """
    Remove all empty folders in the given path

    Args:
        path (str): The path to the directory to prune
    """
    for root, dirs, files in track(os.walk(path, topdown=False), description="Removing empty folders...", transient=True):
        for name in dirs:
            folder = os.path.join(root, name)
            try:
                if len(os.listdir(folder)) == 0:
                    print(f"Removing empty folder {folder}")
                    os.rmdir(folder)
            except OSError as e:
                print("Error: %s : %s" % (folder, e.strerror))

def get_closest_dir(input_path: str) -> str:
    """
    Get the directory of the input path.
    If it is a file, get the directory of the file, otherwise return the directory itself.

    Args:
        input_path (str): The input path

    Returns:
        str: The directory of the input path
    """
    input_directory = os.path.abspath(input_path)
    if not os.path.isdir(input_path):
        input_directory = os.path.dirname(input_directory)
    return input_directory

def _sync_file_mapping(file_mapping: dict, progress: Progress) -> None:
    # Remove files that have been upscaled but not in the input folder anymore
    for f in progress.track(file_mapping.keys(), description="Removing upscaled files..."):
        if not os.path.exists(f):
            print(f"Removing (comic) {file_mapping[f]}")
            try:
                os.remove(file_mapping[f])
                file_mapping.pop(f)
            except Exception as e:
                log.debug(f"_sync_mapping: got exception {e}")
                print(f"    <<< Error removing {file_mapping[f]} >>>")

def _remove_non_mapping_files(file_mapping: dict, output_dir: str, progress: Progress) -> None:
    wanted_outputs = {x.lower() for x in file_mapping.values()}

    # Remove other files and folders
    for root, dirs, files in progress.track(os.walk(output_dir, topdown=False), description="Removing other files..."):
        for name in files:
            file = os.path.join(root, name)
            if file.lower() not in wanted_outputs:
                print(f"Removing (other) {file}")
                try:
                    os.remove(file)
                except Exception as e:
                    log.debug(f"_sync_mapping: got exception {e}")
                    print(f"    <<< Error removing {file} >>>")

def sync_files(args: argparse.Namespace, file_mapping: dict) -> None:
    """
    Sync the output folder with the input folder.
    Remove files that have been upscaled but not in the input folder anymore.
    Remove other files and folders.

    Args:
        args (argparse.Namespace): The arguments
        file_mapping (dict): The file mapping
    """
    p = Progress(transient=True)
    log.info("Syncing files...")
    _sync_file_mapping(file_mapping, p)
    _remove_non_mapping_files(file_mapping, args.output, p)



def get_sorted_comic_files(input_path: str, extension: tuple=('cbz', 'zip')) -> Generator[str, None, None]:
    """
    Get all the comic files in the input path recursively and sort them

    Args:
        input_path (str): The path to the folder or file
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').

    Yields:
        Generator[str, None, None]: A generator of comic files (cbz or zip)
    """
    for files in dir_by_dir_get_sorted_comic_files(input_path, extension):
        yield from files

def get_sorted_comic_files_parallel(input_path: str, extension: tuple=('cbz', 'zip'),
                                    executor_class: Type[Executor]=ProcessPoolExecutor) -> Iterator[str]:
    """
    Get all the comic files in the input path recursively and sort them. Use a parallel process to reduce IO waiting times.

    Args:
        input_path (str): The path to the folder or file
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').
        executor_class (Type[Executor], optional): The executor class to use. Defaults to ProcessPoolExecutor.

    Yields:
        Generator[str, None, None]: A generator of comic files (cbz or zip)
    """
    for files in dir_by_dir_get_sorted_comic_files_parallel(input_path, extension, executor_class):
        yield from files

def dir_by_dir_get_sorted_comic_files_parallel(input_path: str, extension: tuple = ('cbz', 'zip'), executor_class: Type[Executor] = ProcessPoolExecutor) -> Iterator[list[str]]:
    for files in dir_by_dir_parallel_walk(input_path, executor_class):
        filtered = list(x for x in files if any(x.endswith(ext) for ext in extension))
        if len(filtered) > 0:
            yield filtered

def dir_by_dir_get_sorted_comic_files(input_path: str, extension: tuple = ('cbz', 'zip')) -> Iterator[list[str]]:
    for files in dir_by_dir_walk(input_path):
        filtered = list(x for x in files if any(x.endswith(ext) for ext in extension))
        if len(filtered) > 0:
            yield filtered

def dir_by_dir_walk(input_path: str) -> Iterable[list[str]]:
    to_process = [input_path]
    while len(to_process) > 0:
        current_paths, to_process = to_process, []
        for files, dirs in map(ls_dir, current_paths):
            to_process.extend(dirs)
            yield files

def parallel_walk(input_path: str) -> Iterator[list[str]]:
    for files in dir_by_dir_parallel_walk(input_path):
        yield from files


def dir_by_dir_parallel_walk(input_path: str, executor_class: Type[Executor] = ThreadPoolExecutor) -> Iterator[list[str]]:
    to_process = [input_path]
    with executor_class() as executor:
        while len(to_process) > 0:
            log.debug(f"parallel_walk: len(to_process): {len(to_process)}")
            current_dirs, to_process = to_process, []
            for sub_files, sub_dirs in executor.map(ls_dir, to_process):
                to_process.extend(sub_dirs)
                yield sub_files


def ls_dir(current_path: str) -> tuple[list[str], list[str]]:
    scanned: list[DirEntry[str]] = list(os.scandir(current_path))

    # sub_dirs = natsorted((x for x in scanned if x.is_dir()), key=lambda x: x.name)
    sub_dirs = list(e.path for e in natsorted((x for x in scanned if x.is_dir()), key=lambda x: x.name))
    sub_files = natsorted((x.path for x in scanned if x.is_file()))

    return sub_files, sub_dirs

