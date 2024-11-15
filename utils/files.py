import logging
import multiprocessing
import shutil
import os
import glob
import threading

from rich.progress import track, Progress
import argparse
import re
import queue
import natsort
from typing import Generator

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
    threads = [
        threading.Thread(target=_sync_file_mapping, args=(file_mapping, p)),
        threading.Thread(target=_remove_non_mapping_files, args=(file_mapping, args.output, p)),
    ]

    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        thread.join()


def get_sorted_comic_files(input_path: str, ignore_upscaled: bool=False, extension: tuple=('cbz', 'zip')) -> Generator[str, None, None]:
    """
    Get all the comic files in the input path recursively and sort them

    Args:
        input_path (str): The path to the folder or file
        ignore_upscaled (bool, optional): Ignore upscaled comics. Defaults to False.
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').

    Yields:
        Generator[str, None, None]: A generator of comic files (cbz or zip)
    """
    to_process = queue.LifoQueue()
    to_process.put(input_path)
    while not to_process.empty():
        current_path = to_process.get()

        if os.path.isdir(current_path):
            # Add all files in the folder to the queue (reverse them to be processed in order)
            for file in natsort.natsorted(os.listdir(current_path), reverse=True):
                to_process.put(os.path.join(current_path, file))
        else:
            if ignore_upscaled and re.match(r'.*_x[0-9]+\.cbz', current_path):
                print(f"Ignoring {current_path}")
                continue

            if any(current_path.endswith(ext) for ext in extension):
                yield current_path

def get_sorted_comic_files_parallel_process(input_path: str, ignore_upscaled: bool=False, extension: tuple=('cbz', 'zip'), cache_size=20) -> Generator[str, None, None]:
    """
    Get all the comic files in the input path recursively and sort them. Use a parallel process to reduce IO waiting times.

    Args:
        input_path (str): The path to the folder or file
        ignore_upscaled (bool, optional): Ignore upscaled comics. Defaults to False.
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').

    Yields:
        Generator[str, None, None]: A generator of comic files (cbz or zip)
    """
    files = multiprocessing.Queue(cache_size)

    process = multiprocessing.Process(target=_find_cbz, args=(input_path, ignore_upscaled, extension, files))
    process.start()

    while (val := files.get()) is not None:
        logging.debug(f"qsize (get_sorted_comic_files_parallel_process): {files.qsize()}")
        yield val

    process.join()
    if process.exitcode != 0:
        raise RuntimeError("Error finding comic files")

def get_sorted_comic_files_parallel_thread(input_path: str, ignore_upscaled: bool=False, extension: tuple=('cbz', 'zip'), cache_size=20) -> Generator[str, None, None]:
    """
    Get all the comic files in the input path recursively and sort them. Use a parallel thread to reduce IO waiting times.

    Args:
        input_path (str): The path to the folder or file
        ignore_upscaled (bool, optional): Ignore upscaled comics. Defaults to False.
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').

    Yields:
        Generator[str, None, None]: A generator of comic files (cbz or zip)
    """
    files = multiprocessing.Queue(cache_size)

    thread = threading.Thread(target=_find_cbz, args=(input_path, ignore_upscaled, extension, files))
    thread.start()

    while (val := files.get()) is not None:
        yield val

    thread.join()


def _find_cbz(input_path: str, ignore_upscaled: bool, extension: tuple, files: multiprocessing.Queue):
    for e in get_sorted_comic_files(input_path, ignore_upscaled, extension): files.put(e)
    files.put(None)

