import argparse
import glob
import logging
import os
import shutil
from concurrent.futures import Executor
from concurrent.futures.thread import ThreadPoolExecutor
from os import DirEntry
from typing import Iterator, Iterable, Type

from natsort import natsorted
from rich.progress import track, Progress

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
    for root, dirs, files in track(os.walk(path, topdown=False), description="Removing empty folders...",
                                   transient=True):
        for name in dirs:
            folder = os.path.join(root, name)
            try:
                if len(os.listdir(folder)) == 0:
                    log.info(f"Removing empty folder {folder}")
                    os.rmdir(folder)
            except OSError as e:
                log.error(f"prune_empty_folders_parallel: error while removing empty folder {folder}: {e}")


def prune_empty_folders_parallel(path: str) -> None:
    with ThreadPoolExecutor() as executor:
        for root, dirs, files in track(os.walk(path, topdown=False), description="Removing empty folders...",
                                       transient=True):
            to_process = (os.path.join(root, name) for name in dirs)
            for folder, content in zip(to_process, executor.map(os.scandir, to_process)):
                if next(content) is not None:
                    continue
                log.info(f"Removing empty folder {folder}")
                try:
                    os.rmdir(folder)
                except OSError as e:
                    log.error(f"prune_empty_folders_parallel: error while removing empty folder {folder}: {e}")


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
            except FileNotFoundError as e:
                log.exception(f"could not remove {file_mapping[f]}: {e}", stacklevel=logging.WARNING)
                file_mapping.pop(f)
            except OSError as e:
                log.exception(f"Error while removing (comic) {file_mapping[f]}: {e}")


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


def sync_files_parallel(args: argparse.Namespace, file_mapping: dict) -> None:
    with ThreadPoolExecutor() as executor:
        for file, exists in track(
                zip(file_mapping.keys(), executor.map(os.path.exists, file_mapping.keys())),
                total=len(file_mapping.keys()),
                description="Removing upscaled files...",
                transient=True
        ):
            if exists:
                continue
            log.info(f"Removing (comic) {file}")
            try:
                os.remove(file_mapping[file])
                file_mapping.pop(file)
            except Exception as e:
                log.debug(f"_sync_mapping: got exception {e}")
                log.error(f"    <<< Error removing {file_mapping[file]} >>>")

    wanted_outputs = {x.lower() for x in file_mapping.values()}
    for file in track(parallel_walk(args.output), description="Removing other files...", transient=True):
        if file.lower() not in wanted_outputs:
            log.info(f"Removing (other) {file}")
            try:
                os.remove(file)
            except Exception as e:
                log.debug(f"_sync_mapping: got exception {e}")
                log.error(f"    <<< Error removing {file} >>>")

    log.info("Synced files")


def dir_by_dir_walk(input_path: str) -> Iterable[list[str]]:
    to_process = [input_path]
    while len(to_process) > 0:
        current_paths, to_process = to_process, []
        for files, dirs in map(ls_dir, current_paths):
            to_process.extend(dirs)
            yield files


def parallel_walk(input_path: str) -> Iterator[str]:
    for files in dir_by_dir_parallel_walk(input_path):
        yield from files


def dir_by_dir_parallel_walk(input_path: str, executor_class: Type[Executor] = ThreadPoolExecutor) -> Iterator[
    list[str]]:
    to_process = [input_path]
    with executor_class() as executor:
        while len(to_process) > 0:
            log.debug(f"parallel_walk: len(to_process): {len(to_process)}")
            current_dirs, to_process = to_process, []
            for sub_files, sub_dirs in executor.map(ls_dir, to_process):
                to_process.extend(sub_dirs)
                yield sub_files


def ls_dir(input_path: str) -> tuple[list[str], list[str]]:
    scanned: list[DirEntry[str]] = list(os.scandir(input_path))

    # sub_dirs = natsorted((x for x in scanned if x.is_dir()), key=lambda x: x.name)
    sub_dirs = list(e.path for e in natsorted((x for x in scanned if x.is_dir()), key=lambda x: x.name))
    sub_files = natsorted((x.path for x in scanned if x.is_file()))

    return sub_files, sub_dirs
