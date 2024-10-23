import shutil
import os
import glob
from rich.progress import track
import argparse
import re
import queue
import natsort
from typing import Generator

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


def sync_files(args: argparse.Namespace, file_mapping: dict) -> None:
    """
    Sync the output folder with the input folder.
    Remove files that have been upscaled but not in the input folder anymore.
    Remove other files and folders.

    Args:
        args (argparse.Namespace): The arguments
        file_mapping (dict): The file mapping
    """
    # Remove files that have been upscaled but not in the input folder anymore
    for f in track(file_mapping.keys(), description="Removing upscaled files...", transient=True):
        if not os.path.exists(f):
            print(f"Removing (comic) {file_mapping[f]}")
            try:
                os.remove(file_mapping[f])
                file_mapping.pop(f)
            except:
                print(f"    <<< Error removing {file_mapping[f]} >>>")

    wanted_outputs = {x.lower() for x in file_mapping.values()}

    # Remove other files and folders
    for root, dirs, files in track(os.walk(args.output, topdown=False), description="Removing other files...", transient=True):
        for name in files:
            file = os.path.join(root, name)
            if file.lower() not in wanted_outputs:
                print(f"Removing (other) {file}")
                try:
                    os.remove(file)
                except:
                    print(f"    <<< Error removing {file} >>>")


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

