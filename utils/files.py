import shutil
import os
import glob
from rich.progress import track
import argparse
import re

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


def get_comics_files(input_path: str, ignore_upscaled: bool=False, extension: tuple=('cbz', 'zip')) -> list:
    """
    Get all the comic files in the input path recursively

    Args:
        input_path (str): The path to the folder or file
        ignore_upscaled (bool, optional): Ignore upscaled comics. Defaults to False.
        extension (tuple, optional): The extensions to look for. Defaults to ('cbz', 'zip').

    Returns:
        list: A list of comic files (cbz or zip)
    """

    res = []
    if os.path.isdir(input_path):
        for file in glob.glob(input_path + '/**', recursive=True):
            if ignore_upscaled and re.match(r'.*_x[0-9]+\.cbz', file):
                print(f"Ignoring {file}")
                continue

            if any(file.endswith(ext) for ext in extension):
                res.append(file) # glob already returns the full path
    elif any(input_path.endswith(ext) for ext in extension):
        res.append(input_path)

    return res

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
            print(f"Removing (comic) {f}")
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
