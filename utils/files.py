import shutil
import os
import glob
from rich.progress import track

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
