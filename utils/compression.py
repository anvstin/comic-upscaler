import os
import subprocess
import glob
import zipfile
from rich.progress import Progress
from utils.files import get_size, rm_tree
from utils.paths import get_program_path
from global_config import SEVEN_ZIP_PATH

def compress(folder: str, output_path: str, progress: Progress|None=None) -> None:
    """
    Compress the folder to a cbz file and select the best method

    Args:
        folder (str): The folder to compress
        output_path (str): The path to the output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(SEVEN_ZIP_PATH):
        seven_zip_path = SEVEN_ZIP_PATH
    else:
        seven_zip_path = get_program_path("7z")

    if seven_zip_path is None:
        compress_integrated(folder, output_path, progress)
    else:
        compress_seven_zip(folder, output_path, seven_zip_path, progress)


def compress_integrated(folder: str, output_path: str, progress: Progress|None=None) -> None:
    """
    Compress the folder to a cbz file using standard python libraries

    Args:
        folder (str): The folder to compress
        output_path (str): The path to the output file
        progress (Progress): The progress bar to use
    """
    # Compress the folder to a cbz file using standard python libraries
    # Integraded version seems to have trouble with network drives, use 7zip in those cases
    if progress:
        task_id = progress.add_task(f"    Compressing {folder}...", total=get_size(folder))

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for file in glob.glob(folder + '/**', recursive=True):
            if os.path.isdir(file):
                continue
            zip_ref.write(file, os.path.relpath(file, folder))
            if progress:
                progress.update(task_id, advance=os.path.getsize(file))
    if progress:
        progress.remove_task(task_id)

def compress_seven_zip(folder: str, output_path: str, seven_zip_path: str, progress: Progress|None=None) -> None:
    """
    Compress a folder to a cbz file using 7z.exe

    Args:
        folder (str): The path to the folder to compress
        output_path (str): The path to the output cbz file
        progress (Progress, optional): The progress bar to use. Defaults to Progress(transient=True)
    """
    if not os.path.exists(seven_zip_path):
        raise Exception("7z.exe not found. Please install 7zip and add it to your PATH")
    if progress:
        task_id = progress.add_task(f"    Compressing {folder}...", total=None)
    ret = subprocess.run([seven_zip_path, "-tzip", 'a', output_path, folder + '/*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if progress:
        progress.remove_task(task_id)
    if ret.returncode != 0:
        raise Exception(f"Failed to compress {folder} to {output_path}")


def extract(file: str, output_path: str|None = None, progress=Progress(transient=True)) -> str:
    """
    Extract a zip file to a folder

    Args:
        file (str): The path to the zip file
        output_path (str, optional): The path to the output folder. Defaults to None.
        progress (Progress, optional): The progress bar to use. Defaults to Progress(transient=True).

    Returns:
        str: The path to the output folder
    """
    # Create a folder with the same name as the file
    folder_name = file[:-4] if output_path is None else output_path
    folder_name = os.path.abspath(folder_name)
    folder_name = os.path.normpath(folder_name)
    if os.path.exists(folder_name):
        # Empty the folder
        rm_tree(folder_name)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # Extract the files to the folder
    # task_id = progress.add_task(f"Extracting {file}...", total=os.path.getsize(file))
    # add tabulation before the message to align it with the other messages
    task_id = progress.add_task(f"    Extracting {file}...", total=os.path.getsize(file))
    p = progress.open(file, "rb", task_id=task_id)
    with zipfile.ZipFile(p) as zip_ref:
        zip_ref.extractall(folder_name)
    p.close()
    progress.remove_task(task_id)
    # Move the files to the root of the folder
    for file in glob.glob(folder_name + '/*/**', recursive=True):
        if os.path.isfile(file):
            os.rename(file, folder_name + '/' + os.path.basename(file))

    # Remove the empty folders
    for folder in glob.glob(folder_name + '/*/**', recursive=True):
        if os.path.isdir(folder):
            os.rmdir(folder)

    return folder_name
