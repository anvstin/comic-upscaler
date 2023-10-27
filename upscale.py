import argparse
import glob
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import multiprocessing
import threading # For IO
import shutil
import time

import rich
import rich.progress
from rich.progress import track, Progress
# from rich import print
from rich.console import Console

import natsort
from paths import OutputPathGenerator, PathGenerator
from global_config import *

console = Console()

# if __name__ == '__main__':
#     # Call subprocess to ask for the version of python, opencv and torch (avoid loading them in the main process to avoid high memory usage)
#     p = [
#         subprocess.Popen([sys.executable, "--version"], stdout=subprocess.PIPE),
#         subprocess.Popen([sys.executable, "-c", "import cv2; print(f'cv2={cv2.__version__}')"], stdout=subprocess.PIPE),
#         subprocess.Popen([sys.executable, "-c", "import torch; print(f'torch={torch.__version__}')"], stdout=subprocess.PIPE)
#     ]
#     for process in p:
#         process.wait()
#         print(process.stdout.read().decode('utf-8').strip('\n'))



# pyjion
# if __name__ == '__main__': print(f"Importing pyjion...")
# import pyjion
# pyjion.config(level=1)
# if __name__ == '__main__': print(f"Pyjion: {pyjion.enable()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file or folder')
    parser.add_argument('output', help='Output file or folder', default=None, nargs='?')
    # parser.add_argument('-d', '--divide', type=int, default=2, help='Size to divide the images by')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as a daemon')
    parser.add_argument('-f', '--format', default="jpg", help='Output images format')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Scale the images by')
    parser.add_argument('-w', '--width', type=int, default=0, help='Fit the images to the width')
    parser.add_argument('-t', '--tiles', type=int, default=1024, help='Split the images into tiles')
    parser.add_argument('--tile_pad', type=int, default=50, help='Pad the tiles by')
    parser.add_argument("--remove_root_folders", type=int, default=0, help="Remove the number of folders from the root of the output path. Eg with a value of 2 : ./a/b/test.cbz -> ./test.cbz")
    parser.add_argument('--sync', action='store_true', help='Synchronize the input and output folders')
    parser.add_argument("--fp32", action='store_true', help="Use fp32 instead of fp16")
    # Compress the images after upscaling
    parser.add_argument('-c', '--compress', action='store_true', help='Compress the images after upscaling')
    parser.add_argument("--suffix", default="_upscaled", help="Suffix to add to the output file name", nargs='?')
    parser.add_argument("--rename", action='store_true', help="Rename the output file for kavita")
    parsed_args = parser.parse_args()

    parsed_args.input = parsed_args.input.replace('\\', '/')
    parsed_args.input = os.path.abspath(parsed_args.input)
    parsed_args.input = os.path.normpath(parsed_args.input)

    if parsed_args.suffix is None:
        parsed_args.suffix = ''

    if parsed_args.output is None:
        # Output folder for the compressed files
        if os.path.isdir(parsed_args.input):
            parsed_args.output = parsed_args.input
        else:
            parsed_args.output = os.path.dirname(parsed_args.input)

    parsed_args.output = parsed_args.output.replace('\\', '/')
    parsed_args.output = os.path.abspath(parsed_args.output)
    parsed_args.output = os.path.normpath(parsed_args.output)

    return parsed_args

def get_cbz_files(input_path, ignore_upscaled=False):
    res = []
    if os.path.isdir(input_path):
        for file in glob.glob(input_path + '/**', recursive=True):
            if ignore_upscaled and re.match(r'.*_x[0-9]+\.cbz', file):
                print(f"Ignoring {file}")
                continue

            if file.endswith('.cbz') or file.endswith('.zip'):
                # res.append(input_path + '/' + file)
                res.append(file) # glob already returns the full path
    elif input_path.endswith('.cbz') or input_path.endswith('.zip'):
        res.append(input_path)

    return res

def extract(file, output_path=None, progress=Progress(transient=True)):
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

def divide(input_folder, output_folder, divide, format=None):
    if divide <= 1:
        return

    # Divide the images by 2 to reduce the size
    dirs = os.listdir(input_folder)
    size = len(dirs)

    print(f"  0/{size}")
    for i, image in enumerate(dirs):
        if os.path.isdir(image):
            continue
        if image.endswith(f'.{format}'):
            img = Image.open(input_folder + '/' + image)
            img = img.resize((int(img.width / divide), int(img.height / divide)), Image.LANCZOS)
            img.save(output_folder + '/' + image, format=format)
            print(f"  {i + 1}/{size} {image}", end='\r')

        # Replace print

def fit_to_width(input_folder, width, format=None):
    if width <= 0:
        return
    # Walk each image and resize it to fit the width
    paths = list(enumerate(glob.glob(input_folder + '/**', recursive=True)))
    size = len(paths)
    for i, image in paths:
        if  format is not None and  not image.endswith(f'.{format}'):
            continue
        if os.path.isdir(image):
            continue

        img = Image.open(image)
        # Using CV2 to open the image
        # img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img.width < width:
            continue

        ratio = width / img.width
        if ratio >= 1:
            continue
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
        # Using CV2 to resize the image
        # img = cv2.resize(img, (width, int(img.height * ratio)), interpolation=cv2.INTER_LANCZOS4)

        # Save the image
        img.save(image, format=format)
        # Using CV2 to save the image
        # cv2.imencode(re.search(r'\.([a-zA-Z]+)$', image).group(1), img)[1].tofile(image)
        print(f"  {i + 1}/{size} {os.path.basename(image)}", end='\r')


def compress_seven_zip(folder, output_path, progress=Progress(transient=True)):
    # Compress the folder to a cbz file
    if not os.path.exists(SEVEN_ZIP_PATH):
        raise Exception(f"7z.exe not found at {SEVEN_ZIP_PATH}")

    task_id = progress.add_task(f"    Compressing {folder}...", total=None)
    ret = subprocess.run([SEVEN_ZIP_PATH, "-tzip", 'a', output_path, folder + '/*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    progress.remove_task(task_id)
    if ret.returncode != 0:
        raise Exception(f"Failed to compress {folder} to {output_path}")

def get_size(folder):
    size = 0
    for file in glob.glob(folder + '/**', recursive=True):
        if os.path.isdir(file):
            continue
        size += os.path.getsize(file)
    return size

def compress_integrated(folder, output_path, progress=Progress(transient=True)):
    """
    Compress the folder to a cbz file using standard python libraries

    Args:
        folder (str): The folder to compress
        output_path (str): The path to the output file
        progress (Progress): The progress bar to use
    """

    # Compress the folder to a cbz file using standard python libraries
    # Integraded version seems to have trouble with network drives, use 7zip in those cases
    task_id = progress.add_task(f"    Compressing {folder}...", total=get_size(folder))

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for file in glob.glob(folder + '/**', recursive=True):
            if os.path.isdir(file):
                continue
            zip_ref.write(file, os.path.relpath(file, folder))
            progress.update(task_id, advance=os.path.getsize(file))

    progress.remove_task(task_id)

def compress(folder, output_path):
    """
    Compress the folder to a cbz file and select the best method

    Args:
        folder (str): The folder to compress
        output_path (str): The path to the output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(SEVEN_ZIP_PATH):
        compress_seven_zip(folder, output_path)
    else:
        compress_integrated(folder, output_path)

def upscale(input_path, output_path, scale, format=None, width=0, tiles=1024, wait=True, tile_pad=50, fp32=False):
    args = ["python", UPSCALE_SCRIPT_PATH, '-i', input_path, '-o', output_path, '-n', MODEL_NAME, '-s', f"{scale}", "-t", f"{tiles}", "--tile_pad", f"{tile_pad}", "--width", f"{width}"]
    # args = ["python", executable_path, '-i', input_path, '-o', output_path, '-n', model_name, '-s', str(scale), "-t", f"{tiles}", "--tile_pad", "10", "--width", str(width), "--fp32"]
    if fp32:
        args += ["--fp32"]
    if format is not None:
        args += ['--ext', format]

    # print("Running:", args[0], *[f"'{arg}'" for arg in args[1:] ])
    # Environment variables
    ret = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    if wait:
        ret.wait()

    # end_after_upscaling = (False)
    # while ret.poll() is None:
    #     # Check if the user has pressed q to quit after the upscaling
    #     if keyboard.is_pressed('shift') and keyboard.is_pressed('q') and keyboard.is_pressed('ctrl') and not end_after_upscaling:
    #         print("<<< User pressed ctrl + q, no more files will be processed after the current one >>>")
    #         end_after_upscaling = True
    #     time.sleep(0.1)

    if wait and ret.returncode != 0:
        raise Exception("Upscaling failed")

    # Check length of output folder to see if it the same as the input folder
    input_length = len(glob.glob(input_path + '/**', recursive=True))
    output_length = len(glob.glob(output_path + '/**', recursive=True))
    if input_length > output_length:
        print(f"ERROR: Input length ({input_length}) superior to output length ({output_length})")
        raise Exception("Upscaling failed. Missing images")
    elif input_length < output_length:
        print(f"WARNING: Input length ({input_length}) inferior to output length ({output_length}). Could be due to an image being split into multiple images.")

    return ret

def upscale_parallel(input_path, output_path, scale, format=None, width=0, tiles=1024, thread_count=2):
    # Get all the files in the input path
    files = glob.glob(input_path + '/**', recursive=True)
    files = [f for f in files if not os.path.isdir(f)]
    if format is not None:
        files = [f for f in files if f.endswith(f'.{format}')]

    # Split the files into chunks
    subprocesses = []
    while len(files) > 0:
        if len(subprocesses) >= thread_count:
            # Wait for a thread to finish
            for t in subprocesses:
                if t.poll() is not None:
                    subprocesses.remove(t)
                    break
            time.sleep(0.1)
        else:
            f = files[0]
            print(f"Upscaling {f} to {f.replace(input_path, output_path)}")
            # threads.append(threading.Thread(target=upscale, args=(f, output_path, scale, format, width, tiles)))
            subprocesses.append(upscale(f, output_path, scale, format, width, tiles, wait=False, tile_pad=args.tile_pad, fp32=args.fp32))
            # upscale(f, output_path, scale, format, width, tiles, wait=True)
            files.pop(0)


def rm_tree(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

def normalize_path(path):
    return os.path.abspath(os.path.normpath(path))

def sync_files(args, file_mapping: dict):
    # Remove files that have been upscaled but not in the input folder anymore
    for f in track(file_mapping.keys(), description="Removing upscaled files...", transient=True):
        if not os.path.exists(f):
            print(f"Removing (comic) {f}")
            os.remove(file_mapping[f])
            file_mapping.pop(f)

    wanted_outputs = {x.lower() for x in file_mapping.values()}

    # Remove other files and folders
    for root, dirs, files in track(os.walk(args.output, topdown=False), description="Removing other files...", transient=True):
        for name in files:
            file = os.path.join(root, name)
            if file.lower() not in wanted_outputs:
                print(f"Removing (other) {file}")
                os.remove(file)


def prune_empty_folders(path):
    # threads = []
    for root, dirs, files in track(os.walk(path, topdown=False), description="Removing empty folders...", transient=True):
        for name in dirs:
            folder = os.path.join(root, name)
            try:
                if len(os.listdir(folder)) == 0:
                    print(f"Removing empty folder {folder}")
                    os.rmdir(folder)
                    # threads.append(threading.Thread(target=os.rmdir, args=(folder,)))
            except OSError as e:
                print("Error: %s : %s" % (folder, e.strerror))

    # for t in threads:
    #     t.start()

    # # Wait for all threads to finish
    # for t in track(threads, description="Removing empty folders...", transient=True):
    #     t.join()


def get_input_dir(input_path):
    input_directory = os.path.abspath(input_path)
    if not os.path.isdir(input_path):
        input_directory = os.path.dirname(input_directory)
    return input_directory


def main(args, params, file_mapping: dict):
    seen_files = set(file_mapping.keys())

    # Get the list of .cbz files
    found_paths = get_cbz_files(args.input, ignore_upscaled=True)

    # Sort the files by name
    found_paths = natsort.natsorted(found_paths)

    paths = [p for p in found_paths if p not in seen_files]
    if len(paths) == 0:
        return 0
    print(f"Found {len(paths)} files")
    input_directory = get_input_dir(args.input)

    # Process each .cbz file
    with Progress() as progress:
        progress_bar = progress.add_task("Processing files...", total=len(paths))
        for i, file in enumerate(paths):
            progress.update(progress_bar, advance=1)
            if params.end_after_upscaling:
                print("<<< Exiting... >>>")
                exit()
            if file in seen_files:
                continue

            if not os.path.exists(file):
                print(f"Skipping {os.path.relpath(file, args.input)} (does not exist)")
                paths[i] = None
                continue

            gen = OutputPathGenerator.from_args(args, file)
            if gen.exists():
                print(f"Skipping {os.path.relpath(file, input_directory)} (already exists)")
                paths[i] = None
                file_mapping[file] = gen.output_path
                continue
            print(f"Processing {os.path.relpath(file, args.input)} ({i + 1}/{len(paths)})")
            print(f"    Output: {os.path.relpath(gen.output_path, args.output)}")

            print(f"    Extracting to {os.path.basename(gen.extract_path)}")
            os.makedirs(gen.extract_path, exist_ok=True)
            try:
                extract(file, gen.extract_path, progress=progress)
            except:
                print(f"    <<< Error extracting {os.path.basename(file)} >>>")
                continue

            print(f"    Upscaling to {os.path.basename(gen.upscale_path)} ({len(os.listdir(gen.extract_path))} images)")
            os.makedirs(gen.upscale_path, exist_ok=True)
            # Hide progress bar temporarily
            progress.update(progress_bar, visible=False)
            try:
                upscale(gen.extract_path, gen.upscale_path, args.scale, args.format, width=args.width, tiles=args.tiles, fp32=args.fp32)
                progress.update(progress_bar, visible=False)
                rm_tree(gen.extract_path)
            except:
                print(f"    <<< Error upscaling {os.path.basename(file)} >>>")
                continue

            # print(f"    Fitting images to width", args.width)
            # fit_to_width(upscale_path, args.width)

            print(f"    Writing final images to {gen.output_path}")
            try:
                if args.compress:
                    compress(gen.upscale_path, gen.output_path)
                    rm_tree(gen.upscale_path)
                else:
                    os.rename(gen.upscale_path, gen.output_path)
            except:
                print(f"    <<< Error writing {os.path.basename(file)} >>>")
                continue

            file_mapping[file] = gen.output_path
            print(f"Done {os.path.basename(file)} ({i + 1}/{len(paths)})")
        progress.stop()

    # if all paths are in seen_files, return 0
    if all([path in seen_files for path in paths if path is not None]):
        return 0

    return len([path for path in paths if path is not None])

# def print_sleep(duration: int, step: int = 5):
#     for i in range(0, duration, step):
#         print(f"\033[KSleeping for {duration - i} seconds... ", end='\r')
#         time.sleep(step)
#     print("\033[K", end='\r')

def print_sleep(duration: int, step: int = 0.5):
    # Use rich to print sleep
    with Progress(speed_estimate_period=2, transient=True) as progress:
        task = progress.add_task("Sleeping...", total=duration)
        nb_iter = int(duration / step)
        for i in range(nb_iter):
            progress.update(task, advance=step)
            time.sleep(step)
        time.sleep(duration % step)
        progress.stop()


def start_processing(args, params):
    file_mapping: dict = params.file_mapping
    count = -100
    while True:
        count = -1
        while count != 0:
            print("\033[K", end='\r')
            count = main(args, params, file_mapping)

        if args.sync:
            print("Syncing...", end='\r')
            sync_files(args, file_mapping)
            params.need_pruning = True

        if params.need_pruning:
            prune_empty_folders(args.output)
            params.need_pruning = False

        print(f"Done {count} files")
        if count == 0:
            if not args.daemon:
                print("No more files to process")
                break
            print_sleep(60)
        else:
            print("\033[KChecking for new files...", end='\r')

def terminal(args, params):
    sys.stdin = open(0)
    while True:
        cmd = input()
        if cmd == "sync":
            sync_files(args, params.file_mapping)
            params.need_pruning = (True)
        elif cmd == "q":
            params.end_after_upscaling = (True)
            print("<<< User pressed q, no more files will be processed after the current one >>>")
        else:
            print("Unknown command")

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    params = manager.Namespace()
    params.end_after_upscaling = False
    params.need_pruning = False
    params.file_mapping = manager.dict()

    args = parse_args()

    # Print all the arguments
    print("Arguments:")
    argparse_dict = vars(args)
    for key in argparse_dict:
        print(f"  {key}: {argparse_dict[key]}")
    print()

    # Start the processing
    p = multiprocessing.Process(target=start_processing, args=(args,params))
    p.start()

    # Start a terminal to run commands
    p2 = multiprocessing.Process(target=terminal, args=(args,params))
    p2.start()

    # Wait for the processes to finish
    while True:
        if not p.is_alive():
            break
        if not p2.is_alive():
            break
        time.sleep(1)

    # Kill the processes
    p.terminate()
    p2.terminate()






