import os
import sys

import zipfile
import argparse

from PIL import Image

import subprocess
import re
import glob
Image.MAX_IMAGE_PIXELS = None

import time
import keyboard
import shutil
import math
import natsort
import itertools
import string
import ctypes
import unidecode

# Custom modules
from paths import PathGenerator, OutputPathGenerator

# pyjion
# import pyjion
# print(f"Pyjion: {pyjion.enable()}")
# pyjion.config(level=2)

# os.path.sep = '/'

working_dir = 'C:/Users/aurel/OneDrive/Scripts/upscale/Real-ESRGAN'

model_name = 'RealESRGAN_x4plus_anime_6B'
executable_path = working_dir + '/inference_realesrgan.py'
seven_zip_path = "C:/Users/aurel/OneDrive/Scripts/upscale/7z2201-extra/7za.exe"

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
    parser.add_argument("--remove_root_folders", type=int, default=0, help="Remove the number of folders from the root of the output path. Eg with a value of 2 : ./a/b/test.cbz -> ./test.cbz")
    parser.add_argument('--sync', action='store_true', help='Synchronize the input and output folders')
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

def extract(file, output_path=None):
    # Create a folder with the same name as the file
    folder_name = file[:-4] if output_path is None else output_path
    folder_name = os.path.abspath(folder_name)
    folder_name = os.path.normpath(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # Extract the files to the folder
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(folder_name)

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
        if img.width < width:
            continue

        ratio = width / img.width
        if ratio >= 1:
            continue
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)

        # Save the image
        img.save(image, format=format)
        print(f"  {i + 1}/{size} {os.path.basename(image)}", end='\r')


def compress(folder, output_path):
    # Compress the folder to a cbz file
    ret = subprocess.run([seven_zip_path, 'a', output_path, folder + '/*'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret.returncode != 0:
        raise Exception(f"Failed to compress {folder} to {output_path}")


end_after_upscaling = False
def upscale(input_path, output_path, scale, format=None, width=0, tiles=1024):
    global end_after_upscaling
    args = ["python", executable_path, '-i', input_path, '-o', output_path, '-n', model_name, '-s', str(scale), "-t", f"{tiles}", "--tile_pad", "10", "--width", str(width)]
    if format is not None:
        args += ['--ext', format]

    # print("Running:", args[0], *[f"'{arg}'" for arg in args[1:] ])

    end_after_upscaling = False
    ret = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    while ret.poll() is None:
        # Check if the user has pressed q to quit after the upscaling
        if keyboard.is_pressed('q') and keyboard.is_pressed('ctrl') and not end_after_upscaling:
            print("<<< User pressed ctrl + q, no more files will be processed after the current one >>>")
            end_after_upscaling = True
        time.sleep(0.1)

    if ret.returncode != 0:
        raise Exception("Upscaling failed")

def rm_tree(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

def normalize_path(path):
    return os.path.abspath(os.path.normpath(path))

seen_files = set()
def sync(args):
    global seen_files

    paths = set()
    for f in seen_files:
        gen = OutputPathGenerator(args, f)
        paths.update(gen.possible_paths())

    paths = {normalize_path(p) for p in paths}

    out_paths = get_cbz_files(args.output, ignore_upscaled=False)
    out_paths = {normalize_path(p) for p in out_paths}
    for p in out_paths:
        if p not in paths:
            print(f"Removing {p}")
            # os.remove(p)
            # folder = os.path.dirname(p)
            # if len(os.listdir(folder)) == 0:
            #     os.rmdir(folder)


def main(args):
    global seen_files

    # Get the list of .cbz files
    found_paths = get_cbz_files(args.input, ignore_upscaled=True)

    # Sort the files by name
    found_paths = natsort.natsorted(found_paths)

    paths = [p for p in found_paths if p not in seen_files]
    if len(paths) == 0:
        return 0
    print(f"Found {len(paths)} files")
    input_directory = os.path.abspath(args.input)
    if not os.path.isdir(args.input):
        input_directory = os.path.dirname(input_directory)

    # Process each .cbz file
    for i, file in enumerate(paths):
        if file in seen_files:
            continue
        seen_files.add(file)

        if not os.path.exists(file):
            print(f"Skipping {os.path.relpath(file, args.input)} (does not exist)")
            paths[i] = None
            continue

        gen = OutputPathGenerator(args, file)

        if gen.exists():
            print(f"Skipping {os.path.relpath(file, input_directory)} (already exists)")
            seen_files.update(gen.possible_paths())
            paths[i] = None
            continue
        print(f"Processing {os.path.relpath(file, args.input)} ({i + 1}/{len(paths)})")
        print(f"    Output: {os.path.relpath(gen.output_path, args.output)}")

        print(f"    Extracting to {os.path.basename(gen.extract_path)}")
        os.makedirs(gen.extract_path, exist_ok=True)
        extract(file, gen.extract_path)

        print(f"    Upscaling to {os.path.basename(gen.upscale_path)} ({len(os.listdir(gen.extract_path))} images)")
        os.makedirs(gen.upscale_path, exist_ok=True)
        upscale(gen.extract_path, gen.upscale_path, args.scale, args.format, width=args.width, tiles=args.tiles)

        rm_tree(gen.extract_path)

        # print(f"    Fitting images to width", args.width)
        # fit_to_width(upscale_path, args.width)

        print(f"    Writing final images to {gen.output_path}")
        if args.compress:
            compress(gen.upscale_path, gen.output_path)
            rm_tree(gen.upscale_path)
        else:
            os.rename(gen.upscale_path, gen.output_path)

        print(f"Done {os.path.basename(file)} ({i + 1}/{len(paths)})")
        if end_after_upscaling:
            print("<<< Exiting... >>>")
            exit()

    if args.sync:
        print("Syncing...")
        sync(args)

    # if all paths are in seen_files, return 0
    if all([path in seen_files for path in paths if path is not None]):
        return 0

    return len([path for path in paths if path is not None])

def print_sleep(duration: int, step: int = 5):
    for i in range(0, duration, step):
        print(f"\033[KSleeping for {duration - i} seconds... ", end='\r')
        time.sleep(5)
    print("\033[K", end='\r')

if __name__ == '__main__':
    args = parse_args()

    # Print all the arguments
    print("Arguments:")
    argparse_dict = vars(args)
    for key in argparse_dict:
        print(f"  {key}: {argparse_dict[key]}")
    print()

    count = -1
    while True:
        print("\033[K", end='\r')
        count = main(args)

        print(f"Done {count} files")
        if count == 0:
            if not args.daemon:
                print("No more files to process")
                break
            print_sleep(60)
        else:
            print("\033[KChecking for new files...", end='\r')
