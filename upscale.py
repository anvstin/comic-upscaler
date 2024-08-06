import argparse
import glob
import multiprocessing.managers
import os
import subprocess
import sys
import multiprocessing
import time
import natsort

from rich.progress import Progress
from rich.console import Console

from paths import OutputPathGenerator
from global_config import *
from utils.files import get_closest_dir, rm_tree, prune_empty_folders, get_sorted_comic_files
from utils.compression import compress, extract
from utils.terminal import terminal, print_sleep
from utils.files import sync_files

console = Console()

def parse_args() -> argparse.Namespace:
    """
    Parse the program arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file or folder')
    parser.add_argument('output', help='Output file or folder', default=None, nargs='?')
    # parser.add_argument('-d', '--divide', type=int, default=2, help='Size to divide the images by')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as a daemon')
    parser.add_argument('-f', '--format', default="webp", help='Output images format')
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

def upscale(input_path: str, output_path: str, scale: int, format: str|None = None,
            width: int=0, tiles: int=1024, wait: bool=True, tile_pad: int=50,
            fp32: bool=False) -> subprocess.Popen:
    """
    Upscale the images in the input folder and save them in the output folder

    Args:
        input_path (str): The path to the input folder
        output_path (str): The path to the output folder
        scale (int): The scale to upscale the images by
        format (str, optional): The format of the output images. Defaults to None.
        width (int, optional): The width to fit the images to. Defaults to 0.
        tiles (int, optional): The number of tiles to split the images into. Defaults to 1024.
        wait (bool, optional): Wait for the process to finish. Defaults to True.
        tile_pad (int, optional): The padding of the tiles. Defaults to 50.
        fp32 (bool, optional): Use fp32 instead of fp16. Defaults to False.

    Raises:
        Exception: Upscaling failed
    """

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




def main(args: argparse.Namespace, params: multiprocessing.managers.Namespace, file_mapping: dict) -> int:
    """
    Main function to process the comics files

    Args:
        args (argparse.Namespace): The arguments
        params (multiprocessing.managers.Namespace): The shared parameters (end_after_upscaling, need_pruning, file_mapping)
        file_mapping (dict): The mapping of input files to output files

    Returns:
        int: The number of files processed
    """

    seen_files = set(file_mapping.keys())

    processed_paths = [] # Comics files found
    input_directory = get_closest_dir(args.input)

    # Process each .cbz file
    with Progress(transient=True) as progress:
        # progress_bar = progress.add_task("Processing files...", total=len(paths))
        progress_bar = progress.add_task("Processing files...", total=None, visible=False)
        for i, file in enumerate(get_sorted_comic_files(args.input, ignore_upscaled=True)):
            if params.end_after_upscaling:
                print("<<< Exiting... >>>")
                exit()

            if file in seen_files:
                continue

            progress.update(progress_bar, advance=1, visible=True)
            gen: OutputPathGenerator = OutputPathGenerator.from_args(args, file) # type: ignore
            if gen.exists():
                print(f"Skipping {os.path.relpath(file, input_directory)} (already exists)")
                file_mapping[file] = gen.output_path
                continue

            # Start processing the file
            processed_paths.append(file)

            print(f"Processing {os.path.relpath(file, args.input)} ({i + 1}/{len(processed_paths)})")
            print(f"    Output: {os.path.relpath(gen.output_path, args.output)}")
            # progress.update(progress_bar, visible=False)

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
            try:
                progress.update(progress_bar, visible=False)
                upscale(gen.extract_path, gen.upscale_path, args.scale, args.format, width=args.width, tiles=args.tiles, fp32=args.fp32)
                progress.update(progress_bar, visible=True)
                rm_tree(gen.extract_path)
            except:
                print(f"    <<< Error upscaling {os.path.basename(file)} >>>")
                continue

            # print(f"    Fitting images to width", args.width)
            # fit_to_width(upscale_path, args.width)

            print(f"    Writing final images to {gen.output_path}")
            try:
                if args.compress:
                    compress(gen.upscale_path, gen.output_path, progress=progress)
                    rm_tree(gen.upscale_path)
                else:
                    os.rename(gen.upscale_path, gen.output_path)
            except:
                print(f"    <<< Error writing {os.path.basename(file)} >>>")
                continue

            file_mapping[file] = gen.output_path # Update the file mapping (seen_files)
            print(f"Done {os.path.basename(file)} ({i + 1}/{len(processed_paths)})")

        progress.remove_task(progress_bar)
        progress.stop()

    return len(processed_paths)


def start_processing(args: argparse.Namespace, params: multiprocessing.managers.Namespace) -> None:
    """
    Start the processing of the comics files

    Args:
        args (argparse.Namespace): The arguments
        params (multiprocessing.managers.Namespace): The shared parameters (end_after_upscaling, need_pruning, file_mapping)
    """

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






