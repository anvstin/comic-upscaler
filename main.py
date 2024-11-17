import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from upscaling import get_realesrgan_model, ImageContainer, UpscaleConfig, UpscaleData
from upscaling.containers import ZipInterface, DirInterface
from upscaling.upscale import upscale_file
from upscaling.upscaler_config import ModelDtypes
import os
log = logging.getLogger()
logging.basicConfig(level=os.getenv('LEVEL', 'INFO'), format = '{asctime} {module:10} [{levelname}] - {message}', style='{')

import argparse
import multiprocessing.managers
import os
import multiprocessing

from rich.progress import Progress

from paths import OutputPathGenerator
from utils.files import get_closest_dir, prune_empty_folders, get_sorted_comic_files, get_sorted_comic_files_parallel, \
    dir_by_dir_get_sorted_comic_files_parallel, dir_by_dir_parallel_walk, ls_dir
from utils.terminal import print_sleep
from utils.files import sync_files

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


def does_exists(gen_args, file):
    gen: OutputPathGenerator = OutputPathGenerator.from_dict(gen_args, file) # type: ignored

    return file, gen.output_path, not gen.exists()

def get_to_process(args: argparse.Namespace, file_mapping: dict[str], extensions=("cbz", "zip"), executor_class = ThreadPoolExecutor):
    to_process_dirs = [args.input]
    to_process_files = []

    seen_files = set(file_mapping.keys())

    gen_args = {
        "input": args.input,
        "output": args.output,
        "suffix": args.suffix,
        "compress": args.compress,
        "rename": args.rename,
        "remove_root_folders": args.remove_root_folders,
    }
    with executor_class() as executor:
        while len(to_process_dirs) > 0:
            log.debug(f"parallel_walk: len(to_process): {len(to_process_dirs)}")
            current_dirs = []
            for sub_files, sub_dirs in executor.map(ls_dir, to_process_dirs):
                current_dirs.extend(sub_dirs)

                unseen_cbz_files = set(x for x in sub_files if any(x.endswith(ext) for ext in extensions)) - seen_files
                to_process_files.extend(unseen_cbz_files)
            to_process_dirs = current_dirs

        for filepath, output_filepath, should_process in executor.map(does_exists, *zip(*((gen_args, x) for x in to_process_files))):
            if not should_process:
                file_mapping[filepath] = output_filepath
            yield filepath, should_process

def main(args: argparse.Namespace, params: multiprocessing.managers.Namespace, file_mapping: dict[str], model_data: UpscaleData) -> int:
    """
    Main function to process the comics files

    Args:
        args (argparse.Namespace): The arguments
        params (multiprocessing.managers.Namespace): The shared parameters (end_after_upscaling, need_pruning, file_mapping)
        file_mapping (dict): The mapping of input files to output files

    Returns:
        int: The number of files processed
    """

    processed_paths = [] # Comics files found
    input_directory = get_closest_dir(args.input)

    # Process each .cbz file
    with Progress(transient=True) as progress:
        # progress_bar = progress.add_task("Processing files...", total=len(paths))
        progress_bar = progress.add_task("Processing files...", total=None, visible=False)
        for i, (file, should_process) in enumerate(get_to_process(args, file_mapping)):
            if params.end_after_upscaling:
                log.info("<<< Exiting... >>>")
                exit()
            if not should_process:
                log.info(f"Skipping {os.path.relpath(file, input_directory)} (already exists)")
                continue

            gen: OutputPathGenerator = OutputPathGenerator.from_args(args, file) # type: ignored

            # Start processing the file
            processed_paths.append(file)

            log.info(f"Processing {os.path.relpath(file, args.input)}")
            log.info(f"    Output: {os.path.relpath(gen.output_path, args.output)}")
            # os.makedirs(gen.output_path_folder)
            image_container = ImageContainer(container_path=Path(file))
            tmp_out = Path(gen.output_path + ".tmp")
            output_interface = ZipInterface(tmp_out, write=True)

            try:
                upscale_file(model_data, image_container, output_interface)
                log.info(f"Renaming {tmp_out.name} to {gen.output_path}")
                tmp_out.rename(gen.output_path)
            except Exception as e:
                log.error(f"    <<< Error upscaling {os.path.basename(file)} >>>")
                log.error(f"Exception: {e}")
                # raise RuntimeError("EOEOEOE")
                continue

            file_mapping[file] = gen.output_path # Update the file mapping (seen_files)
            log.info(f"Done {os.path.basename(file)} ({i + 1}/{len(processed_paths)})")

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

    model_data = get_realesrgan_model(
        UpscaleConfig(
            model_name="R-ESRGAN AnimeVideo",
            device="cuda",
            model_dtype=ModelDtypes.HALF
        )
    )
    model_data.download_model()

    while True:
        count = -1
        while count != 0:
            print("\033[K", end='\r')
            count = main(args, params, file_mapping, model_data)


        if args.sync:
            print("Syncing...", end='\r')
            sync_files(args, file_mapping)
            params.need_pruning = True

        if params.need_pruning:
            prune_empty_folders(args.output)
            params.need_pruning = False

        print(f"Done {count} files")
        model_data.unload_model()
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
    start_processing(args,params)
    # # Start the processing
    # p = multiprocessing.Process(target=start_processing, args=(args,params))
    # p.start()

    # # Start a terminal to run commands
    # p2 = multiprocessing.Process(target=terminal, args=(args,params))
    # p2.start()

    # # Wait for the processes to finish
    # while True:
    #     if not p.is_alive():
    #         break
    #     if not p2.is_alive():
    #         break
    #     time.sleep(1)

    # # Kill the processes
    # p.terminate()
    # p2.terminate()






