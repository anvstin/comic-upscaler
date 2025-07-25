import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from rich.progress import Progress

from .paths import OutputPathGenerator, OutputPathGeneratorConfig
from .upscaling import get_realesrgan_model, ImageContainer, UpscaleConfig, UpscaleData
from .upscaling.containers import ZipInterface
from .upscaling.upscale import upscale_file
from .upscaling.upscaler_config import ModelDtypes
from .utils.files import get_closest_dir, prune_empty_folders, ls_dir, sync_files
from .utils.terminal import print_sleep

logging.basicConfig(level=os.getenv('LEVEL', 'INFO'), format='{asctime} {module:10} [{levelname}] - {message}',
                        style='{', force=True)
log = logging.getLogger()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse the program arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    def positive_int(value):
        try:
            value = int(value)
            if value < 0:
                raise argparse.ArgumentTypeError("{} is not a positive integer".format(value))
        except ValueError:
            raise Exception("{} is not an integer".format(value))
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file or folder')
    parser.add_argument('output', help='Output file or folder')
    # parser.add_argument('-d', '--divide', type=int, default=2, help='Size to divide the images by')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as a daemon')
    parser.add_argument('-f', '--format', default="webp", help='Output images format')
    # parser.add_argument('-s', '--scale', type=int, default=4, help='Scale the images by')
    parser.add_argument('-w', '--width', type=positive_int, default=UpscaleConfig.output_max_width, help='Fit the images to the width')
    # parser.add_argument('-t', '--tiles', type=int, default=1024, help='Split the images into tiles')
    # parser.add_argument('--tile_pad', type=int, default=50, help='Pad the tiles by')
    parser.add_argument("--remove_root_folders", type=positive_int, default=0,
                        help="Remove the number of folders from the root of the output path. Eg with a value of 2 : ./a/b/test.cbz -> ./test.cbz")
    parser.add_argument('--sync', action='store_true', help='Synchronize the input and output folders')
    parser.add_argument("--fp32", action='store_true', help="Use fp32 instead of fp16")
    # Compress the images after upscaling
    # parser.add_argument('-c', '--compress', action='store_true', help='Compress the images after upscaling')
    parser.add_argument("--suffix", default="_upscaled", help="Suffix to add to the output file name", nargs='?')
    parser.add_argument("--rename", action='store_true', help="Rename the output file for kavita")
    parser.add_argument("--stop-on-failures",  action='store_true', help="Stop the program execution if a failure occurs")
    parser.add_argument("--max_workers", type=positive_int, default=UpscaleConfig.upscale_workers, help="Maximum number of concurrent workers for GPU upscaling")

    parsed_args = parser.parse_args(argv[1:])

    parsed_args.input = parsed_args.input.replace('\\', '/')
    parsed_args.input = os.path.abspath(parsed_args.input)
    parsed_args.input = os.path.normpath(parsed_args.input)

    parsed_args.compress = True

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


def does_exists(config: OutputPathGeneratorConfig, file: str):
    gen: OutputPathGenerator = OutputPathGenerator(file,config)

    return file, gen.output_path, not gen.exists()


def get_to_process(args: argparse.Namespace, file_mapping: dict[str, str], extensions=("cbz", "zip"),
                   executor_class=ThreadPoolExecutor):
    to_process_dirs = [args.input]
    to_process_files = []

    seen_files = set(file_mapping.keys())

    config = OutputPathGeneratorConfig.from_args(args)
    with executor_class() as executor:
        while len(to_process_dirs) > 0:
            log.debug(f"parallel_walk: len(to_process): {len(to_process_dirs)}")
            current_dirs = []
            for sub_files, sub_dirs in executor.map(ls_dir, to_process_dirs):
                current_dirs.extend(sub_dirs)
                to_process_files.extend(
                    x for x in sub_files if any(x.endswith(ext) for ext in extensions) and not x in seen_files)
            to_process_dirs = current_dirs

        for filepath, output_filepath, should_process in executor.map(does_exists,
                                                                      *zip(*((config, x) for x in to_process_files))):
            if not should_process:
                file_mapping[filepath] = output_filepath
            yield filepath, should_process


def process_files(args: argparse.Namespace, params: multiprocessing.managers.Namespace, file_mapping: dict[str, str],
                  model_data: UpscaleData, stop_on_failures: bool) -> int:
    """
    Main function to process the comics files

    Args:
        args (argparse.Namespace): The arguments
        params (multiprocessing.managers.Namespace): The shared parameters (end_after_upscaling, need_pruning, file_mapping)
        file_mapping (dict): The mapping of input files to output files
        model_data (UpscaleData): The model data
        stop_on_failures (bool): whether to continue processing if an error occurs

    Returns:
        int: The number of files processed
    """

    processed_paths = []  # Comics files found
    input_directory = get_closest_dir(args.input)

    output_generator_config = OutputPathGeneratorConfig.from_args(args)

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

            gen: OutputPathGenerator = OutputPathGenerator(file, output_generator_config)

            log.info(f"Processing {os.path.relpath(file, args.input)}")
            log.info(f"    Output: {os.path.relpath(gen.output_path, args.output)}")
            # os.makedirs(gen.output_path_folder)
            image_container = ImageContainer(container_path=Path(file))
            tmp_out = Path(gen.output_path + ".tmp")
            output_interface = ZipInterface(tmp_out, write=True)

            try:
                upscale_file(model_data, image_container, output_interface, stop_on_failures)
                log.info(f"Renaming {tmp_out.name} to {gen.output_path}")
                tmp_out.rename(gen.output_path)
            except Exception as e:
                log.exception(f"    <<< Error upscaling {os.path.basename(file)} >>>")
                if stop_on_failures:
                    raise e
                continue

            file_mapping[file] = gen.output_path  # Update the file mapping (seen_files)
            log.info(f"Done {os.path.basename(file)} ({i + 1}/{len(processed_paths)})")

            processed_paths.append(file)

        progress.remove_task(progress_bar)
        progress.stop()

    return len(processed_paths)


def start_processing(args: argparse.Namespace, params: multiprocessing.managers.Namespace, stop_on_failures: bool) -> None:
    """
    Start the processing of the comics files

    Args:
        args (argparse.Namespace): The arguments
        params (multiprocessing.managers.Namespace): The shared parameters (end_after_upscaling, need_pruning, file_mapping)
        stop_on_failures (bool): whether to continue processing if an error occurs
    """

    file_mapping: dict[str, str] = params.file_mapping

    model_data = get_realesrgan_model(
        UpscaleConfig(
            model_name="R-ESRGAN AnimeVideo",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_dtype=ModelDtypes.FLOAT if args.fp32 else ModelDtypes.HALF,
            output_max_width=args.width,
            output_format=args.format.lower(),
            upscale_workers=args.max_workers,
        )
    )
    model_data.download_model()

    while True:
        count = -1
        try:
            while count != 0:
                count = process_files(args, params, file_mapping, model_data, stop_on_failures)
                log.info(f"Done {count} files")
        except Exception as e:
            log.exception(f"Could not process library: {e}")
            if stop_on_failures:
                raise e
            continue

        model_data.unload_model()

        if args.sync:
            sync_files(args, file_mapping)
            params.need_pruning = True

        if params.need_pruning:
            prune_empty_folders(args.output)
            params.need_pruning = False

        if not args.daemon:
            log.info("No more files to process")
            break

        print_sleep(60)


def main(argv: list[str]):
    manager = multiprocessing.Manager()
    params = manager.Namespace()
    params.end_after_upscaling = False
    params.need_pruning = False
    params.file_mapping = manager.dict()

    args = parse_args(argv)

    # Print all the arguments
    log.info("Arguments:")
    argparse_dict = vars(args)
    for key in argparse_dict:
        log.info(f"  {key}: {argparse_dict[key]}")

    start_processing(args, params, args.stop_on_failures)
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


if __name__ == "__main__":
    main(sys.argv)
