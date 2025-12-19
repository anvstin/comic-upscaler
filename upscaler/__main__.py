import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator
import torch
from rich.progress import Progress

from .paths import OutputPathGenerator, OutputPathGeneratorConfig
from .upscaling import get_realesrgan_model, ImageContainer, UpscaleConfig, UpscaleData
from .upscaling.containers import ZipInterface
from .upscaling.upscale import upscale_file
from .upscaling.upscaler_config import ModelDtypes
from .utils.files import get_closest_dir, prune_empty_folders, ls_dir, sync_files
from .utils.terminal import print_sleep
from .metadata import save_metadata, load_metadata, create_metadata, build_metadata_from_files, FileMetadata
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





def _validate_metadata(args_tuple: tuple[OutputPathGeneratorConfig, str, dict[str, FileMetadata]]) -> str:
    config, filepath, file_mapping = args_tuple
    gen = OutputPathGenerator(filepath, config)

    log.debug(f"Validating {filepath}")

    try:
        dest_stat =  Path(gen.output_path).stat()
    except FileNotFoundError:
        return filepath

    metadata = file_mapping.get(filepath)
    if metadata:
        try:
            src_stat = Path(filepath).stat()
        except FileNotFoundError:
            log.info(f"File {filepath} not found, deleting from metadata")
            return filepath
        if src_stat.st_size != metadata.source_size or dest_stat.st_size != metadata.destination_size:
            log.info(f"File {filepath} has been modified, re-upscaling")
            return filepath
        if dest_stat.st_size < src_stat.st_size:
            log.info(f"Invalid size for {filepath}, re-upscaling")
            return filepath

    return None


def get_to_process(args: argparse.Namespace, valid_files: set[str], extensions=("cbz", "zip"),
                   executor_class=ThreadPoolExecutor) -> Generator[str, None, None]:
    """
    Get the files to process

    Args:
        args (argparse.Namespace): The arguments
        valid_files (set): Set of files that are already processed and valid
        extensions (tuple, optional): The extensions to process. Defaults to ("cbz", "zip").
        executor_class (_type_, optional): The executor class to use. Defaults to ThreadPoolExecutor.

    Yields:
        Generator[str, None, None]: The files to process
    """
    to_process_dirs = [args.input]

    with executor_class() as executor:
        while len(to_process_dirs) > 0:
            log.debug(f"parallel_walk: len(to_process): {len(to_process_dirs)}")
            current_dirs = []
            for sub_files, sub_dirs in executor.map(ls_dir, to_process_dirs):
                current_dirs.extend(sub_dirs)
                for filepath in sub_files:
                    if any(filepath.endswith(ext) for ext in extensions) and filepath not in valid_files:
                        yield filepath
            to_process_dirs = current_dirs


def process_files(args: argparse.Namespace, params: multiprocessing.managers.Namespace, file_mapping: dict[str, FileMetadata],
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
        for i, file in enumerate(get_to_process(args, params.valid_files)):
            if params.end_after_upscaling:
                log.info("<<< Exiting... >>>")
                exit()

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
                tmp_out.replace(gen.output_path)
            except Exception as e:
                log.exception(f"<<< Error upscaling {os.path.basename(file)} >>>")
                if stop_on_failures:
                    raise e
                continue

            file_mapping[file] = create_metadata(file, gen.output_path)
            params.valid_files.append(file)
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

    file_mapping: dict[str, FileMetadata] = params.file_mapping

    loaded_metadata = load_metadata(args.output)
    if not loaded_metadata:
        log.info("No metadata found, building from existing files...")
        loaded_metadata = build_metadata_from_files(args.input, OutputPathGeneratorConfig.from_args(args))
    file_mapping.update(loaded_metadata)

    # Validate metadata once at startup
    if file_mapping:
        log.info(f"Validating metadata for {len(file_mapping)} files...")
        config = OutputPathGeneratorConfig.from_args(args)
        with ThreadPoolExecutor() as executor:
            args_tuples = [(config, filepath, file_mapping) for filepath in file_mapping.keys()]
            invalid_files = [f for f in executor.map(_validate_metadata, args_tuples) if f is not None]

        # Remove invalid entries from metadata
        for filepath in invalid_files:
            file_mapping.pop(filepath, None)

        if invalid_files:
            log.info(f"Found {len(invalid_files)} invalid files, removed from metadata")
        log.info(f"Metadata validation complete, {len(file_mapping)} valid files")

    params.valid_files[:] = file_mapping.keys()

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

        save_metadata(args.output, list(file_mapping.values()))

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
    params.valid_files = manager.list()

    args = parse_args(argv)

    # Print all the arguments
    log.info("Arguments:")
    argparse_dict = vars(args)
    for key in argparse_dict:
        log.info(f"  {key}: {argparse_dict[key]}")

    start_processing(args, params, args.stop_on_failures)

if __name__ == "__main__":
    main(sys.argv)
