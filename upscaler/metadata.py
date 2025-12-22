import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from zipfile import ZipFile

from upscaler.paths import OutputPathGenerator, OutputPathGeneratorConfig

log = logging.getLogger(__file__)

@dataclass
class FileMetadata:
    source: str
    destination: str
    source_size: int
    destination_size: int


def _validate_zip_content(source: str, destination: str, src_size: int, dest_size: int) -> bool:
    """Validate ZIP file content, accounting for split files."""
    if dest_size >= src_size:
        return True
    log.info(f"File {source} size {src_size} bigger than destination size {dest_size}, validating file content")
    with ZipFile(source, 'r') as src, ZipFile(destination, 'r') as dest:
        src_files = set(src.namelist())
        dest_files = set(dest.namelist())
        for src_file in src_files:
            stem = Path(src_file).stem
            # Check if file exists with same name, or with split suffix (stem.0, stem.1, stem.0.0, etc.)
            if not any(Path(df).stem == stem or Path(df).stem.startswith(stem + ".") for df in dest_files):
                log.error(f"Source file {src_file} missing in destination")
                return False
    return True


def create_metadata(source: str, destination: str) -> FileMetadata:
    src_size = os.path.getsize(source)
    dest_size = os.path.getsize(destination)
    if not _validate_zip_content(source, destination, src_size, dest_size):
        raise RuntimeError(f"Source files missing in destination")
    return FileMetadata(source, destination, src_size, dest_size)


def validate_metadata(config: OutputPathGeneratorConfig, filepath: str, file_mapping: dict[str, FileMetadata]) -> str:
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
        if not _validate_zip_content(filepath, gen.output_path, src_stat.st_size, dest_stat.st_size):
            return filepath

    return None


def load_metadata(output_dir: str) -> dict[str, FileMetadata]:
    metadata_path = os.path.join(output_dir, '.upscale_metadata.json')
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    return {k: FileMetadata(**v) for k, v in data.items()}


def build_metadata_from_files(input_dir: str, config) -> dict[str, FileMetadata]:
    from .utils.files import parallel_walk
    from .paths import OutputPathGenerator
    metadata = {}
    log.info(f"Building metadata for {input_dir}")
    for filepath in parallel_walk(input_dir):
        if not any(filepath.endswith(ext) for ext in ('.cbz', '.zip')):
            continue
        log.debug(f"Building metadata for {filepath}")
        gen = OutputPathGenerator(filepath, config)
        if os.path.exists(gen.output_path):
            log.debug(f"File {gen.output_path} already exists, skipping.")
            metadata[filepath] = create_metadata(filepath, gen.output_path)
    return metadata


def build_metadata_from_files(input_dir: str, config) -> dict[str, FileMetadata]:
    from concurrent.futures import ThreadPoolExecutor
    from .utils.files import parallel_walk
    from .paths import OutputPathGenerator

    log.info(f"Building metadata for {input_dir}")
    files = [f for f in parallel_walk(input_dir) if any(f.endswith(ext) for ext in ('.cbz', '.zip'))]
    output_paths = [OutputPathGenerator(f, config).output_path for f in files]

    with ThreadPoolExecutor() as executor:
        exists_results = list(executor.map(os.path.exists, output_paths))
        files = [f for f, exists in zip(files, exists_results) if exists]
        output_paths = [o for o, exists in zip(output_paths, exists_results) if exists]
        source_sizes = list(executor.map(os.path.getsize, files))
        dest_sizes = list(executor.map(os.path.getsize, output_paths))

    return {f: FileMetadata(f, o, ss, ds) for f, o, ss, ds in zip(files, output_paths, source_sizes, dest_sizes)}


def save_metadata(output_dir: str, metadata_list: list[FileMetadata]) -> None:
    with open(os.path.join(output_dir, '.upscale_metadata.json'), 'w') as f:
        json.dump({m.source: asdict(m) for m in metadata_list}, f, indent=2)
    log.info(f"Metadata saved to {output_dir}")
