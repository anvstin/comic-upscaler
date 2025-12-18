import json
import logging
import os
from dataclasses import dataclass, asdict

log = logging.getLogger(__file__)

@dataclass
class FileMetadata:
    source: str
    destination: str
    source_size: int
    destination_size: int

def create_metadata(source: str, destination: str) -> FileMetadata:
    return FileMetadata(source, destination, os.path.getsize(source), os.path.getsize(destination))

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

def save_metadata(output_dir: str, metadata_list: list[FileMetadata]) -> None:
    with open(os.path.join(output_dir, '.upscale_metadata.json'), 'w') as f:
        json.dump({m.source: asdict(m) for m in metadata_list}, f, indent=2)
    log.info(f"Metadata saved to {output_dir}")
