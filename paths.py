import platform
import string
import itertools
import ctypes
import os
import re
import unidecode
import copy
import functools
from functools import cached_property, cache

class PathGenerator(object):

    @staticmethod
    def get_available_drives():
        if 'Windows' not in platform.system():
            return []
        drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
        return set(itertools.compress(string.ascii_uppercase,
                map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))

    @staticmethod
    def apply_ramdisk(output_as_folder, letter="R:"):
        # Check if drive letter R: is mounted in windows
        ramdisk = letter.strip(":") in PathGenerator.get_available_drives()
        if ramdisk:
            # Replace the drive letter with R:
            output_as_folder = output_as_folder.replace(os.path.splitdrive(output_as_folder)[0], 'R:')
        return output_as_folder

@cache
class OutputPathGenerator(PathGenerator):
    allowed_characters = string.ascii_letters + string.digits + " -_().,;':/\\"

    @staticmethod
    def from_args(args, file):
        return OutputPathGenerator(file, args.input, args.output, args.suffix, args.compress, args.rename, args.remove_root_folders)

    @staticmethod
    def from_dict(d, file):
        return OutputPathGenerator(file, d["input"], d["output"], d["suffix"], d["compress"], d["rename"], d["remove_root_folders"])

    def __init__(self, file, input_path, output_path, suffix, compress, rename, remove_root_folders):
        super().__init__()

        self.file = file

        self.input = input_path
        self.output = output_path
        self.suffix = suffix
        self.compress = compress
        self.rename = rename
        self.remove_root_folders = remove_root_folders

        self.compress_extensions = [".cbz", ".zip"]

        # Useful values
        self.path_no_ext = self.get_path_no_ext(file)
        self.input_directory = self.get_input_directory()
        self.output_as_folder = self.get_output_as_folder()
        self.output_path_compress = f"{self.output_as_folder}{self.suffix}.cbz"
        self.output_path_folder = f"{self.output_as_folder}{self.suffix}"
        self.output_path = self.output_path_compress if self.compress else self.output_path_folder

        self.last = dict()
        self.possible_paths_dict = {
            self.output_path_compress, self.output_path_folder
        }
        self.output_path = self.remove_invalid_characters(self.output_path)
        self.output_path_folder = self.remove_invalid_characters(self.output_path_folder)
        self.output_path_compress = self.remove_invalid_characters(self.output_path_compress)

        self.possible_paths_dict.update({self.output_path_compress, self.output_path_folder})



    def get_input_directory(self):
        input_directory = os.path.abspath(self.input)
        if not os.path.isdir(self.input):
            input_directory = os.path.dirname(input_directory)
        return input_directory

    def remove_invalid_characters(self, path):
        # path = unidecode.unidecode(path, errors="preserve")
        # Convert all characters to ascii
        path = "".join([c if c in self.allowed_characters else unidecode.unidecode(c) for c in path])
        # Remove double spaces
        path = re.sub(r' +', ' ', path)
        # Strip spaces for each folder
        path = os.path.sep.join([c.strip() for c in path.split(os.path.sep)])
        path = path.strip()
        # Remove all characters that are not in the allowed characters
        return "".join([c for c in path if c in self.allowed_characters])

    def get_output_as_folder(self):
        relpath = os.path.relpath(self.path_no_ext, self.input_directory)
        # relpath = unidecode.unidecode(relpath)
        output_as_folder = os.path.join(self.output, relpath)
        if self.rename:
            # Get the basename of ..
            tmp = os.path.realpath(output_as_folder + "/..")
            tmp = os.path.basename(tmp)
            output_as_folder = os.path.join(os.path.dirname(output_as_folder), f"{tmp} - {os.path.basename(output_as_folder)}")
            # Remove all invisible characters
            output_as_folder = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', output_as_folder)

        if self.remove_root_folders > 0:
            # Remove the n number of root folder of the output path
            # Example with n=1: "./test/some_folder/file.cbz" -> "./some_folder/file.cbz"
            tmp = os.path.relpath(output_as_folder, self.output)
            n = self.remove_root_folders
            split = tmp.split(os.path.sep)
            if len(split) <= n:
                raise ValueError(f"Cannot remove {n} folders from {tmp}")
            tmp = os.path.join(self.output, os.path.join(*split[n:]))
            output_as_folder = os.path.join(self.output, tmp)

        return output_as_folder

    def get_path_no_ext(self, file):
        res, _ = os.path.splitext(os.path.basename(file))
        res = os.path.join(os.path.dirname(file), res)
        return res

    @property
    def output_path_compress_no_ext(self): return self.output_path_folder

    @cached_property
    def extract_path(self):
        res = super().apply_ramdisk(f"{self.output_as_folder}_extracted")
        res = self.remove_invalid_characters(res)
        return res

    @cached_property # Dynamic properties (ramdisk can be removed or added at runtime)
    def upscale_path(self):

        self.last["upscale_path"] = super().apply_ramdisk(f"{self.output_as_folder}_upscaled")
        self.last["upscale_path"] = self.remove_invalid_characters(self.last["upscale_path"])
        return self.last["upscale_path"]

    @cache
    def possible_paths(self, generate=True):
        paths = set()
        if generate:
            tmp = OutputPathGenerator(self.file, self.input, self.output, self.suffix, self.compress, not self.rename, self.remove_root_folders)
            paths = tmp.possible_paths(generate=False)

        return paths | self.possible_paths_dict | {
            f"{self.output_path_compress_no_ext}{ext}" for ext in self.compress_extensions
        }


    def exists(self):
        to_check = self.possible_paths()
        return any([os.path.exists(path) for path in to_check])

