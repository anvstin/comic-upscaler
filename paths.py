import platform
import string
import itertools
import ctypes
import os
import re
import unidecode

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

class OutputPathGenerator(PathGenerator):

    def __init__(self, args, file):
        super().__init__()

        self.input = args.input
        self.output = args.output
        self.suffix = args.suffix
        self.compress = args.compress
        self.rename = args.rename
        self.remove_root_folders = args.remove_root_folders
        self.file = file

        self.compress_extensions = [".cbz", ".zip"]

        # Useful values
        self.path_no_ext = self.get_path_no_ext(file)
        self.input_directory = self.get_input_directory()
        self.output_as_folder = self.get_output_as_folder()
        self.output_path_compress = f"{self.output_as_folder}{self.suffix}.cbz"
        self.output_path_folder = f"{self.output_as_folder}{self.suffix}"
        self.output_path = self.output_path_compress if self.compress else self.output_path_folder

    def get_input_directory(self):
        input_directory = os.path.abspath(self.input)
        if not os.path.isdir(self.input):
            input_directory = os.path.dirname(input_directory)
        return input_directory

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

    @property # Dynamic properties (ramdisk can be removed or added at runtime)
    def extract_path(self): return super().apply_ramdisk(f"{self.output_as_folder}_extracted")
    @property # Dynamic properties (ramdisk can be removed or added at runtime)
    def upscale_path(self): return super().apply_ramdisk(f"{self.output_as_folder}{self.suffix}_tmp")

    def possible_paths(self):
        return [
            self.output_path_compress,
            self.output_path_folder,
        ] + [
            f"{self.output_path_compress_no_ext}{ext}" for ext in self.compress_extensions
        ]

    def exists(self):
        to_check = self.possible_paths()
        return any([os.path.exists(path) for path in to_check])
