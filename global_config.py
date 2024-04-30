######## CONFIGURATION ########
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))

UPSCALE_SCRIPT_PATH = join(SCRIPT_DIR, "Real-ESRGAN", "inference_realesrgan.py")
COMPRESS_METHOD = "7z" # "7z" or "integrated"
SEVEN_ZIP_PATH = join(SCRIPT_DIR, "7z2201-extra", "x64", "7za.exe") # Path to 7z executable or None
MODEL_NAME = "realesr-animevideov3"
RAMDISK_LETTER = None # The letter of the RAM disk (Windows only, None for no RAM disk)
VERSION = "0.0.1"
