######## CONFIGURATION ########
from os.path import join, dirname, realpath

SCRIPT_DIR = dirname(realpath(__file__))




UPSCALE_SCRIPT_PATH = join(SCRIPT_DIR, "Real-ESRGAN", "inference_realesrgan.py")
SEVEN_ZIP_PATH = join(SCRIPT_DIR, "7z2201-extra", "x64", "7za.exe")
MODEL_NAME = "realesr-animevideov3"
VERSION = "0.0.1"
