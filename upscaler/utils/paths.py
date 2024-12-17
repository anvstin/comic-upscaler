import os
import subprocess

def get_program_path(program_name: str) -> str|None:
    """
    Get the path to the program with the given name

    Args:
        program_name (str): The name of the program

    Returns:
        str: The path to the program
    """
    try:
        # Windows
        if os.name == 'nt':
            return subprocess.check_output(['where', program_name], stderr=subprocess.DEVNULL).decode().strip()
        # Linux and MacOS
        else:
            return subprocess.check_output(['which', program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def cleanup_path(path: str) -> str:
    """
    Normalize a path and convert it to an absolute path

    Args:
        path (str): The path to normalize

    Returns:
        str: The normalized path
    """
    return os.path.abspath(os.path.normpath(path))
