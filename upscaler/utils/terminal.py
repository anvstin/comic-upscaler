import argparse
import multiprocessing.managers
import sys
import time

from rich.progress import Progress

from . import sync_files


def terminal(args: argparse.Namespace, params: multiprocessing.managers.Namespace):
    sys.stdin = open(0)
    while True:
        cmd = input()
        if cmd == "sync":
            sync_files(args, params.file_mapping)
            params.need_pruning = True
        elif cmd == "q":
            params.end_after_upscaling = True
            print("<<< User pressed q, no more files will be processed after the current one >>>")
        else:
            print("Unknown command")


def print_sleep(duration: int, step: float = 0.5) -> None:
    """
    Print a sleep with a progress bar

    Args:
        duration (int): The duration of the sleep
        step (int, optional): The step of the progress bar. Defaults to 0.5.
    """
    # Use rich to print sleep
    with Progress(speed_estimate_period=2, transient=True) as progress:
        task = progress.add_task("Sleeping...", total=duration)
        nb_iter = int(duration / step)
        for i in range(nb_iter):
            progress.update(task, advance=step)
            time.sleep(step)
        time.sleep(duration % step)
        progress.stop()
