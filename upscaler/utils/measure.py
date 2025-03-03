import logging
import time
from typing import Callable, Any

import numpy as np

log = logging.getLogger(__name__)


def measure_ns(func: Callable) -> tuple[int, Any]:
    start = time.perf_counter_ns()
    res = func()
    end = time.perf_counter_ns()
    return end - start, res


def nanoseconds_pretty_str(total_nanoseconds: int) -> str:
    total_nanoseconds = int(total_nanoseconds)
    total_milliseconds, _ = np.divmod(total_nanoseconds, 1_000_000)
    total_seconds, milliseconds = np.divmod(total_milliseconds, 1_000)
    total_minutes, seconds = np.divmod(total_seconds, 60)
    hours, minutes = np.divmod(total_minutes, 60)
    return f"{hours}h {minutes}m {seconds}.{milliseconds:03}s (total: {np.divide(total_nanoseconds, 1_000_000):_}ms)"


class Measurement(object):
    def __init__(self, func: Callable, log_call: bool = True, log_stats: bool = True):
        self.measurements = list[int]()
        self.func = func
        self.log_call = log_call
        self.log_stats = log_stats

    def clear(self):
        self.measurements = []

    def call(self):
        total, res = measure_ns(self.func)
        self.measurements.append(total)
        if self.log_call: log.info(f"Function {self.func.__name__} done in {nanoseconds_pretty_str(total)}")
        return res

    def average(self) -> int:
        mean = np.mean(self.measurements).round()
        if self.log_stats:
            log.info(f"Average measurement for {self.func.__name__}: {nanoseconds_pretty_str(mean)}")
        return mean

    def median(self) -> int:
        median = np.median(self.measurements).round()
        if self.log_stats:
            log.info(f"Median measurement for {self.func.__name__}: {nanoseconds_pretty_str(median)}")
        return median
