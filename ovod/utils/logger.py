"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson
import time
from collections import Counter

from . import distributed as du
from .env import (pathmgr)


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # Use 1K buffer if writing to cloud storage.
    io = pathmgr.open(
        filename, "a", buffering=1024 if "://" in filename else -1
    )
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None, set_debug=True):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    if set_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        if set_debug:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None and du.is_master_proc(du.get_world_size()):
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        if set_debug:
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """

    logger = logging.getLogger(name)
    # Things the logger cannot handle well
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    return logger


def log_json_stats(stats, output_dir=None):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.3f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
    if du.is_root_proc() and output_dir:
        filename = os.path.join(output_dir, "json_stats.log")
        try:
            with pathmgr.open(
                filename, "a", buffering=1024 if "://" in filename else -1
            ) as f:
                f.write("json_stats: {:s}\n".format(json_stats))
        except Exception:
            logger.info(
                "Failed to write to json_stats.log: {}".format(json_stats)
            )


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "detectron2"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}

def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time





# main:
if __name__ == '__main__':
    # Set up logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = get_logger("Training Script")
    #
    # # Use logger in your script
    # logger.info("Starting training...")
    # # ... your training code
    # logger.info("Training completed.")
    # Setting up the logging.
    # Assume we want to write the log to an 'output' directory (you can change this)

    if not os.path.exists('stam_output'):
        os.makedirs('stam_output')
    setup_logging(output_dir='stam_output')

    logger = get_logger(__name__)  # Getting the logger using your get_logger function

    # Logging messages of different levels
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")


