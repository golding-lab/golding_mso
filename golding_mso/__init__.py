"""
GoldingMSO - A package for neuronal modeling and simulation

This package provides tools for compartmental MSO modeling using the NEURON simulator,
with specialized modules for handling cell morphologies, running simulations, and
analyzing data related to interaural time difference (ITD) processing.

"""

import logging
import os
import pathlib
from importlib.resources import files
from logging.handlers import TimedRotatingFileHandler
from neuron import h
h.load_file("import3d.hoc") # Load NEURON 3D import file




from .config import Config
from . import utils

pkg_mech_compiled = (utils.get_package_path() / "mechanisms" / "x86_64").exists() or (
    utils.get_package_path() / "mechanisms" / "nrnmech.dll"
).exists()
if pkg_mech_compiled is False:
    utils.compile_mechs()
else:
    utils.load_pkg_dll()
__version__ = "0.1.0"

# Create library dictionary to access morphologies from top level
morphologies = utils.get_morphologies()
"""Dictionary of available morphology files in the package, keyed by morphology name."""

mechanisms = utils.get_mechanisms()
"""Dictionary of available mechanism files in the package, keyed by mechanism name."""

user_pkg_dir = pathlib.Path.home() / ".golding_mso"
"""Path to the user's golding_mso configuration directory."""

user_config = Config(user_pkg_dir)
"""Current configuration (dict) loaded from the user's package config file"""

from .cell import Cell
from . import cell_calc
from . import math_calc
from . import data
from . import sims
from . import syns


# Source:
# https://stackoverflow.com/questions/49049044/python-setup-of-logging-allowing-multiline-strings-logging-infofoo-nbar
class NewLineFormatter(logging.Formatter):

    def __init__(self, fmt, datefmt=None):
        """
        Init given the log line format and date format
        """
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        """
        Override format function
        """
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])

        return msg

DATEFORMAT = "%d-%m-%Y %H:%M:%S"
LOGFORMAT = "%(asctime)s %(levelname)-8s %(funcName)30s-%(filename)15s-%(lineno)-4s: %(message)s"
CONSOLEFORMAT = "golding_mso: %(levelname)-8s %(message)s"

log_formatter = NewLineFormatter(LOGFORMAT, datefmt=DATEFORMAT)
console_formatter = NewLineFormatter(CONSOLEFORMAT, datefmt=DATEFORMAT)

# Set up logging path for the package
logger = logging.getLogger(__name__)
has_log_folder = os.path.isdir(user_pkg_dir / "logs")
if not has_log_folder:
    # If the logs directory does not exist, create it
    os.makedirs(user_pkg_dir / "logs", exist_ok=True)

# Set up a console handler for real-time logging output
console_handler = logging.StreamHandler(stream=os.sys.stdout)
"""Console handler for logging output to standard output. Use console_handler.setLevel() to adjust verbosity."""
console_handler.setLevel(
    getattr(logging, user_config.get("general", {}).get("console_level", "INFO"))
)
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)


# Set up a timed rotating file handler for log rotation
rotate_handler = TimedRotatingFileHandler(
    user_pkg_dir / "logs" / "golding_mso.log",
    backupCount=2,  # Keep logs for 2 rotations
    when="h",  # Rotate logs every hour
    interval=8,  # Rotate every 8 hours
)
"""Timed rotating file handler for logging output to a file with rotation. Use rotate_handler.setLevel() or modify config file to adjust"""
rotate_handler.setLevel(user_config.get("general", {}).get("logging_level", "DEBUG"))
rotate_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(rotate_handler)