"""
GoldingMSO - A package for neuronal modeling and simulation

This package provides tools for compartmental MSO modeling using the NEURON simulator,
with specialized modules for handling cell morphologies, running simulations, and
analyzing data related to interaural time difference (ITD) processing.

"""

import logging
import os
import pathlib
import shutil
from importlib.resources import files

from logging.handlers import TimedRotatingFileHandler


from neuron import h

h.load_file("import3d.hoc")

# Import key modules for ease of access
from .cell import Cell
from .config import Config
from . import utils
from . import cell_calc
from . import math_calc
from . import data
from . import sims
from . import syns

pkg_mech_compiled = (get_package_path() / "mechanisms" / "x86_64").exists() or (
    get_package_path() / "mechanisms" / "nrnmech.dll"
).exists()
if pkg_mech_compiled is False:
    compile_mechs()
else:
    load_pkg_dll()
__version__ = "0.1.0"

# Load NEURON 3D import file


# Create library dictionary to access morphologies
morphologies = get_morphologies()


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
# Set up logging for the package
user_pkg_dir = pathlib.Path.home() / ".golding_mso"
logger = logging.getLogger(__name__)
has_log_folder = os.path.isdir(user_pkg_dir / "logs")
if not has_log_folder:
    # If the logs directory does not exist, create it
    os.makedirs(user_pkg_dir / "logs", exist_ok=True)
# Set up a console handler for real-time logging output

console_handler = logging.StreamHandler(stream=os.sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)
# Set up a timed rotating file handler for log rotation
rotate_handler = TimedRotatingFileHandler(
    user_pkg_dir / "logs" / "golding_mso.log",
    backupCount=2,  # Keep logs for 2 rotations
    when="h",  # Rotate logs every hour
    interval=8,  # Rotate every 8 hours
)
rotate_handler.setLevel(logging.DEBUG)
rotate_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(rotate_handler)

config = get_config()
console_handler.setLevel(
    getattr(logging, config.get("general", {}).get("logging_level", "INFO"))
)
