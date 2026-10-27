
from .config import Config
import pathlib
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

user_config = Config(str(user_pkg_dir))
"""Current configuration (dict) loaded from the user's package config file"""

anf_spikes_dir = utils.get_package_path() / "anf_spikes" if (utils.get_package_path() / "anf_spikes").exists() else None
"""Path to the directory containing ANF spike time data, if it exists in the package."""