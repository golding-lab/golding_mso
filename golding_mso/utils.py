"""
This module provides utility functions for the package, such as file path helpers.
"""

import json
import logging
import os
import pathlib
import subprocess

from importlib.resources import path, files
from neuron import h
from shutil import copy, rmtree

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

user_pkg_dir = pathlib.Path.home() / ".golding_mso"


def get_cell_file_paths(*filenames: str) -> list[str]:
    """
    Return a list[str] of filepaths to cell morphology files inside the package's 'cells' directory. If no filenames are provided, it returns all cell files included in the package directory.
    Usage:
    >>> from golding_mso.utils import get_cell_file_paths
    >>> from golding_mso.cell import Cell
    >>> cell_file_path = get_cell_file_paths("example_cell.asc")
    >>> cell = Cell(cell_file_path)
    ...

    Parameters
    ----------
    filenames: str
        Optional variable number of filenames to search for in the package's 'cells' directory.

    Returns
    -------
    list:
        A list of file paths to the cell morphology files.
    """

    cell_files = []
    if not filenames:
        cell_files = [str(file) for file in (get_package_path() / "cells").iterdir()]
    else:
        for filename in filenames:
            if not isinstance(filename, str):
                raise TypeError(f"Filename ({filename}) must be a string.")
            files_with_name = [
                file_with_name
                for file_with_name in (get_package_path() / "cells").iterdir()
                if filename.lower() in file_with_name.name.lower()
            ]
            if len(files_with_name) > 1:
                raise FileNotFoundError(
                    f"Multiple cell files containing name {filename} found in package data. Please specify a unique filename."
                )
            if len(files_with_name) == 0:
                raise FileNotFoundError(
                    f"Cell file {filename} not found in package data."
                )
            cell_files.append(str(files_with_name[0]))
    return cell_files


def get_cell_dir_path() -> str:
    """
    Get the path to the cell morphology files directory.

    Returns
    -------
    str:
        The path to the 'cells' directory within the golding_mso package.
    """
    return get_package_path() / "cells"


def reset_config():
    """
    Reset the configuration of the golding_mso package to its default state.
    This function is useful for testing purposes or when you want to clear any custom configurations.
    """
    with path("golding_mso", "golding_mso_default_config.json") as dcp:
        def_config_path = dcp
    try:
        if not user_pkg_dir.is_dir():
            # Create the user config directory if it does not exist
            user_pkg_dir.mkdir(parents=True, exist_ok=True)
        copy(
            def_config_path,
            user_pkg_dir / "golding_mso_config.json",
        )
        copy(
            def_config_path,
            user_pkg_dir / "golding_mso_config.json",
        )
        logger.info("Default configuration files copied to user config directory.")
    except FileNotFoundError:
        logger.exception(
            "Default configuration file 'golding_mso_default_config.json' not found. "
            "You may need to reinstall the golding_mso package or replace the file manually.",
        )
    return get_config()


def get_config(config_path: str = None) -> dict:
    """
    Get the current configuration of the golding_mso package.

    Returns
    -------
    dict:
        The current configuration as a dictionary.

    Raises
    ------
    FileNotFoundError:
        If the configuration file 'golding_mso_config.json' does not exist in the user config path.
    """

    if config_path is None:
        config_path = user_pkg_dir / "golding_mso_config.json"
    with path("golding_mso", "golding_mso_default_config.json") as dcp:
        default_config_path = dcp
    if not os.path.exists(config_path):
        logger.error(
            "Configuration directory 'golding_mso' not found in user config path. Please run reset_config() to create the user's default configuration. Using default configuration instead."
        )
        config_path = default_config_path
    with open(
        config_path,
        "r",
    ) as f:
        config = json.load(f)
    with open(default_config_path, "r") as default_f:
        default_config = json.load(default_f)

    def compare_structure(a, b):
        for (a_setting, a_value), (b_setting, b_value) in zip(a.items(), b.items()):
            if not a_setting == b_setting:
                logger.warning(
                    f"Configuration setting '{a_setting}' not found in default configuration. Adding it with default value."
                )
                a[a_setting] = b_value
            if all(isinstance(x, dict) for x in (a_value, b_value)):
                compare_structure(a_value, b_value)

    compare_structure(config, default_config)

    return config


def set_config(config: dict, config_path: str = None):
    """
    Set the configuration of the golding_mso package.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    config_path : str, optional
        The path to save the configuration file. If None, saves to the default user config path.
    """
    if config_path is None:
        config_path = user_pkg_dir / "golding_mso_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


def get_package_path() -> pathlib.Path:
    """
    Get the path to the golding_mso package directory.

    Returns
    -------
    str:
        The path to the golding_mso package directory.
    """
    return files("golding_mso")


def load_dll(dll_path: str = ""):
    """
    Load a compiled NEURON mechanism DLL.

    Parameters
    ----------
    dll_path : str
        Path to the compiled NEURON mechanism DLL.
    """
    if os.path.isfile(dll_path):
        logger.info(f"Loading compiled mechanisms from {dll_path}")
        try:
            h.nrn_load_dll(str(dll_path))
        except Exception as e:
            logger.error(f"Error loading DLL: {e}")

    else:
        logger.error(f"Compiled mechanism library not found at {dll_path}")
        raise FileNotFoundError(f"Compiled mechanism library not found at {dll_path}")


def load_pkg_dll():
    """
    Load the compiled NEURON mechanism DLL from the package directory.
    """
    cwd = get_package_path() / "mechanisms"
    dll_paths = [("x86_64", "libnrnmech.dylib"), (r"nrnmech.dll",)]
    for dll_path in dll_paths:
        full_path = cwd.joinpath(*dll_path)
        if os.path.exists(full_path):
            logger.info(f"Loading compiled mechanism from {full_path}")
            try:
                load_dll(full_path)
            except Exception as e:
                logger.error(f"Failed to load DLL {full_path}: {e}")


def compile_mechs(
    mech_path: str = None, workdir: str = None, nrnivmodl_path: str = None
):
    """
    Compile NEURON mechanisms in the specified directory using nrnivmodl.

    Parameters
    ----------
    mech_path : str, optional
        Path to the directory containing NEURON mechanism files.
    workdir : str, optional
        Working directory where compilation will occur and load compiled mechanisms from.
    nrnivmodl_path : str, optional
        Path to the nrnivmodl executable.
    """
    logger.info("Starting mechanism compilation")
    # if mech_path is None:
    #     raise ValueError("mech_path not specified.")

    mechd = pathlib.Path(mech_path) if mech_path else get_package_path() / "mechanisms"
    cwd = (
        pathlib.Path(workdir)
        if workdir is not None
        else get_package_path() / "mechanisms"
    )
    nrnivmodl_path = nrnivmodl_path if nrnivmodl_path is not None else "nrnivmodl"

    logger.debug(f"Current working directory: {cwd}")
    logger.debug(f"Mechanism path: {mech_path}")
    if os.path.isdir(cwd / "x86_64"):
        os.rename(cwd / "x86_64", cwd / "x86_64_old")
        rmtree(cwd / "x86_64_old")
    try:
        logger.debug(
            f"Running nrnivmodl with command: {f'{nrnivmodl_path} {str(pathlib.Path(mech_path))}'}"
        )
        logger.debug(
            subprocess.run(
                f"{nrnivmodl_path} {mechd}",
                cwd=str(cwd),
                check=True,
                shell=True,
                capture_output=True,
            )
        )
    except:
        try:
            logger.error("Failed to run with shell = True.")
            subprocess.run(
                [nrnivmodl_path, str(mechd)],
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(e.stderr)
    dll_paths = ["/x86_64/libnrnmech.dylib", "/x86_64/libnrnmech.so", r"\nrnmech.dll"]
    for dll_path in dll_paths:
        full_path = cwd.joinpath(*dll_path.split("/"))
        if os.path.exists(full_path):
            logger.info(f"Loading compiled mechanism from {full_path}")
            try:
                load_dll(full_path)
            except Exception as e:
                logger.error(f"Failed to load DLL {full_path}: {e}")


def get_morphologies() -> dict:
    """
    Create a dictionary mapping morphology names to their file paths in the package's 'cells' directory.

    Returns
    -------
    dict:
        A dictionary where keys are morphology names (file stems) and values are file paths.
    """
    return {pathlib.Path(path).stem: path for path in get_cell_file_paths()}
