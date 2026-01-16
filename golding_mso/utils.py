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

def get_cell_file_paths(*filenames: str) -> list[str]:
    """
    Return a list[str] of filepaths to cell morphology files inside the package's 'cells' directory. If no filenames are provided, it returns all cell files included in the package directory.
    Usage:
    >>> from golding_mso.utils import get_cell_file_paths
    >>> from golding_mso.cell import Cell
    >>> cell_file_path = get_cell_file_paths("151124_03")[0]
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
    # If no filenames are provided, return all package cell files
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
    Load the compiled NEURON mechanism DLL from the package directory. Runs at package initialization.
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
            logger.error(f"Mechanism compilation failed: {e}")
            raise e
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

def get_mechanisms() -> dict:
    """
    Create a dictionary mapping mechanism names to their file paths in the package's 'mechanisms' directory.

    Returns
    -------
    dict:
        A dictionary where keys are mechanism names (file stems) and values are file paths.
    """
    mech_dir = get_package_path() / "mechanisms"
    mechanism_files = [
        str(file) for file in mech_dir.iterdir() if file.suffix in {".mod", ".hoc"}
    ]
    return {pathlib.Path(path).stem: path for path in mechanism_files}

def add_mechanism(mech_file_path: str):
    """
    Add a new mechanism file to the package's 'mechanisms' directory.

    Parameters
    ----------
    mech_file_path : str
        The file path of the mechanism file to add.
    """
    mech_dir = get_package_path() / "mechanisms"
    dest_path = mech_dir / pathlib.Path(mech_file_path).name
    copy(mech_file_path, dest_path)
    logger.info(f"Mechanism file {mech_file_path} copied to package @ {dest_path}")
    
def add_morphology(morph_file_path: str):
    """
    Add a new morphology file to the package's 'cells' directory.

    Parameters
    ----------
    morph_file_path : str
        The file path of the morphology file to add.
    """
    cell_dir = get_package_path() / "cells"
    dest_path = cell_dir / pathlib.Path(morph_file_path).name
    copy(morph_file_path, dest_path)
    logger.info(f"Morphology file {morph_file_path} copied to package @ {dest_path}")