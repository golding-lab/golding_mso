"""
This module provides data analysis utilities for ITD and synapse test results, including averaging and curve fitting functions.
"""

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


def get_itd_averages(
    filepath: str = None,
    df: pd.DataFrame = None,
    parameters: list[str] = [],
    delay_values: np.ndarray = np.arange(-0.5, 0.51, 0.01),
) -> pd.DataFrame:
    """
    Calculate ITD averages from a CSV file,
    where columns are structured [name, parameters..., delay_values]
    and each row is a separate ITD sweep.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing ITD data.
    parameters : list, optional
        List of parameter names to group by.
    delay_values : np.ndarray, optional
        Array of delay values.

    Returns
    -------
    itd_averages_pd : pd.DataFrame
        DataFrame containing averaged ITD data.
    """
    logger.info("Calculating ITD averages from file: %s", filepath)
    if df is not None:
        itd_df = df
    elif filepath is not None:
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        itd_df = pd.read_csv(filepath)
    else:
        raise ValueError("Either 'filepath' or 'df' must be provided.")
    names = pd.unique(itd_df.get("name"))
    parameter_vals = {}
    sort_by = ["names"]
    if len(parameters) != 0:
        for parameter in parameters:
            parameter_vals[parameter] = pd.unique(itd_df.get(parameter))
            sort_by.append(parameter)
    itd_averages_pd = pd.DataFrame()
    names_to_keep = []
    parameters_to_keep = []
    drop_columns = [
        column
        for column in itd_df.columns[: -len(delay_values) - 1]
        if ((column not in parameters) and column != "name")
    ]
    itd_df = itd_df.drop(columns=drop_columns)
    for name in names:
        if parameters:
            combinations = list(product(*list(parameter_vals.values())))
            for combination in combinations:
                cell_itds_df = itd_df.loc[(itd_df["name"] == name)]
                for parameter, parameter_val in zip(parameters, combination):
                    cell_itds_df = cell_itds_df.loc[
                        cell_itds_df[parameter] == parameter_val
                    ]

                average_cell_itd = (
                    cell_itds_df.mean(axis=0, numeric_only=True).to_frame().T
                )
                names_to_keep.append(name)
                itd_averages_pd = pd.concat([itd_averages_pd, average_cell_itd], axis=0)
        else:
            cell_itds_df = itd_df.loc[itd_df["name"] == name]
            average_cell_itd = cell_itds_df.mean(axis=0, numeric_only=True).to_frame().T
            names_to_keep.append(name)
            itd_averages_pd = pd.concat(
                [
                    itd_averages_pd,
                    average_cell_itd,
                ],
                axis=0,
            )
    itd_averages_pd.insert(0, "name", names_to_keep)
    logger.info("Completed ITD averages calculation")
    return itd_averages_pd


def get_syntest_averages(filepath: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Get average synapse data from a given file path.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing synapse test data.

    Returns
    -------
    averaged_groupsyn_pd : pd.DataFrame
        DataFrame containing averaged synapse test data.
    """
    logger.info("Calculating synapse test averages from file: %s", filepath)
    groupsyn_sheet = filepath
    groupsyn_data = pd.read_csv(groupsyn_sheet) if df is None else df
    groupsyn_data.sort_values("tau", inplace=True)
    param_label = groupsyn_data.columns[2]
    groupsyn_np = groupsyn_data.to_numpy()
    averaged_groupsyn_pd = pd.DataFrame()

    cell_names = np.unique(groupsyn_np[:, 1])
    param_vals = np.unique(groupsyn_np[:, 2])

    for cell_name in cell_names:
        for param_val in param_vals:
            trial_num = 0
            summed_groupsyn_data = np.zeros(len(groupsyn_np[0, 3:]))
            trial_exists = False
            for row in groupsyn_np:

                if cell_name in row and param_val in row:
                    summed_groupsyn_data += np.array(row[3:], dtype=float)
                    trial_exists = True
                    trial_num += 1
            if trial_exists:
                averaged_groupsyn_data = summed_groupsyn_data / trial_num
                dataframe_row = {
                    "Cell name": cell_name,
                    param_label: param_val,
                }
                logger.info("Calculated averaged data:%s", averaged_groupsyn_data)
                for i in range(len(averaged_groupsyn_data)):
                    dataframe_row[groupsyn_data.columns[i + 3]] = (
                        averaged_groupsyn_data[i]
                    )
                temp_df = pd.DataFrame(dataframe_row, index=[0])
                averaged_groupsyn_pd = pd.concat((averaged_groupsyn_pd, temp_df))
            else:
                continue
    logger.info("Completed synapse test averages calculation")
    return averaged_groupsyn_pd


def fit_gaussian_to_itd(
    delay_values: list[float], probabilities: list[float]
) -> np.ndarray:
    """
    Fit a Gaussian curve to ITD data.

    Parameters
    ----------
    delay_values : np.ndarray
        Array of delay values.
    probabilities : np.ndarray
        Array of probabilities.

    Returns
    -------
    y_fit : np.ndarray
        Fitted Gaussian curve values.
    """
    logger.info("Fitting Gaussian to ITD data")
    x = delay_values
    y = probabilities

    if sum(y) == 0:
        logger.warning("Sum of probabilities is zero, returning zeros")
        return np.zeros_like(x)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    popt, pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    y_fit = Gauss(x, *popt)
    equation = popt
    return y_fit


def find_centroid(data: list[float], delay_values: list[float]) -> float:
    """
    Find the centroid of a distribution.

    Parameters
    ----------
    data : np.ndarray
        Array of data values.
    delay_values : np.ndarray
        Array of delay values.

    Returns
    -------
    centroid : float or None
        Centroid value or None if not computable.
    """

    if data[0] == data[-1]:
        cutoff_height = data[0] if data[0] > data[-1] else data[-1]
    else:
        cutoff_height = 0
    total = np.sum(data[data >= cutoff_height])
    if total == 0:
        return None
    return (
        np.sum(data[data > cutoff_height] * delay_values[data > cutoff_height]) / total
    )
