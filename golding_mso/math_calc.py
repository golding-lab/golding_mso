"""
This module provides functions for geometric and mathematical calculations.
"""

import math
from typing import overload


def get_line_from_points(
    point1: tuple[float, float], point2: tuple[float, float]
) -> tuple[float, float]:
    """
    Calculate the slope and y-intercept of a line given two points.

    Parameters
    ----------
    point1 : tuple
        Coordinates of the first point (x, y).
    point2 : tuple
        Coordinates of the second point (x, y).

    Returns
    -------
    m : float
        Slope of the line.
    b : float
        Y-intercept of the line.
    """
    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]
    m = ydiff / xdiff if xdiff != 0 else float("inf")  # slope
    b = point1[1] - m * point1[0]  # y-intercept
    return m, b


def dist_from_line(line: tuple[float, float], point: tuple[float, float]) -> float:
    """
    Calculates the distance from a point to a line defined by its slope and y-intercept.
    The line is defined in the x-y plane, and the point is a 2D point (x, y).

    Parameters
    ----------
    line : tuple
        Slope (m) and y-intercept (b) of the line.
    point : tuple
        Coordinates of the point (x, y).

    Returns
    -------
    distance : float
        Perpendicular distance from the point to the line.
    """
    x0, y0 = point
    m, b = line
    b_perp = y0 + m * x0
    m_perp = -m
    if m == float("inf"):
        # Vertical line case
        return abs(point[0] - b)
    elif m == 0:
        return abs(point[1] - b)

    else:
        # Distance from point (x0, y0) to line y = mx + b
        x1 = (m_perp * x0 + b_perp - b) / m
        y1 = m * x1 + b
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def define_xy_line(
    point1: tuple[float, float],
    point2: tuple[float, float],
    point3: tuple[float, float],
) -> tuple[float, float]:
    """
    Defines a line in 3D space using two points and a third point to determine the plane.
    Returns the slope (m) and y-intercept (b) of the line in the x-y plane.

    Parameters
    ----------
    point1 : tuple
        Coordinates of the first point (x, y, z).
    point2 : tuple
        Coordinates of the second point (x, y, z).
    point3 : tuple
        Coordinates of the third point (x, y, z) used to define the plane.

    Returns
    -------
    m : float
        Slope of the line in the x-y plane.
    b : float
        Y-intercept of the line in the x-y plane.
    """
    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]

    if xdiff == 0:
        m = -float("inf")  # vertical line
        b = None
    else:
        m = -xdiff / ydiff
        b = point3[1] - m * point3[0]

    return m, b


@overload
def perpendicular_distance_point_to_line(
    line: tuple[float, float],
    point: tuple[float, float],
) -> float:
    x0, y0 = point
    m, b = line
    x1, y1 = (0, b)
    x2, y2 = (1, m + b)


@overload
def perpendicular_distance_point_to_line(
    line: tuple[tuple[float, float], tuple[float, float]],
    point: tuple[float, float],
) -> float:
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]


def perpendicular_distance_point_to_line(line, point) -> float:
    """
    Calculate the perpendicular distance from a point to a line defined by two points.

    This function uses the cross product method for accurate distance calculation
    and handles all edge cases including vertical and horizontal lines.

    Parameters
    ----------
    point : tuple
        Coordinates of the point (x, y).
    line_point1 : tuple
        Coordinates of the first point defining the line (x, y).
    line_point2 : tuple
        Coordinates of the second point defining the line (x, y).

    Returns
    -------
    distance : float
        Perpendicular distance from the point to the line.

    Raises
    ------
    ValueError:
        If the two line points are identical (cannot define a line).
    """
    if (
        isinstance(line, tuple)
        and len(line) == 2
        and all(isinstance(pt, tuple) and len(pt) == 2 for pt in line)
    ):
        x0, y0 = point
        x1, y1 = line[0]
        x2, y2 = line[1]
    else:
        x0, y0 = point
        m, b = line
        x1, y1 = (0, b)
        x2, y2 = (1, m + b)
    # Check if line points are identical
    if x1 == x2 and y1 == y2:
        raise ValueError("Line points cannot be identical - unable to define a line")

    # Calculate perpendicular distance using cross product formula
    # |((x2-x1)(y1-y0) - (x1-x0)(y2-y1))| / sqrt((x2-x1)^2 + (y2-y1)^2)
    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return numerator / denominator


def perpendicular_distance_point_to_line_3d(
    point: tuple[float, float, float],
    line_point1: tuple[float, float, float],
    line_point2: tuple[float, float, float],
) -> float:
    """
    Calculate the perpendicular distance from a point to a line in 3D space.

    Uses vector cross product to find the shortest distance from a point to a line
    defined by two points in 3D space.

    Parameters
    ----------
    point : tuple
        Coordinates of the point (x, y, z).
    line_point1 : tuple
        Coordinates of the first point defining the line (x, y, z).
    line_point2 : tuple
        Coordinates of the second point defining the line (x, y, z).

    Returns
    -------
    distance : float
        Perpendicular distance from the point to the line in 3D space.

    Raises
    ------
    ValueError:
        If the two line points are identical (cannot define a line).
    """
    x0, y0, z0 = point
    x1, y1, z1 = line_point1
    x2, y2, z2 = line_point2

    # Check if line points are identical
    if x1 == x2 and y1 == y2 and z1 == z2:
        raise ValueError("Line points cannot be identical - unable to define a line")

    # Vector from line_point1 to line_point2 (direction vector)
    d = (x2 - x1, y2 - y1, z2 - z1)

    # Vector from line_point1 to point
    w = (x0 - x1, y0 - y1, z0 - z1)

    # Cross product w × d
    cross_product = (
        w[1] * d[2] - w[2] * d[1],  # i component
        w[2] * d[0] - w[0] * d[2],  # j component
        w[0] * d[1] - w[1] * d[0],  # k component
    )

    # Magnitude of cross product
    cross_magnitude = math.sqrt(
        cross_product[0] ** 2 + cross_product[1] ** 2 + cross_product[2] ** 2
    )

    # Magnitude of direction vector
    d_magnitude = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

    return cross_magnitude / d_magnitude


def distance3D(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """
    Calculates the Euclidean distance between two 3D points.

    Parameters
    ----------
    a : tuple
        Coordinates of the first point (x, y, z).
    b : tuple
        Coordinates of the second point (x, y, z).

    Returns
    -------
    distance : float
        Euclidean distance between the two points.
    """
    distance = abs(
        math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2))
    )
    return distance
