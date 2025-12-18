"""
This module provides functions for analysis of NEURON cell morphologies, such as path length, surface area, and terminal sections, etc.
"""

import logging
import math
import neuron
import numpy as np
from neuron import h
from .math_calc import distance3D, dist_from_line, define_xy_line
from .nrn_types import Section, Segment, Exp2Syn, NetCon, NetStim
from .cell import Cell

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")


def tiplist(section_list: list[Section]) -> list[Section]:
    """
    Identifies all terminal sections in a given list.

    Parameters
    ----------
    section_list : list
        List of NEURON sections to analyze.

    Returns
    -------
    end_sections : list
        List of terminal sections (sections with no children).
    """
    section_list = list(section_list)
    end_sections = []
    for sec in section_list:
        section_ref = h.SectionRef(sec=sec)
        # subtree = 1 -> no children
        if len(sec.subtree()) == 1:
            end_sections.append(sec)
        else:
            # check for non-absolute end sections (like omitting filopodia)
            if not any(child in section_list for child in section_ref.child):
                end_sections.append(sec)
    return end_sections


def parentlist(
    section: Section,
    starts_from_soma: bool = True,
    include_soma: bool = False,
    include_section: bool = True,
) -> list[Section]:
    """
    Retrieves all parent sections of a given section.

    Parameters
    ----------
    section : h.Section
        The NEURON section to analyze.
    starts_from_soma : bool, optional
        If True, the list starts from the soma. Default is True.
    include_soma : bool, optional
        If True, includes the soma in the list. Default is False.
    include_section : bool, optional
        If True, includes the original section in the list. Default is True.

    Returns
    -------
    parent_list : SectionList
        List of parent sections, including the original section.
    """

    parent_list = h.SectionList()
    sectionref = h.SectionRef(section)
    first = True
    while sectionref.has_parent():
        if first and include_section:
            parent_list.append(sectionref.sec)
            first = False
        elif not first:
            parent_list.append(sectionref.sec)
        sectionref = h.SectionRef(sectionref.parent)
    if include_soma:
        parent_list.append(sectionref.root)
    if starts_from_soma:
        # Reverse to start from soma
        reversed_list = h.SectionList()
        for sec in reversed(list(parent_list)):
            reversed_list.append(sec)
        parent_list = reversed_list
    return parent_list


def section_list_length(
    cell: Cell, section_list: list[Section], return_array: bool = True
) -> tuple[float, list[float]]:
    """
    Calculates the total and individual lengths of sections in a list.

    Parameters
    ----------
    cell : Cell
        The cell instance to check for somatic sections.
    section_list : SectionList
        List of NEURON sections to analyze.
    return_array : bool, optional
        If True, returns an array of individual section lengths. Default is True.

    Returns
    -------
    total_length : float
        Total length of all sections in the list.
    length_array : list, optional
        Array of individual section lengths (if `return_array` is True).
    """

    total_length = 0
    length_array = []
    for sec in section_list:

        # halves length to measure from soma center
        if sec in cell.somatic:
            total_length += sec.L / 2
            length_array.append(sec.L / 2)
        else:
            total_length += sec.L
            length_array.append(sec.L)

    if return_array:
        return total_length, length_array

    return total_length


def furthest_point(cell: Cell, section_list: list[Section]) -> tuple[float, Segment]:
    """
    Finds the furthest segment and its distance from the soma.

    Parameters
    ----------
    cell : Cell
        The cell instance to find the somatic compartment.
    section_list : SectionList
        List of NEURON sections to analyze.

    Returns
    -------
    furthest_distance : float
        Distance of the furthest segment from the soma.
    furthest_seg : Segment
        The furthest segment object.
    """
    furthest_seg = None
    furthest_distance = 0
    for sec in section_list:

        for seg in sec:
            seg_distance = distance3D(getsegxyz(cell.somatic[0](0.5)), getsegxyz(seg))
            if seg_distance > furthest_distance:
                furthest_distance = seg_distance
                furthest_seg = seg
    return furthest_distance, furthest_seg


def surface_area(section_list: list[Section]) -> float:
    """
    Calculates the total surface area of all sections in a list.

    Parameters
    ----------
    section_list : SectionList
        List of NEURON sections to analyze.

    Returns
    -------
    area : float
        Total surface area of all sections.
    """
    area = 0
    for sec in section_list:
        for seg in sec:
            area += seg.area()
    return area


def getsegxyz(seg: Segment) -> tuple[float, float, float]:
    """
    Retrieves the 3D coordinates of a NEURON segment.

    Parameters
    ----------
    seg : Segment
        The NEURON segment to analyze.

    Returns
    -------
    coords : tuple
        3D coordinates of the segment (x, y, z).
    """

    seg_location = seg.x  # 0-1 val

    section = seg.sec
    xyz_num = section.n3d()
    if xyz_num == 0:
        raise ValueError(
            "Section has no 3D points. Ensure the section is properly initialized."
        )
    xyz_point = int(xyz_num * seg_location)  # finding nearest x,y,z measurement to seg

    if xyz_point == xyz_num:
        xyz_point = xyz_num - 1
    x = section.x3d(xyz_point)
    y = section.y3d(xyz_point)
    z = section.z3d(xyz_point)
    coords = (x, y, z)
    return coords


def closest_tip(section_list: list[Section], seg: Segment) -> tuple[float, Segment]:
    """
    Returns the closest tip section's terminal segment to a given segment in a section list.
    
    Parameters
    ----------
    section_list : SectionList
        List of NEURON sections to consider in distance search.
    seg : Segment
        The NEURON segment to find the closest tip to.
    
    Returns
    -------
    closest_distance : float
        Distance to the closest terminal segment.
    closest_endseg : Segment
        The closest terminal segment object.
    """
    endsecs = list(tiplist(section_list))
    if len(endsecs) == 0:
        raise ValueError("No terminal sections found in the provided section list.")
    closest_endsec = endsecs[0]
    try:
        closest_endsec_dist = distance3D(
            getsegxyz(seg), getsegxyz(closest_endsec(0.999999))
        )
    except:
        closest_endsec_dist = float("inf")
    for endsec in endsecs:
        path_secs = parentlist(endsec)
        if seg.sec in path_secs:
            try:
                endsec_dist = distance3D(getsegxyz(seg), getsegxyz(endsec(0.999999)))
            except:
                endsec_dist = float("inf")
            if endsec_dist < closest_endsec_dist:
                closest_endsec = endsec
                closest_endsec_dist = endsec_dist
    return closest_endsec_dist, closest_endsec(1)


def get_children_secs(
    sec: Section, section_list: list[Section] = None
) -> list[Section]:
    """
    Returns a list of all child sections in the cell (recursively).
    
    Parameters
    ----------
    sec : Section
        The NEURON section to find children of.
    section_list : SectionList, optional
        If provided, only returns children within this list. Default is None.
    
    Returns
    -------
    all_children : list
        List of all child sections.
    """

    children = list(sec.children())
    all_children = list(children)
    for child in children:
        all_children.extend(get_children_secs(child, section_list=section_list))
    if section_list is not None:
        all_children = [child for child in all_children if child in section_list]

    return all_children


def average_path_length(section_list: list[Section]) -> float:
    """
    Finds the average pathlength from all paths to the soma in the section list

    Parameters
    ----------
    section_list: SectionList
        list of sections to find paths to the soma within

    Returns
    -------
    average_path_length : float
    """
    total_path_length = 0
    total_path_length = 0
    end_secs = tiplist(section_list)
    for sec in end_secs:
        for sec in parentlist(sec):
            length = sec.L
            total_path_length += length
    if len(end_secs) == 0:
        raise ValueError(
            "No terminal sections found in the provided section list. Cannot compute average path length."
        )
    average_path_length = total_path_length / (len(list(end_secs)))
    return average_path_length


def mep(cell: Cell, section_list: list[Section]) -> float:
    """
    Calculates the mean electrotonic pathlength (MEP).
    Uses paths to soma from each section in section_list.

    MEP calculation based on (van Elburg and van Ooyen, 2010).

    Parameters
    ----------
    cell: Cell
        cell instance to pass along to pj()
    section_list: SectionList
        sections considered as 'endpoints' of a pathlength

    Returns
    -------
    mep : float
        calculated MEP value

    """
    sections = section_list
    pjarray = np.zeros(len(list(sections)))
    section_count = 0

    def pj(cell: Cell, section_list: list[Section]) -> float:
        """
        Returns sum of electrotonic pathlengths. Used for MEP calculation.
        MEP calculation based on (van Elburg and van Ooyen, 2010).

        Parameters
        ----------
        cell : Cell
            instance of a Cell to pull membrane resistance from
        section_list: SectionList
            list of sections to calculate electrotonic pathlength

        Returns
        -------
        pj : float
            sum of electrotonic pathlengths
        """
        # array to store electrotonic pathlengths
        electrolength_array = np.zeros(len(list(section_list)))
        sectionCount = 0
        for sec in section_list:

            rm = cell.membrane_resistance  # cm^2 * Ω
            bi = (sec.diam / 2) / 10000  # cm
            numerator = bi * rm  # cm^3 * Ω
            denominator = 2 * sec.Ra  # Ω * cm
            tosqrt = numerator / denominator  # cm^2
            length_const = math.sqrt(tosqrt)  # cm

            length = sec.L / 10000  # cm
            electrolength = length / length_const

            electrolength_array[sectionCount] = electrolength
            sectionCount += 1

        pj = np.sum(electrolength_array)
        return pj

    for sec in sections:

        sec_parentlist = parentlist(sec)
        pjarray[section_count] = pj(cell, sec_parentlist)
        section_count += 1

    mep = np.sum(pjarray) / len(pjarray)
    return mep


def axon_length_to_terminal(
    cell: Cell, seg: Segment, section_list: list[Section], method="plane"
):
    """
    Calculates the length of the axon to the terminal segment.

    Parameters
    ----------
    cell : Cell
        The cell instance containing the sections.
    seg : Segment
        The segment object for which to calculate the axon length to the terminal.
    section_list : SectionList
        List of NEURON sections to analyze.
    method : str, optional
        The method to use for calculation. Options are 'plane', 'cartesian', or 'straight'. Default is 'plane'.
        plane: Uses a defined distal perpendicular plane based on cell orientation and furthest segment.
        cartesian: Calculates the distance in Cartesian coordinates, assuming axon travels along one axis at a time.
        straight: Calculates the straight-line distance to the terminal segment, assuming a direct path.

    Returns
    -------
    axon_length_to_terminal : float
        The calculated axon length to the terminal segment.
    closest_endseg : Segment
        The closest terminal segment object.
    """
    furthest_dist, furthest_seg = furthest_point(cell, section_list)
    if method == "plane":
        distal_line = define_xy_line(
            *get_average_endpoints(cell),
            getsegxyz(furthest_seg),
        )
        children = (
            get_children_secs(seg.sec)
            if get_children_secs(seg.sec, section_list=section_list)
            else [seg.sec]
        )
        closest_dist, closest_endseg = closest_tip(children, seg)
        axon_length_to_terminal = dist_from_line(
            distal_line, getsegxyz(closest_endseg)[0:2]
        )

    elif method == "cartesian":
        closest_dist, closest_endseg = closest_tip(
            get_children_secs(seg.sec, section_list=section_list), seg
        )
        furthest_x, furthest_y, furthest_z = getsegxyz(furthest_seg)
        closest_x, closest_y, closest_z = getsegxyz(closest_endseg)
        # Axon trajectories prior to reaching a terminal segment are assumed
        # to only travel along one of the three axes at once
        axon_length_to_terminal = (
            abs(closest_x - furthest_x)
            + abs(closest_y - furthest_y)
            + abs(closest_z - furthest_z)
        )

    elif method == "straight":
        closest_dist, closest_endseg = closest_tip(
            get_children_secs(seg.sec, section_list=section_list), seg
        )
        furthest_x, furthest_y, furthest_z = getsegxyz(furthest_seg)
        axon_length_to_terminal = distance3D(getsegxyz(seg), getsegxyz(closest_endseg))

    else:
        raise ValueError(
            "Invalid method for axon length calculation. Choose 'plane', 'cartesian', or 'straight'."
        )

    return axon_length_to_terminal, closest_endseg


def axon_length_along(seg_1: Segment, seg_2: Segment) -> float:
    """
    Calculates the length of the axon segment between two given segments. Equivalent to h.distance(seg_1, seg_2).

    Parameters
    ----------
    seg_1 : Segment
        The starting segment of the axon path.
    seg_2 : Segment
        The ending segment of the axon path.

    Returns
    -------
    axon_pathlength : float
        The calculated length of the axon segment between the two given segments.
    """
    axon_pathlength = h.distance(seg_1, seg_2)
    if axon_pathlength == 1e20:
        logger.warning("Path between segments not found, returning 1e20")
    return axon_pathlength


def get_input_lengths(cell: Cell, section_lists: list[list[Section]], **kwargs):
    """
    Calculates the total length of all input paths in a cell.

    Parameters
    ----------
    cell : Cell
        The cell instance containing the sections.
    section_lists : list of lists of Section, optional
        A list of sections to consider. If None, uses all sections in the cell.

    Returns
    -------
    dict
        A dictionary mapping each segment to the total length of its input path.
    """
    input_lengths = {}
    for section_list in section_lists:
        for sec in section_list:
            for seg in sec:
                length_to_term, end_seg = axon_length_to_terminal(
                    cell,
                    seg,
                    section_list=section_list,
                    method=kwargs.get("method", "plane"),
                )
                length_along = axon_length_along(seg, end_seg)
                input_lengths[seg] = length_to_term + length_along
    return input_lengths


def get_average_endpoints(
    cell: Cell,
    sectionlist_1: list[Section] = None,
    sectionlist_2: list[Section] = None,
):
    """
    Calculates the average coordinates of the terminal segments at the ends of the cell for use defining line parallel to cell orientation.

    Parameters
    ----------
    cell : Cell
        The cell instance containing the sections.
    sectionlist_1 : SectionList, optional
        A list of sections to consider for the first set of terminal segments. If None, uses lateral_nofilopodia.
    sectionlist_2 : SectionList, optional
        A list of sections to consider for the second set of terminal segments. If None, uses medial_nofilopodia.

    Returns
    -------
    tuple
        Two tuples containing the average coordinates (x, y) of the lateral and medial terminal segments, respectively.
    """
    ends_1 = list(
        tiplist(cell.lateral_nofilopodia if sectionlist_1 is None else sectionlist_1)
    )
    x_1 = 0
    y_1 = 0
    for end in ends_1:
        x, y, z = getsegxyz(end(0.999))
        x_1 += x
        y_1 += y
    x_1 /= len(ends_1)
    y_1 /= len(ends_1)
    x_2 = 0
    y_2 = 0
    ends_2 = list(
        tiplist(cell.medial_nofilopodia if sectionlist_2 is None else sectionlist_2)
    )
    for end in ends_2:
        x, y, z = getsegxyz(end(0.999))
        x_2 += x
        y_2 += y
    x_2 /= len(ends_2)
    y_2 /= len(ends_2)
    return (x_1, y_1), (x_2, y_2)


def get_all_input_lengths(cell, section_lists, **kwargs):
    """
    Calculates the total length of all input paths in a cell.

    Parameters
    ----------
    cell : Cell
        The cell instance containing the sections.
    section_lists : list of lists of Section, optional
        A list of lists of sections to consider. If None, uses all sections in the cell.

    Returns
    -------
    dict
        A dictionary mapping each segment to the total length of its input path.
    """
    input_lengths = {}

    for section_list in section_lists:
        for sec in section_list:
            for seg in sec:
                length_to_term, end_seg = axon_length_to_terminal(
                    cell,
                    seg,
                    section_list=section_list,
                    method=kwargs.pop("method", "plane"),
                )
                length_along = axon_length_along(seg, end_seg)
                input_lengths[seg] = length_to_term + length_along
    return input_lengths
