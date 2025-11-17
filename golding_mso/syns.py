"""
This module provides classes and functions for creating and managing synaptic units (Exp2Syn, Netcon & Netstim)
in NEURON simulations, including random and path-based innervation of cell sections.
"""

import numpy as np
import random

from neuron import h
from .nrn_types import Section, Segment, Exp2Syn, NetCon, NetStim
from .cell import Cell
from .cell_calc import tiplist, parentlist, section_list_length


class SynapseUnit:
    """
    Represents a synaptic unit consisting of a Exp2Syn, NetStim, and NetCon on a given segment.
    Handles creation and management of synaptic parameters and firing probability.
    """

    def __init__(self, segment: Segment, **kwargs) -> None:
        r"""
        Initialize a SynapseUnit on a given segment.
        A Synapse consists of an Exp2Syn synapse, a NetStim for stimulation, and a NetCon to connect them.

        Parameters
        ----------
        segment : Segment
            The NEURON segment to place the synapse on.
        \**kwargs:
            Optional parameters for synapse and stimulation properties.
            firing_probability (float): Probability the synapse will fire (default 1.0).
            start (float): Start time for NetStim (default 0).
            tau1, tau2, e, number, interval, noise, gmax, delay: Passed to respective NEURON objects.
        """
        self.section = segment.sec
        self.segment = segment
        self.firing_probability = kwargs.pop("firing_probability", 1.0)
        self.start = kwargs.pop("start", 0)
        # Create NEURON objects for this synapse unit
        self.syn = self.create_syn(**kwargs)
        self.netstim = self.create_netstim(**kwargs)
        self.netcon = self.create_netcon(**kwargs)

    def create_syn(self, **kwargs) -> Exp2Syn:
        r"""
        Create and configure an Exp2Syn synapse on the segment.

        Parameters
        ----------
        \**kwargs:
            tau1, tau2, e (optional synaptic parameters).

        Returns
        -------
        Exp2Syn:
            The created synapse object.
        """
        syn = h.Exp2Syn(self.segment)
        syn.tau1 = kwargs.pop("tau1", 0.29)
        syn.tau2 = kwargs.pop("tau2", 0.29)
        syn.e = kwargs.pop("e", 15)
        return syn

    def create_netstim(self, **kwargs) -> NetStim:
        r"""
        Create and configure a NetStim for this synapse.

        Parameters
        ----------
        \**kwargs:
            number, interval, noise (optional stimulation parameters).

        Returns
        -------
        NetStim:
            The created NetStim object.
        """
        netstim = h.NetStim()
        netstim.number = kwargs.pop("number", 1)
        netstim.interval = kwargs.pop("interval", 1)
        netstim.noise = kwargs.pop("noise", 0)
        # Determine if this synapse will fire based on probability
        self.firing = random.random() < self.firing_probability
        netstim.start = self.start if self.firing else -1
        return netstim

    def create_netcon(self, **kwargs) -> NetCon:
        r"""
        Create and configure a NetCon connecting NetStim to the synapse.

        Parameters
        ----------
        \**kwargs:
            gmax (weight), delay (optional connection parameters).

        Returns
        -------
        NetCon:
            The created NetCon object.
        """
        netcon = h.NetCon(self.netstim, self.syn)
        netcon.weight[0] = kwargs.pop("gmax", 0.037)
        netcon.delay = kwargs.pop("delay", 0)
        return netcon

    def reroll_firing(self, **kwargs) -> None:
        r"""
        Reroll the firing probability of the synapse unit and update NetStim start time.

        Parameters
        ----------
        \**kwargs:
            firing_probability (float, optional): New probability to use.
        """
        self.firing_probability = kwargs.pop(
            "firing_probability", self.firing_probability
        )
        self.firing = random.random() < self.firing_probability
        self.netstim.start = self.start if self.firing else -1

    def __repr__(self) -> str:
        """
        Return a string representation of the SynapseUnit.
        """
        return f"Synapse @ {self.segment} on section {str(self.section)}"

    def destroy(self) -> None:
        """
        Destroy the synapse unit by deleting the synapse, netstim, and netcon objects.
        """
        del self.syn
        del self.netstim
        del self.netcon


def innervate_total(section_list: list[Section], **kwargs) -> list[SynapseUnit]:
    r"""
    Create SynapseUnits for every segment in the provided list of sections.

    Parameters
    ----------
    section_list : list[Section]
        List of NEURON Section objects.
    \**kwargs:
        Passed to SynapseUnit constructor.

    Returns
    -------
    list[SynapseUnit]:
        List of created SynapseUnit objects.
    """
    syn_units = innervate_points(
        *[seg for sec in section_list for seg in sec], **kwargs
    )
    return syn_units


def innervate_points(*segments: list[Segment], **kwargs) -> list[SynapseUnit]:
    r"""
    Create SynapseUnits for each provided segment.

    Parameters
    ----------
    \*segments : list[Segment]
        Segments to innervate.
    \**kwargs:
        Passed to SynapseUnit constructor.

    Returns
    -------
    list[SynapseUnit]:
        List of created SynapseUnit objects.
    """
    units = []
    for segment in segments:
        synapse_unit = create_synunit(segment, **kwargs)
        units.append(synapse_unit)
    return units


def innervate_random(
    cell: Cell,
    section_list: list[Section],
    numgroups: int = 1,
    numsyns: int = 4,
    synspace: float = 7,
    **kwargs,
) -> list[list[SynapseUnit]]:
    r"""
    Randomly innervate a cell with groups of SynapseUnits along random paths.

    Parameters
    ----------
    cell : Cell
        The cell to innervate.
    section_list : list[Section]
        List of NEURON Section objects.
    numgroups : int
        Number of synapse groups to create.
    numsyns : int
        Number of synapses per group.
    synspace : float
        Minimum spacing between synapses.
    \**kwargs:
        Passed to SynapseUnit constructor.

    Returns
    -------
    list[list[SynapseUnit]]:
        List of groups, each a list of SynapseUnit objects.
    """
    synapse_groups = []
    for group_num in range(numgroups):
        ends = list(tiplist(section_list))
        random_end = random.choice(ends)
        # Ensure the chosen path is long enough for the desired number of synapses
        while (
            section_list_length(cell, parentlist(random_end, starts_from_soma=False))[0]
            < numsyns * synspace
        ):
            random_end = random.choice(ends)
        chosen_path_sections = parentlist(random_end, starts_from_soma=False)

        # Pick a random length along the path
        random_length_on_path = random.uniform(
            0, section_list_length(cell, chosen_path_sections)[0]
        )
        while random_length_on_path < numsyns * synspace:
            random_length_on_path = random.uniform(
                0, section_list_length(cell, chosen_path_sections)[0]
            )
        chosen_length_on_path = random_length_on_path
        chosen_path_lengths = section_list_length(
            cell, chosen_path_sections, return_array=True
        )[1]

        # Determine the locations for synapse placement
        lengths_to_place = np.linspace(
            chosen_length_on_path - (synspace * numsyns),
            chosen_length_on_path,
            numsyns,
            endpoint=True,
        )

        segments_for_placement = []
        for length in lengths_to_place:
            length_total = 0
            for section, section_length in zip(
                chosen_path_sections, chosen_path_lengths
            ):
                length_total += section_length
                if length_total > length:
                    # Find the segment at the correct location along the section
                    segment = section((length_total - length) / section.L)
                    segments_for_placement.append(segment)
                    break
        synapse_groups.append(innervate_points(*segments_for_placement, **kwargs))
    return synapse_groups


def syn_path_place(
    cell: Cell,
    section_list: list[Section],
    locations: list[float],
    tau1=0.270,
    tau2=0.271,
    e=15,
):

    lengtharray = []
    synlist = []
    for sec in section_list:  # record each length of each section
        if sec in cell.somatic:
            lengtharray.append(sec.L / 2)  # cutting soma in half, to keep on one side
        else:
            lengtharray.append(sec.L)

    for loc in locations:
        sectioncount = 0
        lengthcount = 0
        sectionindex = 0
        loconsec = 0

        # Move along the length of the path until section and relative location are found
        for length in lengtharray:

            if length + lengthcount > loc:
                sectionindex = sectioncount
                loconsec = loc - lengthcount
                break
            else:
                sectioncount += 1
                lengthcount += length
        syn = SynapseUnit(
            list(section_list)[sectionindex](
                loconsec / list(section_list)[sectionindex].L
            ),
            tau1=tau1,
            tau2=tau2,
            e=e,
        )
        # Old implementation. New is less tested
        # syn = h.Exp2Syn(
        #     list(section_list)[sectionindex](
        #         loconsec / list(section_list)[sectionindex].L
        #     )
        # )  # place syn
        # syn.tau1 = tau1
        # syn.tau2 = tau2
        # syn.e = e
        synlist.append(syn)

        syn = None

    return synlist


def create_synunit(segment: Segment, **kwargs) -> SynapseUnit:
    r"""
    Helper function to create a SynapseUnit on a given segment.

    Parameters
    ----------
    segment : Segment
        The NEURON segment to place the synapse on.
    \**kwargs:
        Passed to SynapseUnit constructor.

    Returns
    -------
    SynapseUnit:
        The created SynapseUnit object.
    """
    return SynapseUnit(segment, **kwargs)
