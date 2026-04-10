"""
This module provides classes and functions for creating and managing synaptic units (Exp2Syn, Netcon & Netstim)
in NEURON simulations, including random and path-based innervation of cell sections.
"""

import numpy as np
import random

from neuron import h
from .nrn_types import Section, Segment, Exp2Syn, NetCon, NetStim
from .cell import Cell
from .cell_calc import get_terminal_sections, get_parent_sections, section_list_length
from .math_calc import distance3D


class SynapseTerminal:
    """
    Represents a synaptic unit consisting of a Exp2Syn, NetStim, and NetCon on a given segment.
    Handles creation and management of synaptic parameters and firing probability.
    """

    def __init__(self, segment: Segment, **kwargs) -> None:
        r"""
        Initialize a SynapseTerminal on a given segment.
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
        self._start = kwargs.pop("start", 0)
        self.firing_times = kwargs.pop("firing_times", [0])
        self.fake_cell = kwargs.pop("fake_cell", h.VecStim())
        self._gmax = kwargs.pop("gmax", 0.037)
        # Create NEURON objects for this synapse unit
        self.syn = self.create_syn(**kwargs)
        self.netstim = self.create_netstim(**kwargs)
        self.netcon = self.create_netcon(**kwargs)
        
        # self.vesicle_pool = None  # Initialize vesicle pool attribute
        
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
        netstim.start = self._start if self.firing else -1
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

        self.netcon = h.NetCon(self.fake_cell, self.syn)
        self.netcon.weight[0] = kwargs.pop("gmax", 0.037)
        self.netcon.delay = kwargs.pop("delay", 0)
        self.set_firing_times()
        return self.netcon

    def set_firing_times(self, firing_times: list[float] = None) -> None:
        """
        Set specific firing times for the synapse unit and update NetCon to use VecStim.

        Parameters
        ----------
        firing_times : list[float]
            List of times at which the synapse should fire.
        """
        print('insuper')
        firing_times = firing_times if firing_times is not None else self.firing_times
        self.firing_times = firing_times
        firing_times = np.array(firing_times) + self._start
        self.fake_cell.play(h.Vector(firing_times.flatten()), 1)

    def clear_firing_times(self) -> None:
        """
        Clear any set firing times and reset to using NetStim for firing.
        """
        self.firing_times = None
        self.reset_firing()

    def reset_firing(self) -> None:
        """
        Reset the synapse unit to use NetStim for firing instead of VecStim.
        """
        gmax = self.netcon.weight[0]
        delay = self.netcon.delay
        self.netcon = h.NetCon(self.netstim, self.syn)
        self.netcon.weight[0] = gmax
        self.netcon.delay = delay
        self.firing_times = None

    def get_components(self) -> tuple[Exp2Syn, NetStim, NetCon]:
        """
        Get the NEURON objects that make up this synapse unit.

        Returns
        -------
        tuple[Exp2Syn, NetStim, NetCon]:
            The synapse, netstim, and netcon objects.
        """
        return self.syn, self.netstim, self.netcon

    def __repr__(self) -> str:
        """
        Return a string representation of the SynapseTerminal.
        """
        return f"Synapse @ {self.segment} on section {str(self.section)}"

    def destroy(self) -> None:
        """
        Destroy the synapse unit by deleting the synapse, netstim, and netcon objects.
        """
        del self.syn
        del self.netstim
        del self.netcon

    @property
    def start(self) -> float:
        return self._start

    @start.setter
    def start(self, value: float) -> None:
        self._start = value
        self.netstim.start = self._start if self.firing else -1
        self.set_firing_times(self.firing_times)

    @property
    def gmax(self) -> float:
        return self._gmax
    
    @gmax.setter
    def gmax(self, value: float) -> None:
        self._gmax = value
        self.netcon.weight[0] = self._gmax
class SynapseFiber:
    """Represents a fiber of synapses along a path of segments, allowing for coordinated management of multiple SynapseTerminals."""

    def __init__(self, synapse_terminals: list[SynapseTerminal], **kwargs) -> None:
        self.synapse_terminals = synapse_terminals
        self.shared_firing_times = kwargs.pop("firing_times", [-1])
        self.fake_cell = h.VecStim()
        if self.shared_firing_times is not None:
            self.set_firing_times(self.shared_firing_times)
            
    def set_firing_times(self, firing_times: list[float] = None) -> None:
        """
        Set specific firing times for the synapse unit and update NetCon to use VecStim.

        Parameters
        ----------
        firing_times : list[float]
            List of times at which the synapse should fire.
        """
        firing_times = firing_times if firing_times is not None else self.shared_firing_times
        self.shared_firing_times = firing_times
        for syn_terminal in self.synapse_terminals:
            syn_terminal.set_firing_times(firing_times)
        self.shared_firing_times = firing_times
        
    def clear_firing_times(self) -> None:
        """Clear firing times for all synapse terminals in the fiber."""
        for syn_terminal in self.synapse_terminals:
            syn_terminal.clear_firing_times()



class StochasticVesiclePool:
    """Represents a stochastic vesicle pool for synaptic release, modeling the dynamics of vesicle depletion and replenishment, according to Callan et al. 2021."""

    def __init__(self, rzero, p_release, tau_replenish):
        self.rzero = rzero
        self.p_release = p_release
        self.tau_replenish = tau_replenish
        self.r = rzero
        self.k = 0
        self.replenish_time = 0

    def release(self, time=None):
        """Simulate vesicle release at the current time, updating the pool state accordingly. If no time is provided, the pool replenishment is not calculated/updated."""
        self.k = np.random.binomial(self.r, self.p_release)
        self.r -= self.k
        print(f"Released {self.k} vesicles  @ {time}, {self.r}/{self.rzero} in pool.")
        if isinstance(time, (int, float)):
            if self.k > 0:
                self.replenish(time)
            else:
                self.replenish(time, release=False)  # No release, but still replenish based on time since last replenishment
        # print(f"After replenishment, {self.r}/{self.rzero} vesicles in pool.")
        return self.k

    def replenish(self, time, release=True):
        """Replenish the vesicle pool based on the elapsed time since the last replenishment."""
        isi = time - self.replenish_time
        self.replenish_time = time if release else self.replenish_time  # Only update replenish_time if a release occurred
        self.r = int(self.rzero + (self.r - self.rzero) * (
            np.exp(-isi / self.tau_replenish)
        ))

    def serial_release(self, time_points):
        """Generator that simulates vesicle release over a series of time points, yielding the number of vesicles released at each point."""
        self.reset()
        self.replenish_time = time_points[0]
        for time in time_points:
            yield self.release(time)

    def reset(self):
        """Reset the vesicle pool to its initial state."""
        self.r = self.rzero
        self.k = 0
        self.replenish_time = 0

class StochasticSynapseTerminal(SynapseTerminal):
    """A SynapseTerminal that incorporates a StochasticVesiclePool for probabilistic synaptic release."""

    def __init__(self, segment: Segment, **kwargs) -> None:
        self.vesicle_pool = StochasticVesiclePool(
            rzero=kwargs.pop("rzero", (118/4)),
            p_release=kwargs.pop("p_release", 0.45),
            tau_replenish=kwargs.pop("tau_replenish", 30),
        )
        super().__init__(segment, **kwargs)
        
    def _compute_pool_depletion(self, firing_times=None):
        if firing_times is None:
            firing_times = self.firing_times
        self.cond_values = np.array(list(self.vesicle_pool.serial_release(firing_times))) * self._gmax / self.vesicle_pool.rzero

    def _play_stochastic_conductances(self, firing_times=None):
        if firing_times is None:
            firing_times = self.firing_times
        self._tvec = h.Vector(firing_times)
        self._cvec = h.Vector(self.cond_values)
        self.cond_vec = self._cvec.play(self.netcon._ref_weight[0], self._tvec, 1)
        
    def set_firing_times(self, firing_times: list[float] = None) -> None:
        """Override to set firing times and configure the NetCon to use the stochastic vesicle pool."""
        self.firing_times = firing_times if firing_times is not None else self.firing_times
        self.firing_times = np.array(self.firing_times) + self._start
        # print(f"Setting firing times: {self.firing_times}")
        self.fake_cell.play(h.Vector(self.firing_times.flatten()))
        self._compute_pool_depletion(self.firing_times)
        self._play_stochastic_conductances(self.firing_times)


def lookup_syns_by_section(synapse_terminals: list[SynapseTerminal], section: Section) -> list[SynapseTerminal]:
    section_syns = []
    for syn in synapse_terminals:
        if syn.section == section:
            section_syns.append(syn)
    return section_syns

def innervate_total(section_list: list[Section], **kwargs) -> list[SynapseTerminal]:
    r"""
    Create SynapseTerminals for every segment in the provided list of sections.

    Parameters
    ----------
    section_list : list[Section]
        List of NEURON Section objects.
    \**kwargs:
        Passed to SynapseTerminal constructor.

    Returns
    -------
    list[SynapseTerminal]:
        List of created SynapseTerminal objects.
    """
    syn_units = innervate_points(
        *[seg for sec in section_list for seg in sec], **kwargs
    )
    return syn_units


def innervate_points(*segments: list[Segment], **kwargs) -> list[SynapseTerminal]:
    r"""
    Create SynapseTerminals for each provided segment.

    Parameters
    ----------
    \*segments : list[Segment]
        Segments to innervate.
    \**kwargs:
        Passed to SynapseTerminal constructor.

    Returns
    -------
    list[SynapseTerminal]:
        List of created SynapseTerminal objects.
    """
    units = []
    for segment in segments:
        if kwargs.get("stochastic", False):
            synapse_unit = StochasticSynapseTerminal(segment, **kwargs)
        else:
            synapse_unit = SynapseTerminal(segment, **kwargs)
        units.append(synapse_unit)
    return units


def innervate_random(
    cell: Cell,
    section_list: list[Section],
    numgroups: int = 1,
    numsyns: int = 4,
    synspace: float = 7,
    **kwargs,
) -> list[list[SynapseTerminal]]:
    r"""
    Randomly innervate a cell with groups of SynapseTerminals along random paths.

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
        Passed to SynapseTerminal constructor.

    Returns
    -------
    list[list[SynapseTerminal]]:
        List of groups, each a list of SynapseTerminal objects.
    """
    synapse_groups = []
    for group_num in range(numgroups):
        ends = list(get_terminal_sections(section_list))
        random_end = random.choice(ends)
        # Ensure the chosen path is long enough for the desired number of synapses
        while (
            section_list_length(
                cell, get_parent_sections(random_end, starts_from_soma=False)
            )[0]
            < numsyns * synspace
        ):
            random_end = random.choice(ends)
        chosen_path_sections = get_parent_sections(random_end, starts_from_soma=False)

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
        syn = SynapseTerminal(
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
