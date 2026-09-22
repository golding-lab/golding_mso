"""
Module for running simulations (propagation, ITD, etc.) on MSO models.
"""

import logging
import math
import numpy as np
import random
from .nrn_types import Section, Segment, Exp2Syn
from collections.abc import Callable
from neuron import h
from .cell import Cell
from .cell_calc import (
    get_terminal_sections,
    get_parent_sections,
    getsegxyz,
    get_all_input_lengths,
    furthest_point,
    section_list_length,
    distance3D,
    axon_length_along,
    axon_length_to_terminal,
)
from .syns import innervate_total, innervate_random, innervate_points, syn_path_place

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def propagation_test(
    cell: Cell, section_list: list[Section], **kwargs
) -> list[dict[str, dict[str, list]]]:
    """
    Tests synapses at every segment and returns time at max depolarization.

    Parameters
    ----------
    cell: Cell
        Cell instance to pass to syntest_max_voltage
    section_list: list[Section]
        sectionlist (from Cell instance) to place synapses on

    Returns
    -------
    list[dict[str, dict[str, list[float]]]]
        List of dictionaries containing max voltage and time data for each section.
    """
    logger.debug("Starting propagation_test")
    py_sections = list(section_list)  # convert SectionList() to python list for len()
    propagation_data = []
    for section_index, section in enumerate(section_list):
        # Progress bar update placeholder
        logger.debug(
            "Processing section: %s, %s of %s",
            section.name(),
            section_index + 1,
            len(py_sections),
        )
        synlist = innervate_total([section])  # makes list of synapses for synlist_test
        max_data = indiv_syn_test(cell, synlist, cell.somatic[0], **kwargs)
        propagation_data.append(max_data)
    # Adds each max voltage and time value to created 2d list: [section][seg time/voltage]
    return propagation_data


def compute_propagation_test_difference(
    rec_site: dict[str, dict[str, float]],
    synapse: dict[str, dict[str, float]],
    func: Callable,
    which: str = "voltage",
    **kwargs,
) -> list:
    r"""
    Compute the difference between rec_site and synapse data using a provided function.

    Parameters
    ----------
    rec_site : dict
        Dictionary containing 'maxv', 'maxt', and 'restv' for the recording site.
    synapse : dict
        Dictionary containing 'maxv', 'maxt', and 'restv' for the synapse.
    func : function
        Function to apply to the max voltage or time values.
    which : str, optional
        Specifies whether to compute for 'voltage' or 'time'. Default is 'voltage'.
    \**kwargs : dict
        Additional keyword arguments to pass to the function.

    Returns
    -------
    section_data: list
        List of computed values for section based on the provided function.
    """
    if which == "voltage":
        key = "maxv"
    elif which == "time":
        key = "maxt"
    else:
        raise ValueError("Invalid value for 'which'. Use 'voltage' or 'time'.")

    section_data = []
    for idx in range(len(rec_site[key])):
        section_data.append(func(rec_site[key][idx], synapse[key][idx], **kwargs))
    return section_data


def find_compensated_cond(
    cell: Cell,
    seg: Segment,
    maxv: float,
    desired_maxv: float,
    original_gmax: float = 0.005,
    tolerance: float = 0.1,
    resting_potential: float = -61.7,
    recursion_limit: int = 20,
    recursion_count: int = 0,
) -> float:
    """
    Recursively adjusts synaptic conductance (gmax) to achieve a desired max voltage.

    Parameters
    ----------
    cell : Cell
        The cell instance to use for the simulation.
    seg : Segment
        The segment where the synapse is located.
    maxv : float
        The maximum voltage recorded from the initial simulation.
    desired_maxv : float
        The target maximum voltage to achieve.
    original_gmax : float, optional
        The initial conductance value to start with.
        Default is 0.005.
    tolerance : float, optional
        The acceptable error range for the maximum voltage.
        Default is 0.1.
    resting_potential : float, optional
        The resting potential of the cell.
        Default is -61.7.
    recursion_limit : int, optional
        The maximum number of recursive calls allowed.
        Default is 20.
    recursion_count : int, optional
        The current recursion depth.
        Default is 0.

    Returns
    -------
    new_gmax: float
        The adjusted conductance value that achieves the desired maximum voltage.
    """
    percent_change = (desired_maxv - resting_potential) / (maxv - resting_potential)
    new_gmax = original_gmax * (percent_change)
    synapse_list = innervate_points(seg)
    data = indiv_syn_test(cell, [synapse_list[0]], cell.somatic[0], gmax=new_gmax)
    error = abs(desired_maxv - data["rec_site"]["maxv"][0])
    if error > tolerance:
        logger.debug(f"Conductance ({new_gmax}) off by {error}")
        if recursion_count > recursion_limit:
            logger.warning(f"Recursion limit reached. Ending with gmax: {new_gmax}")
            for syn in synapse_list:
                syn.destroy()
            return new_gmax
        for syn in synapse_list:
            syn.destroy()
        return find_compensated_cond(
            cell,
            seg,
            data["rec_site"]["maxv"][0],
            desired_maxv,
            original_gmax=new_gmax,
            recursion_count=recursion_count + 1,
            tolerance=tolerance,
            recursion_limit=recursion_limit,
            resting_potential=data["rec_site"]["restv"][0],
        )
    else:
        logger.info(f"Found adjusted conductance: {new_gmax}\nOff by:{error}")
        for syn in synapse_list:
            syn.destroy()
        return new_gmax


def indiv_syn_test(
    cell: Cell, synapse_list: list[Exp2Syn], rec_section: Section, gmax: float = 0.01
) -> dict[str, dict[str, list]]:
    """
    Activates each synapse individually in a given list and, for each synapse, records the
    max voltages and time at those voltages at the recording site and the synapse site

    Parameters
    ----------
    cell: Cell
        Cell instance to pull stabilization_time variable to use.
    synapse_list: list[Exp2Syn]
        list of synapse NEURON point processes to activate.
    rec_section: Section
        Section to record from for the non-synapse compartment measurements.
        Records from center of given section.

    Returns
    -------
    recordings: dict
        Dictionary with keys 'rec_site' and 'syn', each containing dicts for 'maxv', 'maxt', and 'restv'.
    """
    syncount = len(synapse_list)
    max_data: dict[str, dict[str, list]] = {
        "rec_site": {"maxv": [], "maxt": [], "restv": []},
        "syn": {"maxv": [], "maxt": [], "restv": []},
    }
    for i, syn in enumerate(synapse_list):
        logger.debug("Activating synapse %d of %d", i + 1, syncount)
        syn.netstim.start = cell.stabilization_time
        syn.netcon.weight[0] = gmax
        syn_v = h.Vector()
        rec_site_v = h.Vector()
        t_syn = h.Vector()
        t_rec_site = h.Vector()
        cell.cvode.record(
            rec_section(0.5)._ref_v, rec_site_v, t_rec_site, sec=rec_section
        )
        cell.cvode.record(
            syn.section(syn.segment.x)._ref_v, syn_v, t_syn, sec=syn.section
        )
        h.finitialize()
        h.dt = 1
        h.continuerun(cell.stabilization_time)
        h.frecord_init()
        h.dt = 0.001
        h.continuerun(cell.stabilization_time + 2)
        syn.netcon.weight[0] = 0

        logger.debug("Simulation completed for synapse %d", i + 1)
        max_data["rec_site"]["maxv"].append(rec_site_v.max())
        max_data["rec_site"]["maxt"].append(t_rec_site.get(rec_site_v.max_ind()))
        max_data["syn"]["maxv"].append(syn_v.max())
        max_data["syn"]["maxt"].append(t_syn.get(syn_v.max_ind()))
        max_data["rec_site"]["restv"].append(rec_site_v[0])
        max_data["syn"]["restv"].append(syn_v[0])
        logger.debug(
            "v and t at max for rec site: %d mV, %f ms",
            rec_site_v.max(),
            t_rec_site.get(rec_site_v.max_ind()),
        )
        logger.debug(
            "v and t at max for synapse: %d mV, %f ms",
            syn_v.max(),
            t_syn.get(syn_v.max_ind()),
        )
        syn = None
    recordings = max_data
    logger.debug("Completed indiv_syn_test")
    return recordings


# TODO look into changing locs from 0-1 to 0-section.L
def syn_place(
    section: Section,
    tau1: float = 0.271,
    tau2: float = 0.271,
    e: float = 15,
    syn_density: float = None,
    locs: list[float] = None,
) -> list[Exp2Syn]:
    logger.debug("Placing synapses on section: %s", section.name())
    """
    Creates Exp2Syn point processes along a given section.

    Parameters
    ----------
    section : Section
        The NEURON section to place synapses on.
    tau1 : float, optional
        Rise time constant of the synapse (ms). Default is 0.271.
    tau2 : float, optional
        Decay time constant of the synapse (ms). Default is 0.271.
    e : float, optional
        Reversal potential of the synapse (mV). Default is 15.
    syn_density : float, optional
        Density of synapses (synapses per µm). If provided, overrides `locs`.
    locs : list, optional
        Specific locations (0-1) along the section to place synapses.

   	Returns
    -------
    synlist : list
        List of created synapses.
    """
    sec_len = section.L  # Length of the section in µm
    synlist = h.List()

    # Determine synapse placement based on density or specific locations
    if syn_density is None:
        if locs is not None:
            # Place synapses at specified locations
            for loc in locs:
                syn = h.Exp2Syn(section(loc))
                syn.tau1 = tau1
                syn.tau2 = tau2
                syn.e = e
                synlist.append(syn)
            return synlist
        else:
            # Default to placing synapses at each segment
            syncount = section.nseg
            syninc = sec_len / syncount
    else:
        # Calculate synapse placement based on density
        dens = syn_density
        syncount = int(sec_len * dens)
        syninc = sec_len / syncount
    # Create synapses along the section
    for i in range(0, syncount):
        syn = h.Exp2Syn(section(((syninc / 2) + (i * syninc)) / sec_len))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.e = e
        synlist.append(syn)
    logger.debug("Created %d synapses", len(synlist))
    return synlist


def syn_test(
    cell: Cell,
    section_lists: list[list[Section]],
    numsyn: int,
    synspace: float,
    axonspeed: float,
    numtrial: int,
    num_fiber: int = 1,
    gmax: float = 0.037,
    release_probability: float = 0.45,
    traces: bool = False,
    innervation_pattern: str = "random",
) -> dict:
    logger.debug("Starting syn_test")
    """
    Simulates synaptic input on a list of section lists (e.g., for each branch), placing synapse groups
    randomly or fully along them, and activates all synapses at once (not including axonal delay).

    Calculates the average time to peak and average halfwidth, along with error.


    Parameters
    ----------
    cell : Cell
        The cell model instance.
    section_lists : list
        List of sectionlists, usually used for polar sides of cell (lateral vs. medial).
    num_synapses : int
        Number of synapses within a synapse span. Ignored if innervation_pattern is 'total'.
    synspace : float
        Space between each synapse in a group. Ignored if innervation_pattern is 'total'.
    axonspeed : float, optional
        Speed of axonal delay lines (in m/s).
    numtrial : int
        Number of trials to be averaged. Each trial = new synaptic placement.
    num_fiber : int, optional
        Number of axon fibers (or synapse groups) innervating each list. Ignored if innervation_pattern is 'total'.
    gmax : float
        Combined conductance of all synapses in a span.
    release_probability : float (0-1)
        Chance of any individual synaptic release.
    traces : bool
        Whether or not traces for the trials are returned in the dictionary.
    innervation_pattern : str
        Pattern for innervation ('random' or 'total').

    Returns
     -------
     dict
        Dictionary containing the average time to peak, average halfwidth, and their standard errors.
        If traces is True, also includes the traces for the trials.
    """

    maxtimeArray = np.zeros(numtrial)
    halfwidthArray = np.zeros(numtrial)
    trace_list = []
    end_sections = []
    python_end_sections = []
    numpaths = []
    furthestdistance, furthestsegment = furthest_point(cell.somatic[0], section_lists[0])

    for section_listnum in range(len(section_lists)):
        end_sections.append(get_terminal_sections(section_lists[section_listnum]))
        python_end_sections.append(list(end_sections[section_listnum]))
        numpaths.append(len(python_end_sections[section_listnum]))
        if furthest_point(cell.somatic[0], section_lists[section_listnum])[0] > furthestdistance:
            furthestdistance = furthest_point(cell.somatic[0], section_lists[section_listnum])[0]
            furthestsegment = furthest_point(cell.somatic[0], section_lists[section_listnum])[1]

    for trialnum in range(numtrial):
        logger.debug("Running trial %d of %d", trialnum + 1, numtrial)

        synlists = []
        netconlists = []
        netstimlists = []
        syn_conductance_vectors = []
        for section_listnum in range(len(section_lists)):
            netconlist = []
            netstimlist = []
            if innervation_pattern == "random":
                for end in end_sections[section_listnum]:

                    temppath = get_parent_sections(
                        end
                    )  # creating a path composed of each section from an end to the soma
                    temppathlength = section_list_length(cell, temppath)[
                        0
                    ]  # getting the total length of the path
                    if (
                        numsyn * synspace > temppathlength
                    ):  # checking if the synapse group can fit along this path
                        raise Exception(
                            "Spread of synapses is longer than the shortest path length"
                        )

                    for j in range(num_fiber):
                        randpath = random.randint(
                            0, numpaths[section_listnum] - 1
                        )  # chooses a random number to choose a path
                        path = get_parent_sections(
                            python_end_sections[section_listnum][randpath]
                        )  # generates that section list of path from chosen end segment array element
                        pathlength = section_list_length(cell, path)[
                            0
                        ]  # calculates total path length
                        randpoint = (
                            random.random() * pathlength
                        )  # chooses a random value (0,1) and determines placement of most distal synapse of group

                        # pick a new random value if the previous starting point cannot fit on the desired side
                        while randpoint - (numsyn * synspace) < 0:
                            randpoint = random.random() * pathlength

                        # create storage lists
                        synlocations = []

                        for k in range(numsyn):  # create synapses and add to list
                            synlocations.append(randpoint - (synspace * k))

                            # generate group of synapses and add group list to a larger array of them all

                            syngroup = syn_path_place(cell, path, synlocations)
                            synlists.append(syngroup)
                        count = 0
            synlist = []
            if innervation_pattern == "total":
                for section_list in section_lists:
                    synlist = innervate_total(section_list)
                    synlists.append(synlist)
            for syngroup in synlists:
                for syn in syngroup:
                    syn_conductance_vectors.append(h.Vector().record(syn._ref_g))
                    netstim = h.NetStim()
                    netcon = h.NetCon(netstim, syn)
                    netcon.delay = 0
                    chance = random.random()
                    if chance >= release_probability:
                        netcon.weight[0] = 0
                    else:
                        netcon.weight[0] = gmax / numsyn

                    netstim.number = 1  # trigger once
                    netstim.interval = 1  # not necessary
                    if axonspeed is not None:
                        # find the distance of syn's segment from the most distant segment and calculate axon delay using given speed
                        distanceFromFurthestSegment = distance3D(
                            getsegxyz(furthestsegment), getsegxyz(syn.get_segment())
                        )
                        axon_delay = distanceFromFurthestSegment / (1000 * axonspeed)
                        netstim.start = cell.stabilization_time + axon_delay
                    else:
                        netstim.start = cell.stabilization_time
                    netstimlist.append(netstim)
                    netconlist.append(netcon)

                netconlists.append(netconlist)
                netstimlists.append(netstimlist)
        v = h.Vector().record(cell.somatic[0](0.5)._ref_v)
        t = h.Vector().record(h._ref_t)
        g = syn_conductance_vectors[0]
        h.finitialize()
        h.dt = 1
        h.continuerun(cell.stabilization_time - 5)
        h.frecord_init()
        h.dt = 0.001
        h.continuerun(cell.stabilization_time + 10)
        t = (
            t - cell.stabilization_time
        )  # set time relative to the start of synaptic activity
        for syn_conductance_vector in syn_conductance_vectors[1:]:
            g = g.add(syn_conductance_vector)
        g = g.div(len(syn_conductance_vectors))
        if traces == True:
            trace_list.append(
                {
                    "time": t.to_python(),
                    "voltage": v.to_python(),
                    "conductance": g.to_python(),
                }
            )
        maxVind = v.max_ind()
        maxtimeArray[trialnum] = t[maxVind]
        halfV = (v.max() + v[0]) / 2

        firsthalf = t[v.indwhere(">=", halfV)]
        secondhalf = t[v.cl(maxVind).indwhere("<=", halfV) + maxVind]
        halfwidth = secondhalf - firsthalf
        halfwidthArray[trialnum] = halfwidth
    maxtimeaverage = np.average(maxtimeArray)
    halfwidthaverage = np.average(halfwidthArray)

    maxtimesumofsquares = 0
    for maxtime in maxtimeArray:
        square = (maxtime - maxtimeaverage) ** 2
        maxtimesumofsquares += square

    maxtimevariance = maxtimesumofsquares / (numtrial - 1)
    maxtimestandarddev = math.sqrt(maxtimevariance)
    maxtimestandarderror = maxtimestandarddev / math.sqrt(numtrial)

    halfwidthsumofsquares = 0
    for halfwidth in halfwidthArray:
        square = (halfwidth - halfwidthaverage) ** 2
        halfwidthsumofsquares += square

    halfwidthvariance = halfwidthsumofsquares / (numtrial - 1)
    halfwidthstandarddev = math.sqrt(halfwidthvariance)
    halfwidthstandarderror = halfwidthstandarddev / math.sqrt(numtrial)
    if traces == True:
        return {
            "maxtime": maxtimeaverage,
            "maxtimestandarderror": maxtimestandarderror,
            "halfwidth": halfwidthaverage,
            "halfwidthstandarderror": halfwidthstandarderror,
            "traces": trace_list,
        }
    else:
        return {
            "maxtime": maxtimeaverage,
            "maxtimestandarderror": maxtimestandarderror,
            "halfwidth": halfwidthaverage,
            "halfwidthstandarderror": halfwidthstandarderror,
        }


def _cross_threshold(voltage_trace, threshold=0, relative=True):
    if len(voltage_trace) == 0:
        return False
    resting = voltage_trace[0] if relative else 0
    return max(voltage_trace) - resting >= threshold


def itd_test_sweep(
    cell: Cell,
    offset_sections: list[Section],
    stable_sections: list[Section],
    innervation_pattern: str,
    axon_speed: float = 1,
    cycles: int = 1,
    interval: float = 1,
    exc_fiber_gmax: float = 0.037,
    inhibition: bool = False,
    inh_timing: float = -0.32,
    inh_fiber_gmax: float = 0.022,
    threshold: float = 25,
    relative_threshold: bool = True,
    record_axon: bool = False,
    itd_vals: list[float] = None,
    traces: bool = False,
    **kwargs,
) -> dict:
    r"""
    Perform ITD (Interaural Time Difference) tests on a given cell.

    Parameters
    ----------
    cell : Cell
        The cell instance to be tested. : list
    offset_sections : list
        List of polar branches for the first section.
    stable_sections : list
        List of polar branches for the second section.
    innervation_pattern : str
        Pattern for innervation ('random', 'total').
    axon_speed : float, optional
        The speed of axonal propagation (in m/s). Default is 1.
    cycles : int, optional
        The number of netstim cycles. Default is 1.
    interval : float, optional
        The time between netstim cycles. Default is 1.
    exc_fiber_gmax : float, optional
        The maximum conductance for excitatory fibers. Default is 0.037.
    inhibition : bool, optional
        Whether to include inhibition in the test. Default is False.
    inh_timing : float, optional
        The delay for inhibition (in ms). Default is -0.32.
    inh_fiber_gmax : float, optional
        The maximum conductance for inhibitory fibers. Default is 0.022.
    threshold : float, optional
        The threshold for spike detection. Default is 0.
    relative_threshold : bool, optional
        Whether to use a threshold relative to cell's resting potential. Default is True.
    record_axon : bool, optional
        Whether to record axonal activity. Default is False.
    itd_vals : list, optional
        List of ITD values to test. Default is None.
    traces : bool, optional
        Whether to record traces. Default is False.
    syn_space : float, optional
        Space between each synapse in a group. Ignored if innervation_pattern is 'total'.
    num_syn : int, optional
        Number of synapses in each group. Ignored if innervation_pattern is 'total'.
    num_fiber : int, optional
        Number of axon fibers (or synapse groups) innervating each list. Ignored if innervation_pattern is 'total'.

    Returns
    -------
    results
        Results of the ITD test sweep {spike_counts, traces, itd_vals}.
    """

    saved_args = {**locals()}
    section_lists = [offset_sections, stable_sections]
    trace_list = {}
    if itd_vals is None:
        if "itd_range" in kwargs and "itd_step" in kwargs:
            itd_vals = np.arange(
                -(kwargs["itd_range"] / 2),
                (kwargs["itd_range"] / 2) + kwargs["itd_step"],
                kwargs["itd_step"],
            )
        else:
            raise Exception(
                "itd_vals must be provided or itd_range and itd_step must be specified"
            )
    spike_counts = np.zeros(len(itd_vals))
    inh_delays = np.array([inh_timing, inh_timing - 0.06])
    sim_start_time = (
        abs(min(np.min(itd_vals), np.min(inh_delays))) + cell.stabilization_time
    )
    logger.debug(f"itd vals:{itd_vals}")

    if innervation_pattern == "random":
        if {"numsyn", "synspace", "numfiber"}.issubset(kwargs):
            numsyn = kwargs["num_syn"]
            synspace = kwargs["syn_space"]
            numfiber = kwargs["num_fiber"]
        else:
            raise Exception(
                "numsyn, synspace, and numfiber must be provided for random innervation pattern"
            )

    input_length_lookup = get_all_input_lengths(
        cell, [offset_sections, stable_sections], **kwargs
    )
    if innervation_pattern == "total":
        total_synlists = []
        for section_list in section_lists:
            syngroups = [
                innervate_total(section_list, interval=interval, cycles=cycles)
            ]
            total_synlists.append(syngroups)
        dist_cond = exc_fiber_gmax / (
            sum([sec.nseg for sec in (list(offset_sections) + list(stable_sections))])
        )

    itd_vals_print = ", ".join([f"{val:.2f}" for val in itd_vals[:3]])
    itd_vals_print += "..."
    logger.info("Starting itd_test with delays: %s", itd_vals_print)
    line_width = 50
    log_str = "Parameters: \n"
    for key, value in saved_args.items():
        param_str = f"{key}: {value}"
        space = " " * (len(key) + 2)
        if len(param_str) > line_width:
            log_str += f"\t{param_str[:line_width]}...\n"
            param_str = param_str[line_width:]
            while len(param_str) > line_width:

                log_str += f"\t{space}{param_str[:line_width]}...\n"
                param_str = param_str[line_width:]
            if len(param_str) != 0:
                log_str += f"\t{space}{param_str}\n"
        elif len(param_str) != 0:
            log_str += f"\t{param_str}\n"
    logger.debug(log_str)

    for itd_num, itd_val in enumerate(itd_vals):
        logger.debug("Processing delay step %.2f ms", itd_val)
        inhibitsyns = np.empty(2, dtype=object)

        for inhibitsyn_num in range(2):
            inhibitsyn = h.Exp2Syn(cell.somatic[0](0.5))
            inhibitsyn.e = -90
            inhibitsyn.tau1 = 0.28
            inhibitsyn.tau2 = 1.85
            inhibitstim = h.NetStim()
            inhibitstim.start = -1
            inhibitstim.number = cycles
            inhibitstim.interval = interval
            inhibitcon = h.NetCon(inhibitstim, inhibitsyn)
            inhibitcon.delay = 0
            inhibitcon.weight[0] = inh_fiber_gmax / 2

            if inhibition:
                inhibitstim.start = (
                    sim_start_time
                    + inh_delays[inhibitsyn_num]
                    + (itd_val * inhibitsyn_num)
                    + (
                        axon_length_to_terminal(
                            cell,
                            inhibitsyn.get_segment(),
                            section_list=section_lists[inhibitsyn_num],
                            method="straight",
                        )[0]
                        / (axon_speed * 1000)
                    )
                    + (
                        axon_length_along(
                            inhibitsyn.get_segment(),
                            cell.somatic[0](0.5),
                        )
                        / (axon_speed * 1000)
                    )
                )

            inhibitsyns[inhibitsyn_num] = (inhibitsyn, inhibitstim, inhibitcon)

        syn_lists = [] if innervation_pattern != "total" else total_synlists
        for section_list_idx, section_list in enumerate(section_lists):
            if innervation_pattern == "random":
                logger.debug("Processing section list #%d", section_list_idx + 1)
                syngroups = innervate_random(
                    cell,
                    section_list,
                    numfiber,
                    numsyn,
                    synspace,
                    interval=interval,
                    cycles=cycles,
                )
                syn_lists.append(syngroups)

            for syn_group in syn_lists[section_list_idx]:
                for syn_unit in syn_group:

                    syn_unit.netstim.number = cycles
                    syn_unit.netstim.interval = interval
                    syn_unit.netcon.delay = 0
                    if random.random() >= 0.45:
                        syn_unit.netcon.weight[0] = 0
                    else:

                        syn_unit.netcon.weight[0] = (
                            exc_fiber_gmax / numsyn
                            if innervation_pattern == "random"
                            else dist_cond
                        )

                    axon_delay = 0
                    if axon_speed != 0:
                        axon_length = input_length_lookup[syn_unit.segment]
                        axon_delay = axon_length / (1000 * axon_speed)

                    syn_unit.netstim.start = sim_start_time + axon_delay

                    syn_unit.netstim.start += (
                        itd_val if section_list == offset_sections else 0
                    )

        v_soma = h.Vector()
        v_axon = h.Vector()
        t_soma = h.Vector()
        t_axon = h.Vector()
        cell.cvode.record(
            cell.somatic[0](0.5)._ref_v, v_soma, t_soma, sec=cell.somatic[0]
        )
        if record_axon:
            cell.cvode.record(
                cell.nodes[-1](0.5)._ref_v, v_axon, t_axon, sec=cell.nodes[-1]
            )
        v_monitor = v_axon if record_axon else v_soma
        t_monitor = t_axon if record_axon else t_soma

        h.finitialize(cell.resting_potential+1)
        h.continuerun(sim_start_time + interval * cycles + 5)


        curr_traces = {}
        if traces:
            curr_traces["time_soma"]= t_soma.to_python(),
            curr_traces["voltage_soma"] = v_soma.to_python()
            if record_axon:
                curr_traces["time_axon"] = t_axon.to_python()
                curr_traces["voltage_axon"] = v_axon.to_python()

        spike_counts[itd_num] = int(
            _cross_threshold(
                v_monitor, threshold=threshold, relative=relative_threshold
            )
        )


        for syn_list in syn_lists:
            for syn_group in syn_list:
                for syn_unit in syn_group:
                    syn_unit.netcon.weight[0] = 0
        del syn_lists

        trace_list[itd_val] = curr_traces if traces else None
    logger.info("Completed itd_test for cell: %s", cell.cell_name)
    logger.debug("Returning delay threshold probabilities and traces")
    return {
        "spike_counts": spike_counts,
        "traces": trace_list if traces else None,
        "itd_vals": itd_vals,
    }


def get_attenuation_values(
    cell,  # cell instance
    sectionlist1,
    sectionlist2,  # list of polar branches
    exc_fiber_gmax=0.037,
):
    logger.info("Calculating attenuation values for cell: %s", cell.cell_name)
    """Get attenuation values for a cell"""
    section_lists = [sectionlist1, sectionlist2]
    section_list_data = [dict(), dict()]
    for list_index, section_list in enumerate(section_lists):
        for sec in section_list:
            for seg in sec:
                syn = h.Exp2Syn(seg)
                syn.tau1 = 0.29
                syn.tau2 = 0.29
                syn_con = h.NetCon(None, syn, weight=exc_fiber_gmax)
                syn_con.delay = 0
                syn.event(1000)
                t = h.Vector.record(h._ref_t)
                v_soma = h.Vector.record(cell.somatic[0](0.5)._ref_v)
                v_syn = h.Vector.record(seg._ref_v)
                h.finitialize()
                h.continuerun(10010)
                v_proportion = v_syn.max() / v_soma.max()
                try:
                    section_list_data[list_index][sec.nchild].append(v_proportion)
                except:
                    section_list_data[list_index][sec.nchild] = [v_proportion]

    logger.info("Completed attenuation values calculation")
    return section_list_data
