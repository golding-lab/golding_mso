"""
Module for running simulations (propagation, ITD, etc.) on MSO models.
"""

import logging
import math
from golding_mso.utils import load_spike_times
import numpy as np
import random
from .nrn_types import Section, Segment, Exp2Syn
from collections.abc import Callable
from neuron import h
from .cell import Cell
from .cell_calc import (
    find_nonoverlapping_paths,
    get_terminal_sections,
    get_parent_sections,
    getsegxyz,
    get_all_input_lengths,
    closest_terminal_segment,
    furthest_point,
    section_list_length,
    distance3D,
    axon_length_along,
    axon_length_to_terminal,
)
from .syns import (
    SynapseFiber,
    SynapseTerminal,
    innervate_total,
    innervate_random,
    innervate_points,
    lookup_syns_by_section,
    syn_path_place,
)

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


def get_attenuation_values(
    cell,  # cell instance
    sectionlist1,
    sectionlist2,  # list of polar branches
    exc_gmax=0.037,
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
                syn_con = h.NetCon(None, syn, weight=exc_gmax)
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


class ITDTest:
    def __init__(
        self,
        cell,
        offset_sections,
        stable_sections,
        axon_speed=1,
        cycles=1,
        interval=1,
        exc_gmax=0.037,
        gmax_per_arbor=False,
        fibers=6,
        inhibition=False,
        inh_timing=-0.32,
        inh_gmax=0.022,
        threshold=25,
        relative_threshold=True,
        record_axon=False,
        itd_vals=None,
        traces=False,
        seed=None,
        stochastic=True,
        **kwargs,
    ):
        self._init_state = True
        self.seed = seed
        if self.seed: random.seed(self.seed)
        self.cell = cell
        self.offset_sections = offset_sections
        self.stable_sections = stable_sections
        self.section_lists = [offset_sections, stable_sections]
        self.axon_speed = axon_speed
        self.cycles = cycles
        self.interval = interval
        self.exc_gmax = exc_gmax
        self.gmax_per_arbor = gmax_per_arbor
        self.fibers = fibers
        self.stochastic=stochastic
        self.inhibition = inhibition
        self.inh_timing = inh_timing
        self.inh_gmax = inh_gmax
        self.threshold = threshold
        self.relative_threshold = relative_threshold
        self.record_axon = record_axon
        self.itd_vals = itd_vals
        self.traces = traces
        self.input_length_lookup = get_all_input_lengths(self.cell, self.section_lists)
        self.inh_delays = np.array([self.inh_timing, self.inh_timing - 0.06])
        self.sim_start_time = (
            abs(min(np.min(self.itd_vals), np.min(self.inh_delays)))
            + self.cell.stabilization_time
        )
        self.distr_conds = [self.exc_gmax / (
            sum(
                [
                    sec.nseg
                    for sec in ((list(self.offset_sections) + list(self.stable_sections)) if not self.gmax_per_arbor else section_list)
                ]
            )
        ) for section_list in self.section_lists] 
        self._setup_run()
        self._init_state = False
    
    def create_synapses(self):
        exc_synlists = []
        for sl_num, section_list in enumerate(self.section_lists):
            syngroup = innervate_total(
                section_list,
                interval=self.interval,
                cycles=self.cycles,
                gmax=self.distr_conds[sl_num],
                stochastic=self.stochastic
            )
            exc_synlists.append(syngroup)

        inhibitsyns = innervate_points(
                self.cell.somatic[0](0.5), self.cell.somatic[0](0.5),
                gmax=self.inh_gmax / 2,
                number=self.cycles,
                interval=self.interval,
                tau1=0.28,
                tau2=1.85,
                e=-90,
                stochastic=self.stochastic,
            )
        return {"exc": exc_synlists, "inh": inhibitsyns}

    def _get_inhibition_input_length(
        self, syn, section_list
    ):
        input_length = self.sim_start_time + (
            h.distance(closest_terminal_segment(section_list, syn.segment), syn.segment)
        )
        return input_length

    def set_all_synapse_activation(self, itd_val=0):
        if self.inhibition:
            for inhibitsyn_num in range(2):
                spike_train = None
                if hasattr(self, "inh_spike_trains"):
                    spike_train = (
                        random.choice(self.inh_spike_trains)
                        if self.inh_spike_train_method == "random"
                        else self.inh_spike_trains[inhibitsyn_num%len(self.inh_spike_trains)]
                    )

                self._set_inhibition_activation(
                    self.inhibitsyns[inhibitsyn_num],
                    itd_val=itd_val,
                    delay=self.inh_delays[inhibitsyn_num],
                    input_length=self._get_inhibition_input_length(
                        self.inhibitsyns[inhibitsyn_num],
                        (
                            self.offset_sections
                            if inhibitsyn_num == 0
                            else self.stable_sections
                        ),
                    ),
                    spike_train=(spike_train),
                    offset=inhibitsyn_num == 0,
                )

        logger.debug("Processing delay step %.2f ms", itd_val)
        self.syn_lists = self._set_all_exc_syn_timing(itd_val=itd_val)

    def _setup_run(self):
        allsyns = self.create_synapses()
        self.syn_lists, self.inhibitsyns = allsyns["exc"], allsyns["inh"]

    def run_sweep(self, itd_vals=None, duration=None):
        itd_vals = self.itd_vals if itd_vals is None else itd_vals
        spike_counts = np.zeros(len(itd_vals))
        trace_list = {}

        for itd_num, itd_val in enumerate(itd_vals):
            spike_counts[itd_num], curr_traces = self.run_at_itd(
                itd_val=itd_val, duration=duration
            )
            if self.traces:
                trace_list[itd_val] = curr_traces

        logger.info("Completed itd_test for cell: %s", self.cell.cell_name)
        logger.debug("Returning delay threshold probabilities and traces")
        return {
            "spike_counts": spike_counts,
            "traces": trace_list if self.traces else None,
            "itd_vals": itd_vals,
        }

    def run_at_itd(self, itd_val=0, duration=None):

        self.set_all_synapse_activation(itd_val=itd_val)
        v_monitor, t_monitor, v_soma, t_soma, v_axon, t_axon = self._complete_itd_sim(
            duration=duration
        )
        curr_traces = {}
        if self.traces:
            curr_traces["time_soma"] = t_soma.to_python()
            curr_traces["voltage_soma"] = v_soma.to_python()
            if self.record_axon:
                curr_traces["time_axon"] = t_axon.to_python()
                curr_traces["voltage_axon"] = v_axon.to_python()

        spike_count = int(
            self._cross_threshold(
                v_monitor,
                threshold=self.threshold,
                relative=self.relative_threshold,
            )
        )

        return spike_count, curr_traces

    def _set_exc_syn_timing(self, fiber, itd_val=0, spike_train=None, gmax=0.037):
        for syn in fiber.synapse_terminals:
            syn.netstim.number = self.cycles
            syn.netstim.interval = self.interval
            syn.netcon.delay = 0
            if random.random() <= 0.45 or (hasattr(self, "exc_spike_trains")):
                syn.netcon.weight[0] = gmax
            else:
                syn.netcon.weight[0] = 0
            axon_delay = 0
            if self.axon_speed != 0:
                input_length = self.input_length_lookup[syn.segment]
                axon_delay = input_length / (1000 * self.axon_speed)
            syn.start = self.sim_start_time + axon_delay
            syn.start += itd_val
            # print(syn.start)
        if spike_train is not None:
            fiber.set_firing_times(spike_train)
            # print(fiber.shared_firing_times)

    def _set_all_exc_syn_timing(self, itd_val=0):
        for section_list_idx, section_list in enumerate(self.section_lists):
            current_synapses = []
            if self.fibered:
                for path in find_nonoverlapping_paths(section_list).values():
                    fiber_syns = []
                    for sec in path:
                        fiber_syns += lookup_syns_by_section(self.syn_lists[section_list_idx], sec)
                    current_synapses.append(SynapseFiber(fiber_syns))
            
            else: current_synapses = [SynapseFiber([syn]) for syn in self.syn_lists[section_list_idx]]
            print('len(current_synapses):', len(current_synapses))
            for fiber_num, fiber in enumerate(current_synapses):
                spike_train = None
                if hasattr(self, "exc_spike_trains"):
                    if self.exc_spike_train_method == "random":
                        spike_train = random.choice(self.exc_spike_trains)
                    elif self.exc_spike_train_method == "repeat":
                        spike_train = self.exc_spike_trains[
                            fiber_num
                            * (section_list_idx + 1)
                            % len(self.exc_spike_trains)
                        ]
                self._set_exc_syn_timing(
                    fiber,
                    itd_val=(
                        itd_val if section_list == self.offset_sections else 0
                    ),
                    spike_train=spike_train,
                    gmax=self.distr_conds[section_list_idx]
                )
        return self.syn_lists

    def set_spike_trains(self, spike_trains=None, freq=None, method="repeat", which="both"):
        if spike_trains is None:
            spike_trains = load_spike_times(freq, self.fibers)
        if len(spike_trains) != len(self.syn_lists):
            logger.debug(
                "Number of spike trains does not match number of synapse groups. Will start repeating them from the beginning after exhausting them."
            )
        if which not in ["exc", "inh", "both"]:
            raise ValueError("Invalid value for 'which'. Use 'exc' or 'inh'.")
        if method not in ["repeat", "random"]:
            raise ValueError("Invalid value for 'method'. Use 'repeat' or 'random'.")
        if which == "both":
            self.__setattr__(f"exc_spike_trains", spike_trains)
            self.__setattr__(f"exc_spike_train_method", method)
            self.__setattr__(f"inh_spike_trains", spike_trains)
            self.__setattr__(f"inh_spike_train_method", method)
        else:
            self.__setattr__(f"{which}_spike_trains", spike_trains)
            self.__setattr__(f"{which}_spike_train_method", method)

    def _complete_itd_sim(self, duration=None):
        v_soma = h.Vector()
        v_axon = h.Vector()
        t_soma = h.Vector()
        t_axon = h.Vector()
        runtime = (
            self.sim_start_time + duration
            if duration is not None
            else self.sim_start_time + self.interval * self.cycles + 5
        )
        self.cell.cvode.record(
            self.cell.somatic[0](0.5)._ref_v, v_soma, t_soma, sec=self.cell.somatic[0]
        )
        if self.record_axon:
            self.cell.cvode.record(
                self.cell.nodes[-1](0.5)._ref_v, v_axon, t_axon, sec=self.cell.nodes[-1]
            )
        v_monitor = v_axon if self.record_axon else v_soma
        t_monitor = t_axon if self.record_axon else t_soma

        h.finitialize(self.cell.resting_potential + 1)
        h.continuerun(self.sim_start_time - 1)
        h.frecord_init()
        h.continuerun(runtime)

        return v_monitor, t_monitor, v_soma, t_soma, v_axon, t_axon

    def _set_inhibition_activation(
        self, syn, delay=0, itd_val=0, input_length=0, offset=False, spike_train=None
    ):
        
        syn.start = (
            self.sim_start_time 
            + delay 
            + (itd_val if offset else 0) 
            + input_length / (self.axon_speed * 1000)
        )
        
        if spike_train is not None:
            syn.set_firing_times(spike_train)

    def _cross_threshold(self, voltage_trace, threshold=0, relative=True):
        if len(voltage_trace) == 0:
            return False
        resting = voltage_trace[0] if relative else 0
        abs_threshold = resting + threshold if relative else threshold
        spike_bin = h.Vector().spikebin(voltage_trace, abs_threshold)
        return spike_bin.sum()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if (
            name in ["cell", "section_lists", "inh_timing", "itd_vals"]
            and not self._init_state
        ):
            self.input_length_lookup = get_all_input_lengths(
                self.cell, self.section_lists
            )
            self.inh_delays = np.array([self.inh_timing, self.inh_timing - 0.06])
            self.sim_start_time = (
                abs(min(np.min(self.itd_vals), np.min(self.inh_delays)))
                + self.cell.stabilization_time
            )
            self._setup_run()

    def __str__(self):
        itd_vals_print = ", ".join([f"{val:.2f}" for val in self.itd_vals[:3]])
        itd_vals_print += "..."
        line_width = 50
        log_str = f"ITD Test for cell: {self.cell.cell_name}\n"
        log_str += "Parameters: \n"
        for key, value in self.__dict__.items():
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
        return log_str
