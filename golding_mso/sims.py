"""
Module for running simulations (propagation, ITD, etc.) on MSO models.
"""

import logging
import random
import numpy as np
from collections.abc import Callable
from neuron import h
from .nrn_types import Section, Segment
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
)
from .syns import (
    SynapseFiber,
    SynapseTerminal,
    innervate_total,
    innervate_points,
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
        Cell instance to pass to indiv_syn_test.
    section_list: list[Section]
        sectionlist (from Cell instance) to place synapses on

    Returns
    -------
    list[dict[str, dict[str, list[float]]]]
        List of dictionaries containing max voltage and time data for each section.
    """
    sections = list(section_list)  # convert SectionList() to a python list for len()
    logger.debug("Starting propagation_test over %d sections", len(sections))
    propagation_data = []
    for section_index, section in enumerate(sections, start=1):
        logger.debug(
            "Processing section: %s, %s of %s", section.name(), section_index, len(sections)
        )
        synapses = innervate_total([section])  # one synapse per segment in the section
        propagation_data.append(indiv_syn_test(cell, synapses, cell.somatic[0], **kwargs))
    logger.debug("Completed propagation_test")
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
    key = {"voltage": "maxv", "time": "maxt"}.get(which)
    if key is None:
        raise ValueError("Invalid value for 'which'. Use 'voltage' or 'time'.")
    return [
        func(rec_val, syn_val, **kwargs)
        for rec_val, syn_val in zip(rec_site[key], synapse[key])
    ]


def find_compensated_cond(
    cell: Cell,
    seg: Segment,
    maxv: float,
    desired_maxv: float,
    original_gmax: float = 0.005,
    tolerance: float = 0.1,
    resting_potential: float = -61.7,
    recursion_limit: int = 20,
) -> float:
    """
    Iteratively adjusts synaptic conductance (gmax) to achieve a desired max voltage.

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
        The maximum number of adjustment attempts allowed.
        Default is 20.

    Returns
    -------
    gmax: float
        The adjusted conductance value that achieves the desired maximum voltage.
    """
    gmax = original_gmax
    for attempt in range(1, recursion_limit + 1):
        percent_change = (desired_maxv - resting_potential) / (maxv - resting_potential)
        gmax *= percent_change

        synapse = innervate_points(seg)[0]
        try:
            data = indiv_syn_test(cell, [synapse], cell.somatic[0], gmax=gmax)
        finally:
            synapse.destroy()

        maxv = data["rec_site"]["maxv"][0]
        resting_potential = data["rec_site"]["restv"][0]
        error = abs(desired_maxv - maxv)
        if error <= tolerance:
            logger.info("Found adjusted conductance: %s\nOff by: %s", gmax, error)
            return gmax
        logger.debug("Attempt %d: conductance (%s) off by %s", attempt, gmax, error)

    logger.warning("Recursion limit reached. Ending with gmax: %s", gmax)
    return gmax


def indiv_syn_test(
    cell: Cell, synapse_list: list[SynapseTerminal], rec_section: Section, gmax: float = 0.01
) -> dict[str, dict[str, list]]:
    """
    Activates each synapse individually in a given list and, for each synapse, records the
    max voltages and time at those voltages at the recording site and the synapse site

    Parameters
    ----------
    cell: Cell
        Cell instance to pull stabilization_time variable to use.
    synapse_list: list[SynapseTerminal]
        list of SynapseTerminal objects to activate.
    rec_section: Section
        Section to record from for the non-synapse compartment measurements.
        Records from center of given section.
    gmax: float, optional
        Conductance to activate each synapse with. Default is 0.01.

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
    for i, syn in enumerate(synapse_list, start=1):
        logger.debug("Activating synapse %d of %d", i, syncount)
        syn.set_firing_times([cell.stabilization_time])
        syn.netcon.weight[0] = gmax

        rec_site_v, t_rec_site = h.Vector(), h.Vector()
        syn_v, t_syn = h.Vector(), h.Vector()
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

        for site, v_vec, t_vec in (
            ("rec_site", rec_site_v, t_rec_site),
            ("syn", syn_v, t_syn),
        ):
            max_ind = v_vec.max_ind()
            max_data[site]["maxv"].append(v_vec.max())
            max_data[site]["maxt"].append(t_vec.get(max_ind))
            max_data[site]["restv"].append(v_vec[0])

        logger.debug(
            "Synapse %d: rec site %.2f mV @ %.3f ms | synapse %.2f mV @ %.3f ms",
            i,
            max_data["rec_site"]["maxv"][-1],
            max_data["rec_site"]["maxt"][-1],
            max_data["syn"]["maxv"][-1],
            max_data["syn"]["maxt"][-1],
        )
    logger.debug("Completed indiv_syn_test")
    return max_data


def syn_place(
    section: Section,
    tau1: float = 0.271,
    tau2: float = 0.271,
    e: float = 15,
    syn_density: float | None = None,
    locs: list[float] | None = None,
    **kwargs,
) -> list[SynapseTerminal]:
    r"""
    Creates SynapseTerminals along a given section.

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
        Density of synapses (synapses per µm). If provided, overrides `locs` and evenly-spaced placement.
    locs : list[float], optional
        Specific locations (0-1) along the section to place synapses. Ignored if `syn_density` is provided.
    \**kwargs:
        Additional keyword arguments passed to the SynapseTerminal constructor (e.g. gmax, delay).

    Returns
    -------
    synlist : list[SynapseTerminal]
        List of created synapse terminals.
    """
    logger.debug("Placing synapses on section: %s", section.name())
    sec_len = section.L  # Length of the section in µm

    if syn_density is not None:
        # Calculate synapse placement based on density
        syncount = int(sec_len * syn_density)
        syninc = sec_len / syncount
        locations = [(syninc / 2 + i * syninc) / sec_len for i in range(syncount)]
    elif locs is not None:
        # Place synapses at specified locations
        locations = locs
    else:
        # Default to placing synapses at each segment
        syncount = section.nseg
        syninc = sec_len / syncount
        locations = [(syninc / 2 + i * syninc) / sec_len for i in range(syncount)]

    synlist = innervate_points(
        *(section(loc) for loc in locations), tau1=tau1, tau2=tau2, e=e, **kwargs
    )
    logger.debug("Created %d synapses", len(synlist))
    return synlist


def _random_syn_groups(
    cell: Cell,
    section_list: list[Section],
    numsyn: int,
    synspace: float,
    num_fiber: int,
) -> list[list[SynapseTerminal]]:
    """
    Place `num_fiber` groups of `numsyn` synapses (spaced `synspace` apart) at random
    locations along random terminal paths of `section_list`.
    """
    terminal_paths = [
        get_parent_sections(end) for end in get_terminal_sections(section_list)
    ]
    path_lengths = [section_list_length(cell, path)[0] for path in terminal_paths]
    if numsyn * synspace > min(path_lengths):
        raise ValueError("Spread of synapses is longer than the shortest path length")

    groups = []
    for _ in range(num_fiber):
        path_index = random.randrange(len(terminal_paths))
        path, pathlength = terminal_paths[path_index], path_lengths[path_index]
        # choose the location of the most distal synapse, leaving room for the full spread
        randpoint = random.uniform(numsyn * synspace, pathlength)
        synlocations = [randpoint - synspace * k for k in range(numsyn)]
        groups.append(syn_path_place(cell, path, synlocations))
    return groups


def _standard_error(values: np.ndarray) -> float:
    """Standard error of the mean (sample standard deviation, ddof=1) for a 1D array of trial values."""
    if len(values) < 2:
        return float("nan")
    return np.std(values, ddof=1) / np.sqrt(len(values))


def syn_test(
    cell: Cell,
    section_lists: list[list[Section]],
    numsyn: int,
    synspace: float,
    axonspeed: float | None,
    numtrial: int,
    num_fiber: int = 1,
    gmax: float = 0.037,
    release_probability: float = 0.45,
    traces: bool = False,
    innervation_pattern: str = "random",
) -> dict:
    """
    Simulates synaptic input on a list of section lists (e.g., for each branch), placing synapse groups
    randomly or fully along them, and activates all synapses at once (including axonal delay if given).

    Calculates the average time to peak and average halfwidth, along with their standard errors.

    Parameters
    ----------
    cell : Cell
        The cell model instance.
    section_lists : list
        List of sectionlists, usually used for polar sides of cell (lateral vs. medial).
    numsyn : int
        Number of synapses within a synapse span. Ignored if innervation_pattern is 'total'.
    synspace : float
        Space between each synapse in a group. Ignored if innervation_pattern is 'total'.
    axonspeed : float, optional
        Speed of axonal delay lines (in m/s). If None, no axonal delay is applied.
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
    logger.debug("Starting syn_test")

    furthest_distance, furthest_segment = -1.0, None
    for section_list in section_lists:
        distance, segment = furthest_point(cell.somatic[0], section_list)
        if distance > furthest_distance:
            furthest_distance, furthest_segment = distance, segment

    maxtimes = np.zeros(numtrial)
    halfwidths = np.zeros(numtrial)
    trace_list = []

    for trialnum in range(numtrial):
        logger.debug("Running trial %d of %d", trialnum + 1, numtrial)

        if innervation_pattern == "random":
            synlists = [
                group
                for section_list in section_lists
                for group in _random_syn_groups(
                    cell, section_list, numsyn, synspace, num_fiber
                )
            ]
        elif innervation_pattern == "total":
            synlists = [innervate_total(section_list) for section_list in section_lists]
        else:
            raise ValueError(f"Unknown innervation_pattern: {innervation_pattern!r}")

        conductance_vectors = []
        for syngroup in synlists:
            for syn in syngroup:
                conductance_vectors.append(h.Vector().record(syn.syn._ref_g))
                syn.netcon.delay = 0
                released = random.random() < release_probability
                syn.gmax = (gmax / numsyn) if released else 0

                axon_delay = 0.0
                if axonspeed is not None:
                    # delay proportional to the synapse's distance from the furthest segment in the tree
                    distance = distance3D(
                        getsegxyz(furthest_segment), getsegxyz(syn.segment)
                    )
                    axon_delay = distance / (1000 * axonspeed)
                syn.set_firing_times([cell.stabilization_time + axon_delay])

        v = h.Vector().record(cell.somatic[0](0.5)._ref_v)
        t = h.Vector().record(h._ref_t)

        h.finitialize()
        h.dt = 1
        h.continuerun(cell.stabilization_time - 5)
        h.frecord_init()
        h.dt = 0.001
        h.continuerun(cell.stabilization_time + 10)

        t = t - cell.stabilization_time  # time relative to the start of synaptic activity
        conductance = conductance_vectors[0]
        for vec in conductance_vectors[1:]:
            conductance = conductance.add(vec)
        conductance = conductance.div(len(conductance_vectors))

        if traces:
            trace_list.append(
                {
                    "time": t.to_python(),
                    "voltage": v.to_python(),
                    "conductance": conductance.to_python(),
                }
            )

        max_ind = v.max_ind()
        maxtimes[trialnum] = t[max_ind]
        half_v = (v.max() + v[0]) / 2
        first_half_t = t[v.indwhere(">=", half_v)]
        second_half_t = t[v.cl(max_ind).indwhere("<=", half_v) + max_ind]
        halfwidths[trialnum] = second_half_t - first_half_t

    result = {
        "maxtime": np.mean(maxtimes),
        "maxtimestandarderror": _standard_error(maxtimes),
        "halfwidth": np.mean(halfwidths),
        "halfwidthstandarderror": _standard_error(halfwidths),
    }
    if traces:
        result["traces"] = trace_list
    logger.debug("Completed syn_test")
    return result


def get_attenuation_values(
    cell: Cell,
    sectionlist1: list[Section],
    sectionlist2: list[Section],
    exc_gmax: float = 0.037,
) -> list[dict[int, list[float]]]:
    """
    Measures voltage attenuation from each segment to the soma for two section lists
    (e.g. polar branches), grouped by branch order (number of child sections).

    Parameters
    ----------
    cell : Cell
        The cell model instance.
    sectionlist1 : list[Section]
        First list of sections (e.g. lateral branches) to measure attenuation for.
    sectionlist2 : list[Section]
        Second list of sections (e.g. medial branches) to measure attenuation for.
    exc_gmax : float, optional
        Conductance of the synapse used to probe each segment. Default is 0.037.

    Returns
    -------
    section_list_data : list[dict[int, list[float]]]
        For each section list, a dictionary mapping branch order (number of child sections) to a
        list of segment/soma peak-voltage ratios for every segment at that branch order.
    """
    logger.info("Calculating attenuation values for cell: %s", cell.cell_name)
    section_list_data = [dict(), dict()]
    for list_index, section_list in enumerate((sectionlist1, sectionlist2)):
        for sec in section_list:
            for seg in sec:
                syn = innervate_points(seg, tau1=0.29, tau2=0.29)[0]
                syn.gmax = exc_gmax
                syn.set_firing_times([100])

                v_soma = h.Vector().record(cell.somatic[0](0.5)._ref_v)
                v_syn = h.Vector().record(seg._ref_v)
                h.finitialize()
                h.continuerun(110)

                v_proportion = v_syn.max() / v_soma.max()
                num_children = len(sec.children())
                section_list_data[list_index].setdefault(num_children, []).append(
                    v_proportion
                )
                syn.destroy()

    logger.info("Completed attenuation values calculation")
    return section_list_data


class ITDTest:
    """Simulates and measures a cell's spiking response to binaural input across a range of ITDs."""

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
        fibered=True,
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
        inh_rzero=70,
        exc_rzero=30,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cell : Cell
            The cell model instance to simulate.
        offset_sections : list[Section]
            Sections innervated by the fiber whose timing is shifted by the tested ITD.
        stable_sections : list[Section]
            Sections innervated by the fiber whose timing stays fixed across the ITD sweep.
        axon_speed : float, optional
            Speed of axonal delay lines (m/s). Default is 1.
        cycles : int, optional
            Number of activations per synapse per trial. Default is 1.
        interval : float, optional
            Interval (ms) between activations when cycles > 1. Default is 1.
        exc_gmax : float, optional
            Total excitatory conductance distributed across all synapses in a section list.
            Default is 0.037.
        gmax_per_arbor : bool, optional
            If True, distribute exc_gmax separately within each section list rather than
            across both combined. Default is False.
        fibered : bool, optional
            If True, group synapses into non-overlapping fibers per path to the soma.
            Default is True.
        inhibition : bool, optional
            Whether to include inhibitory synapses on the soma. Default is False.
        inh_timing : float, optional
            Timing offset (ms) applied to inhibitory synapses. Default is -0.32.
        inh_gmax : float, optional
            Total inhibitory conductance split across the two inhibitory synapses.
            Default is 0.022.
        threshold : float, optional
            Spike detection threshold (mV), absolute or relative to resting potential.
            Default is 25.
        relative_threshold : bool, optional
            If True, `threshold` is relative to the resting potential. Default is True.
        record_axon : bool, optional
            If True, record from the distal axon node instead of the soma. Default is False.
        itd_vals : array-like, optional
            Default ITD values (ms) used by `run_sweep` when none are given. Default is None.
        traces : bool, optional
            Whether simulations should return voltage/time traces. Default is False.
        seed : int, optional
            Random seed applied before each simulation for reproducibility. Default is None.
        stochastic : bool, optional
            Whether synapse creation should use stochastic release. Default is True.
        inh_rzero : float, optional
            Number of release sites used for inhibitory synapse activation. Default is 70.
        exc_rzero : float, optional
            Number of release sites used for excitatory synapse activation. Default is 30.
        """
        self._init_state = True
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.cell = cell
        self.offset_sections = offset_sections
        self.stable_sections = stable_sections
        self.section_lists = [offset_sections, stable_sections]
        self.axon_speed = axon_speed
        self.cycles = cycles
        self.interval = interval
        self.exc_gmax = exc_gmax
        self.gmax_per_arbor = gmax_per_arbor
        self.fibered = fibered
        self.stochastic = stochastic
        self.inhibition = inhibition
        self.inh_timing = inh_timing
        self.inh_gmax = inh_gmax
        self.threshold = threshold
        self.relative_threshold = relative_threshold
        self.record_axon = record_axon
        self.itd_vals = itd_vals
        self.traces = traces
        self.exc_rzero = exc_rzero
        self.inh_rzero = inh_rzero
        self._recompute_derived_state()
        self.exc_gmax_per_segment = self._compute_exc_gmax_per_segment()
        self._setup_run()
        self._init_state = False

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if (
            name in ["cell", "section_lists", "inh_timing", "itd_vals"]
            and not self._init_state
        ):
            self._recompute_derived_state()
            self._setup_run()

    def _recompute_derived_state(self):
        """Refresh values derived from cell geometry, section lists, and ITD/inhibition timing."""
        self.input_length_lookup = get_all_input_lengths(self.cell, self.section_lists)
        self.inh_delays = np.array([self.inh_timing, self.inh_timing - 0.06])
        self.sim_start_time = (
            abs(min(np.min(self.itd_vals), np.min(self.inh_delays)))
            + self.cell.stabilization_time
        )

    def _compute_exc_gmax_per_segment(self):
        """Distribute total excitatory gmax evenly across segments, per-arbor or combined."""
        combined_sections = list(self.offset_sections) + list(self.stable_sections)
        gmax_per_segment = []
        for section_list in self.section_lists:
            sections = section_list if self.gmax_per_arbor else combined_sections
            total_segments = sum(sec.nseg for sec in sections)
            gmax_per_segment.append(self.exc_gmax / total_segments)
        return gmax_per_segment

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

    def run_sweep(self, itd_vals=None, duration=None):
        """
        Run a simulation at each ITD in `itd_vals` and collect spike counts and times for each.

        Parameters
        ----------
        itd_vals : array-like, optional
            ITD values (ms) to test. Defaults to `self.itd_vals`.
        duration : float, optional
            Duration (ms) of each simulation, measured from `sim_start_time`. Defaults to None.

        Returns
        -------
        dict
            Dictionary containing spike counts, spike times, traces (if recorded), and
            the tested ITD values.
        """
        itd_vals = self.itd_vals if itd_vals is None else itd_vals
        spike_counts = np.zeros(len(itd_vals))
        spike_times_list = {}
        trace_list = {}

        for itd_num, itd_val in enumerate(itd_vals):
            spike_counts[itd_num], spike_times, curr_traces, _ = list(
                self.run_at_itd(itd_val=itd_val, duration=duration).values()
            )
            spike_times_list[itd_val] = spike_times
            if self.traces:
                trace_list[itd_val] = curr_traces

        logger.info("Completed itd_test for cell: %s", self.cell.cell_name)
        logger.debug("Returning delay threshold probabilities and traces")
        return {
            "spike_counts": spike_counts,
            "spike_times": spike_times_list,
            "traces": trace_list if self.traces else None,
            "itd_vals": itd_vals,
        }

    def run_at_itd(self, itd_val=0, duration=None):
        """
        Run a single simulation at the given ITD.

        Parameters
        ----------
        itd_val : float, optional
            ITD value (ms) applied to the offset section list. Default is 0.
        duration : float, optional
            Duration (ms) of the simulation, measured from `sim_start_time`. Defaults to None.

        Returns
        -------
        dict
            Dictionary containing the spike count, spike times, traces (if recorded), and
            the ITD value tested.
        """
        random.seed(self.seed)  # reset seed for reproducibility across sweeps
        self._set_all_synapse_activation(itd_val=itd_val)
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

        spike_count, spike_times = self._cross_threshold(
            v_monitor,
            threshold=self.threshold,
            relative=self.relative_threshold,
            time_trace=t_monitor,
        )
        return {
            "spike_count": spike_count,
            "spike_times": spike_times,
            "traces": curr_traces if self.traces else None,
            "itd_val": itd_val,
        }

    def _set_all_synapse_activation(self, itd_val=0):
        if self.inhibition:
            self._set_all_inh_syn_activation(itd_val=itd_val)

        logger.debug("Processing delay step %.2f ms", itd_val)
        self.syn_lists = self._set_all_exc_syn_activation(itd_val=itd_val)

    def set_spike_trains(self, spike_trains, method="repeat", which="exc"):
        """
        Assign a pool of pre-generated spike trains to excitatory or inhibitory fibers.

        Parameters
        ----------
        spike_trains : list
            Pool of spike trains to assign to fibers.
        method : str, optional
            How to distribute `spike_trains` across fibers: 'repeat' cycles through the pool
            in order, 'random' picks one at random for each fiber. Default is 'repeat'.
        which : str, optional
            Which synapse population to assign trains to, 'exc' or 'inh'. Default is 'exc'.
        """
        if len(spike_trains) != len(self.syn_lists):
            logger.debug(
                "Number of spike trains does not match number of synapse groups. Will start repeating them from the beginning after exhausting them."
            )
        if which not in ["exc", "inh"]:
            raise ValueError("Invalid value for 'which'. Use 'exc' or 'inh'.")
        if method not in ["repeat", "random"]:
            raise ValueError("Invalid value for 'method'. Use 'repeat' or 'random'.")
        setattr(self, f"{which}_spike_trains", spike_trains)
        setattr(self, f"{which}_spike_train_method", method)

    def _setup_run(self):
        allsyns = self._create_synapses()
        self.syn_lists, self.inhibitsyns = allsyns["exc"], allsyns["inh"]
        self._build_exc_fiber_cache()
        self._setup_recordings()

    def _create_synapses(self):
        exc_synlists = []
        for sl_num, section_list in enumerate(self.section_lists):
            syngroup = innervate_total(
                section_list,
                interval=self.interval,
                cycles=self.cycles,
                gmax=self.exc_gmax_per_segment[sl_num],
                stochastic=self.stochastic,
                rzero=self.exc_rzero,
            )
            exc_synlists.append(syngroup)

        inhibitsyns = innervate_points(
            self.cell.somatic[0](0.5),
            self.cell.somatic[0](0.5),
            gmax=self.inh_gmax / 2,
            number=self.cycles,
            interval=self.interval,
            tau1=0.28,
            tau2=1.85,
            e=-90,
            stochastic=self.stochastic,
            rzero=self.inh_rzero,
        )
        return {"exc": exc_synlists, "inh": inhibitsyns}

    def _build_exc_fiber_cache(self):
        self.exc_fiber_lists = []
        for section_list_idx, section_list in enumerate(self.section_lists):
            if self.fibered:
                syns_by_section = {}
                for syn in self.syn_lists[section_list_idx]:
                    syns_by_section.setdefault(syn.section, []).append(syn)

                current_synapses = []
                for path in find_nonoverlapping_paths(section_list).values():
                    fiber_syns = []
                    for sec in path:
                        fiber_syns.extend(syns_by_section.get(sec, []))
                    current_synapses.append(SynapseFiber(fiber_syns))
            else:
                current_synapses = [
                    SynapseFiber([syn]) for syn in self.syn_lists[section_list_idx]
                ]
            self.exc_fiber_lists.append(current_synapses)

    def _setup_recordings(self):
        self._v_soma = h.Vector()
        self._t_soma = h.Vector()
        self.cell.cvode.record(
            self.cell.somatic[0](0.5)._ref_v,
            self._v_soma,
            self._t_soma,
            sec=self.cell.somatic[0],
        )

        self._v_axon = None
        self._t_axon = None
        if self.record_axon:
            self._v_axon = h.Vector()
            self._t_axon = h.Vector()
            self.cell.cvode.record(
                self.cell.nodes[-1](0.5)._ref_v,
                self._v_axon,
                self._t_axon,
                sec=self.cell.nodes[-1],
            )

    def _complete_itd_sim(self, duration=None):
        self._v_soma.resize(0)
        self._t_soma.resize(0)
        if self._v_axon is not None:
            self._v_axon.resize(0)
            self._t_axon.resize(0)

        runtime = (
            self.sim_start_time + duration
            if duration is not None
            else self.sim_start_time + self.interval * self.cycles + 5
        )

        v_monitor = self._v_axon if self.record_axon else self._v_soma
        t_monitor = self._t_axon if self.record_axon else self._t_soma

        h.finitialize(self.cell.resting_potential + 1)
        h.continuerun(self.sim_start_time - 1)
        h.frecord_init()
        h.continuerun(runtime)

        return (
            v_monitor,
            t_monitor,
            self._v_soma,
            self._t_soma,
            self._v_axon,
            self._t_axon,
        )

    def _set_all_exc_syn_activation(self, itd_val=0):
        for section_list_idx, section_list in enumerate(self.section_lists):
            current_synapses = self.exc_fiber_lists[section_list_idx]
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
                self._set_indiv_exc_syn_timing(
                    fiber,
                    itd_val=(
                        itd_val if section_list == self.offset_sections else 0
                    ),
                    spike_train=spike_train,
                    gmax=self.exc_gmax_per_segment[section_list_idx],
                )
        return self.syn_lists

    def _set_indiv_exc_syn_timing(
        self, fiber, itd_val=0, spike_train=None, gmax=0.037, g_per_vesicle=False
    ):
        for syn in fiber.synapse_terminals:
            syn.gmax = gmax

            axon_delay = 0
            if self.axon_speed != 0:
                input_length = self.input_length_lookup[syn.segment]
                axon_delay = input_length / (1000 * self.axon_speed)
            syn.delay = self.sim_start_time + axon_delay + itd_val
        if spike_train is not None:
            fiber.set_firing_times(spike_train)

    def _set_all_inh_syn_activation(self, itd_val=0):
        for inhibitsyn_num in range(2):
            spike_train = None
            if hasattr(self, "inh_spike_trains"):
                spike_train = (
                    random.choice(self.inh_spike_trains)
                    if self.inh_spike_train_method == "random"
                    else self.inh_spike_trains[
                        inhibitsyn_num % len(self.inh_spike_trains)
                    ]
                )

            self._set_indiv_inhibition_activation(
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

    def _set_indiv_inhibition_activation(
        self, syn, delay=0, itd_val=0, input_length=0, offset=False, spike_train=None
    ):
        syn.delay = (
            self.sim_start_time
            + delay
            + (itd_val if offset else 0)
            + input_length / (self.axon_speed * 1000)
        )

        if spike_train is not None:
            syn.set_firing_times(spike_train)

    def _get_inhibition_input_length(self, syn, section_list):
        input_length = self.sim_start_time + (
            h.distance(closest_terminal_segment(section_list, syn.segment), syn.segment)
        )
        return input_length

    def _cross_threshold(
        self, voltage_trace, threshold=0, relative=True, time_trace=None
    ):
        if len(voltage_trace) == 0:
            if time_trace is not None:
                return 0, np.array([])
            return 0
        resting = voltage_trace[0] if relative else 0
        abs_threshold = resting + threshold if relative else threshold
        spike_bin = h.Vector().spikebin(voltage_trace, abs_threshold)
        count = int(spike_bin.sum())
        if time_trace is None:
            return count
        spike_times = np.array([])
        if count > 0:
            spike_indices = np.where(np.array(spike_bin) == 1)[0]
            spike_times = np.array(time_trace)[spike_indices]
        return count, spike_times
