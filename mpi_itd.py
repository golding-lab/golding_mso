import os
import shutil
import golding_mso as gmso
import pandas as pd
from neuron import h
import numpy as np
import gc
import time
import random
import pickle
import contextlib
import sys

gmso.console_handler.setLevel("ERROR")
gmso.rotate_handler.setLevel("ERROR")
start_time = time.time()

# ── Configuration ────────────────────────────────────────────────────────────
SIM = True
PLOT = True # Whether to generate plots
TRACES = True # Whether to generate traces
OVERWRITE = True # Whether to overwrite existing results files
ONE_CELL_AT_A_TIME = True  # If False, ranks will split freq/trial combos across all cells. If True, each cell is done sequentially by a subset of ranks (prioritizes maintaining cell instances across multiple trials).
RESULTS_FILE = "train_itd_results.csv"
SPIKE_TIMES_FILE = "train_spike_times.csv"
PLOT_FILE_PREFIX = "frequencies"

CYCLES = 50
DURATION = None  # If None, duration will be calculated from cycles and frequency.

DELTA_VALS = np.arange(-0.75, 0.751, 0.1)  # ITD/IPD values to test
UNITS = "ms"  # "cyc" for cycles (IPD) or "ms" for milliseconds (ITD)
STAB_TIME = 100
TRIALS = 20
ANF_NUM = 5  # Number of ANF spike time sets used per trial


FREQS = [250, 500, 1000, 2000]  # List of stimulus frequencies to test
ORDERS = []  # List of branch orders to test (primary branches = 0, primary+secondary = 1, etc.). [] = full arbor innervation.
ORDER_FREQ = 500  # Frequency used when the testing branch order innervation.
EXC_GMAX = 0.12  # Total maximum conductance for all excitatory synapses
INH_GMAX = 0.12  # Total maximum conductance for all inhibitory synapses

EXC_VESICLE_POOL_SIZE = 40  # Size of the vesicle pool for excitatory synapses
INH_VESICLE_POOL_SIZE = 30  # Size of the vesicle pool for inhibitory synapses

CELLS = {
    "160112_16P": gmso.morphologies["160112_16P"]
}  # Dictionary of cell morphologies to be used in the simulations


# ── Helpers ──────────────────────────────────────────────────────────────────
def calc_duration(cycles, freq, dur=None):
    return dur if dur is not None else (cycles / (freq / 1000))


def delta_to_ms(delta, freq):
    """Convert delta value(s) to milliseconds for a given frequency."""
    if UNITS == "ms":
        return delta
    return delta * 1000.0 / freq


def delta_to_phase(delta, freq):
    """Convert delta value(s) to cycles (phase) for a given frequency."""
    if UNITS == "cyc":
        return delta
    return delta * freq / 1000.0


class _SuppressOutput:
    def write(self, x):
        pass


def create_cell(key, path):
    cell = gmso.Cell(path, compartment_size=15)
    cell.attach_axon()
    cell.__setattr__("pc_id", pc.id())
    return cell


def load_spike_times(freqs, cycles, duration, anf_num=ANF_NUM):
    raw = {
        freq: pickle.load(open(f"spikes/spikes_{freq}hz.pkl", "rb")) for freq in freqs
    }
    if anf_num is None:
        anf_num = len(raw[freqs[0]][0])
    return {
        freq: [
            np.array(
                [s for s in fiber if s < calc_duration(cycles, freq, dur=duration)]
            )
            * 1000
            for fiber in raw[freq][:anf_num]
        ]
        for freq in raw
    }


# ── Data initialisation ─────────────────────────────────────────────────────
def init_dataframe(cells, freqs, orders, phase_vals, results_file):
    itd_columns = [str(v) for v in phase_vals]
    all_columns = ["cell", "freq", "order", "vs", "trials"] + itd_columns

    if os.path.exists(results_file) and not OVERWRITE:
        df = pd.read_csv(results_file)
        for col in ["vs", "trials"] + itd_columns:
            if col not in df.columns:
                df[col] = 0
        # Initialize rows for any cell/freq combos missing from the CSV
        for cell in cells:
            for freq in freqs:
                for order in orders:
                    if not (
                        (df["cell"] == cell)
                        & (df["freq"] == freq)
                        & (df["order"] == order)
                    ).any():
                        new_row = {
                            "cell": cell,
                            "freq": freq,
                            "order": order,
                            "vs": 0.0,
                            "trials": 0,
                        }
                        for col in itd_columns:
                            new_row[col] = 0
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df[all_columns]
    else:
        rows = []
        print("cells, freqs, orders", cells, freqs, orders)
        for cell in cells:
            if len(freqs) == 0:
                freqs = [500]
            for freq in freqs:
                if len(orders) == 0:
                    row = {
                        "cell": cell,
                        "freq": freq,
                        "order": 99,
                        "vs": 0.0,
                        "trials": 0,
                    }
                    for col in itd_columns:
                        row[col] = 0
                    rows.append(row)
                else:
                    for order in orders:
                        row = {
                            "cell": cell,
                            "freq": freq,
                            "order": order,
                            "vs": 0.0,
                            "trials": 0,
                        }
                        for col in itd_columns:
                            row[col] = 0
                        rows.append(row)
        df = pd.DataFrame(rows)
    # print(df)
    df.to_csv(results_file, index=False)
    row_lookup = {}
    for idx, row in df.iterrows():
        row_lookup.setdefault(row["cell"], {}).setdefault(row["freq"], {})[
            row["order"]
        ] = idx
    # print(row_lookup)
    return df, itd_columns, row_lookup


# ── MPI task ─────────────────────────────────────────────────────────────────
def itd_batch_task_split(
    cell_name, exc_gmax, inh_gmax, trials=1, seed=None, order=0, freqs=FREQS
):
    """Run ITD sweeps for one cell. Ranks in the subworld split freq/trial pairs."""
    random.seed(seed)
    with contextlib.redirect_stdout(_SuppressOutput()):
        cell = create_cell(cell_name, gmso.morphologies[cell_name])
        if pc.id() == 0:
            print(
                f"Vrest: {cell.resting_potential}, Tau: {cell.time_constant}, Rin: {cell.input_resistance}"
            )
    cell.stabilization_time = STAB_TIME
    secs_to_use = [[], []]
    for listind, seclist in enumerate(
        [cell.lateral_nofilopodia, cell.medial_nofilopodia]
    ):
        for sec in seclist:
            parents = gmso.cell_calc.get_parent_sections(sec)
            if len(parents) <= order + 1:
                secs_to_use[listind].append(sec)
    if pc.id() != 0:
        my_tasks = [
            (delta, freq, trial)
            for delta in DELTA_VALS
            for freq in freqs
            for trial in range(int(trials))
        ][pc.id() - 1 :: pc.nhost() - 1]
    else:
        my_tasks = []
    if len(secs_to_use[0]) == 0 or len(secs_to_use[1]) == 0:
        print(
            f"Host {pc.id()} - WARNING: No sections found for cell {cell_name} at order {order}. Skipping ITD setup."
        )
        return None
    itd = gmso.sims.ITDTest(
        cell,
        secs_to_use[0],
        secs_to_use[1],
        axonspeed=1,
        record_axon=True,
        threshold=25,
        exc_gmax=EXC_GMAX,
        fibered=True,
        inhibition=True,
        inh_gmax=INH_GMAX,
        itd_vals=delta_to_ms(DELTA_VALS, min(freqs)),
        seed=int(seed),
        traces=TRACES,
        exc_rzero=EXC_VESICLE_POOL_SIZE,
        inh_rzero=INH_VESICLE_POOL_SIZE,
    )
    total_tasks = len(DELTA_VALS) * len(freqs) * int(trials)

    my_sweeps = {
        freq: {
            "spike_counts": np.zeros(len(DELTA_VALS)),
            "spike_times": {},
            "traces": {},
        }
        for freq in freqs
    }
    num_completed = 0
    for idx, (delta, freq, trial) in enumerate(my_tasks):
        duration_val = calc_duration(CYCLES, freq, dur=DURATION)
        itd.set_spike_trains(
            random.choices(SPIKE_TIMES[freq], k=ANF_NUM), which="exc", method="random"
        )
        itd.set_spike_trains(
            random.choices(SPIKE_TIMES[freq], k=ANF_NUM), which="inh", method="random"
        )
        itd.seed = int(seed * (trial + 1) * freq)
        itd_result = itd.run_at_itd(delta_to_ms(delta, freq), duration=duration_val)
        # must replicate the structure of the original result for compatibility with existing code
        my_sweeps[freq]["spike_counts"][
            np.where(DELTA_VALS == delta)[0][0]
        ] += itd_result["spike_count"]
        if my_sweeps[freq]["spike_times"].get(delta) is None:
            my_sweeps[freq]["spike_times"][delta] = itd_result["spike_times"]
        else:
            my_sweeps[freq]["spike_times"][delta] = np.concatenate(
                [my_sweeps[freq]["spike_times"][delta], itd_result["spike_times"]]
            )
        if my_sweeps[freq]["traces"].get(delta) is None and TRACES:
            my_sweeps[freq]["traces"][delta] = [itd_result["traces"]]
        elif TRACES:
            my_sweeps[freq]["traces"][delta].append(itd_result["traces"])
        num_completed += 1
        if (total_tasks // pc.nhost()) >= 10:
            if num_completed % (total_tasks // pc.nhost() // 10) == 0:
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} - Host {pc.id()} completed {num_completed}/{len(my_tasks)} tasks'
                )
            sys.stdout.flush()

    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Completed {num_completed}/{total_tasks} tasks for host {pc.id()}"
    )
    sys.stdout.flush()
    all_parts = pc.py_gather(my_sweeps, 0)
    try:
        del cell, itd
    except:
        pass
    gc.collect()

    if pc.id() == 0:
        combined = _combine_sweep_parts(all_parts)
        return cell_name, order, combined, pc.id_world()
    return None


def _combine_sweep_parts(all_parts):
    combined = {}
    for part in all_parts:
        for freq, data in part.items():
            if freq not in combined:
                combined[freq] = {
                    "spike_counts": np.zeros_like(data["spike_counts"]),
                    "spike_times": {},
                    "traces": {},
                }
            combined[freq]["spike_counts"] += data["spike_counts"]
            for itd_key, times in data["spike_times"].items():
                if itd_key in combined[freq]["spike_times"]:
                    combined[freq]["spike_times"][itd_key] = np.concatenate(
                        [combined[freq]["spike_times"][itd_key], times]
                    )
                else:
                    combined[freq]["spike_times"][itd_key] = times
            for itd_key, tr in data.get("traces", {}).items():
                if itd_key in combined[freq]["traces"]:
                    combined[freq]["traces"][itd_key].extend(tr)
                else:
                    combined[freq]["traces"][itd_key] = tr
    return combined


# ── Result accumulation ──────────────────────────────────────────────────────
def accumulate_sweep(df, row_lookup, cell_name, freq_key, order, sweep):
    spike_counts = sweep["spike_counts"]
    sweep_itd_vals = DELTA_VALS
    row_idx = row_lookup[cell_name][freq_key][order]
    for itd_idx, itd_val in enumerate(sweep_itd_vals):
        old_val = df.at[row_idx, str(itd_val)]
        df.at[row_idx, str(itd_val)] = old_val + spike_counts[itd_idx]
    df.at[row_idx, "trials"] = df.at[row_idx, "trials"] + 1
    return row_idx


# ── Plotting ─────────────────────────────────────────────────────────────────
def init_figures(cells, freqs, traces):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.ion()
    freq_cmap = mpl.colormaps["plasma"]
    freq_norm = mpl.colors.LogNorm(vmin=min(freqs), vmax=max(freqs))
    master_figs, master_axes = {}, {}
    trace_figs, trace_axes = {}, {}
    for name in cells:
        mfig, max_ = plt.subplots(1, 1, figsize=(6, 4))
        mfig.tight_layout()
        master_figs[name] = mfig
        master_axes[name] = max_
        if traces:
            tfig, tax_ = plt.subplots(
                len(freqs),
                1,
                figsize=(24, len(freqs) * 2),
                squeeze=False,
                sharex=False,
                sharey=True,
            )
            tfig.tight_layout()
            trace_figs[name] = tfig
            trace_axes[name] = tax_
    return master_figs, master_axes, trace_figs, trace_axes, freq_cmap, freq_norm


def plot_traces(
    cell_name, sweeps, trace_axes, trace_figs, freqs, vs=None, only_aps=False, itds=None
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.cm as cm

    for freq_idx, freq in enumerate(freqs):
        if freq not in sweeps:
            continue
        trace_data = sweeps[freq].get("traces")
        print(
            f"Plotting traces for {cell_name} at {freq} Hz: {len(trace_data)} ITD values"
        )
        if not trace_data:
            continue
        tax = trace_axes[cell_name][freq_idx, 0]
        title = f"{cell_name[:6]} {freq} Hz - "
        if vs is not None and freq in vs:
            title += f"VS={vs[freq]:.4f}"
        tax.set_title(title, fontsize=7)
        for idx, (itd, tr_list) in enumerate(trace_data.items()):
            if itds is not None and itd not in itds:
                continue
            for tr_idx, tr in enumerate(tr_list):
                if only_aps and max(tr["voltage_axon"]) - min(tr["voltage_axon"]) < 20:
                    continue
                tax.plot(
                    tr["time_soma"],
                    np.array(tr["voltage_soma"]) - tr["voltage_soma"][0],
                    color=cm.gist_rainbow(idx / len(trace_data.keys())),
                    linewidth=0.5,
                    alpha=0.75,
                )
        tax.set_ylabel("mV", fontsize=6)
        if freq_idx == len(freqs) - 1:
            tax.set_xlabel("Time (ms)", fontsize=6)
        handles, labels = zip(
            *[
                (
                    Line2D(
                        [],
                        [],
                        color=cm.gist_rainbow(idx / len(trace_data.keys())),
                        linewidth=0.5,
                    ),
                    f"ITD: {itd} {UNITS}",
                )
                for idx, itd in enumerate(
                    list(trace_data.keys())[:1]
                    + list(trace_data.keys())[
                        len(trace_data.keys()) // 2 : len(trace_data.keys()) // 2 + 1
                    ]
                    + list(trace_data.keys())[-1:]
                )
            ]
        )
        tax.legend(handles, labels, loc="upper right")
    trace_figs[cell_name].tight_layout()
    plt.pause(0.05)


def plot_itd_curves(
    cell_name,
    df,
    row_lookup,
    itd_columns,
    freqs,
    orders,
    master_figs,
    master_axes,
    freq_cmap,
    freq_norm,
):
    import matplotlib.pyplot as plt

    ax = master_axes[cell_name]
    ax.clear()
    ax.set_title(f"{cell_name}", fontsize=7)
    for freq in freqs:
        for order in orders:
            row_idx = row_lookup[cell_name][freq][order]
            spike_data = df.loc[row_idx, itd_columns].to_numpy(dtype=float)
            peak = spike_data.max()
            if peak == 0:
                continue
            normed = spike_data / peak
            color = freq_cmap(freq_norm(freq))
            ax.plot(
                DELTA_VALS,
                normed,
                color=color,
                alpha=1.0,
                linewidth=2.0,
                label=f'{freq} Hz (Max: {peak:.0f} @ {DELTA_VALS[np.argmax(spike_data)]:.2f} {UNITS}) VS: {df.loc[row_idx, "vs"]:.2f}',
            )
    ax.set_ylim(bottom=0, top=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Phase (cycles)" if UNITS == "cyc" else "ITD (ms)", fontsize=6)
    ax.set_ylabel("Norm. spike count", fontsize=6)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h_item, l_item in zip(handles, labels):
        if l_item not in seen:
            seen[l_item] = h_item
    ax.legend(seen.values(), seen.keys(), fontsize=5, loc="upper right")
    master_figs[cell_name].tight_layout()
    plt.pause(0.1)


def accumulate_master_trace(master_traces, cell_name, freq_key, delta_val, trace_list):
    """Average a batch's soma/axon traces (resampled onto a common time axis) into master_traces."""
    if freq_key not in master_traces[cell_name]:
        master_traces[cell_name][freq_key] = {}
    if delta_val not in master_traces[cell_name][freq_key]:
        master_traces[cell_name][freq_key][delta_val] = {
            "voltage_soma": np.zeros(100000),
            "voltage_axon": np.zeros(100000),
        }
    trace_count = 0
    for trace in trace_list:
        trace_count += 1
        for time_key, voltage_key in [
            ["time_soma", "voltage_soma"],
            ["time_axon", "voltage_axon"],
        ]:
            interp_tr = np.interp(
                np.linspace(
                    99, calc_duration(CYCLES, freq_key, DURATION) + 99 + 2, 100000
                ),
                trace.get(time_key, []),
                trace.get(voltage_key, []),
            )
            master_traces[cell_name][freq_key][delta_val][voltage_key] += (
                interp_tr - interp_tr[0]
            )
    master_traces[cell_name][freq_key][delta_val]["voltage_soma"] /= trace_count
    master_traces[cell_name][freq_key][delta_val]["voltage_axon"] /= trace_count


# ── MPI bootstrap ───────────────────────────────────────────────────────────
h.nrnmpi_init()
pc = h.ParallelContext()
SUBWORLD_SIZE = max(1, pc.nhost_world() // len(CELLS))
SUBWORLD_SIZE = pc.nhost_world() if ONE_CELL_AT_A_TIME else SUBWORLD_SIZE

pc.subworlds(SUBWORLD_SIZE)

if pc.id_world() == 0:
    SEED = random.randint(0, 10000)
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(
        f"Simulation started with SEED={SEED}, PLOT={PLOT}, SIM={SIM}, TRACES={TRACES}"
    )
    print(
        f"Subworlds: {pc.nhost_bbs()} subworlds of {SUBWORLD_SIZE} ranks "
        f"({pc.nhost_world()} total MPI ranks)"
    )
    if not len(DELTA_VALS) < 2:
        print(
            f"Cells: {list(CELLS.keys())}\nFrequencies: {FREQS}\nCycles: {CYCLES}\nDeltas ({UNITS}): ({DELTA_VALS[0]:.2f} > +{DELTA_VALS[1]-DELTA_VALS[0]:.2f} > {DELTA_VALS[-1]:.2f})\nTrials per frequency: {TRIALS}"
        )

    sys.stdout.flush()

SPIKE_TIMES = load_spike_times(FREQS, CYCLES, DURATION)

# ── Main (master rank only) ─────────────────────────────────────────────────
pc.runworker()

df, itd_columns, row_lookup = init_dataframe(
    CELLS, FREQS, ORDERS, DELTA_VALS, RESULTS_FILE
)

trial_count = 0
spike_times_rows = []
master_traces = {cell: {} for cell in CELLS.keys()}
if PLOT and pc.id_world() == 0:
    import matplotlib.pyplot as plt

    master_figs, master_axes, trace_figs, trace_axes, freq_cmap, freq_norm = (
        init_figures(CELLS, FREQS, TRACES)
    )
    for cell in df["cell"].unique():
        plot_itd_curves(
            cell_name=cell,
            df=df,
            row_lookup=row_lookup,
            itd_columns=itd_columns,
            freqs=df["freq"].unique(),
            orders=ORDERS if len(ORDERS) > 0 else [99],
            master_figs=master_figs,
            master_axes=master_axes,
            freq_cmap=freq_cmap,
            freq_norm=freq_norm,
        )

if SIM:
    cond_df = pd.read_csv("inh_mag_test.csv")
    rng = np.random.default_rng(SEED)
    cell_reorder = rng.permutation(list(CELLS.keys()))
    for ind, cell in enumerate(cell_reorder):
        inh_gmax = cond_df.loc[cond_df["cell"] == cell, "3"].values[0]
        if len(ORDERS) > 0:
            for order in ORDERS:
                seed = int((ind + 1) * SEED)
                pc.submit(itd_batch_task_split, cell, TRIALS, seed, order, [ORDER_FREQ])
        else:
            seed = int((ind + 2) * SEED)
            pc.submit(itd_batch_task_split, cell, TRIALS, seed, 99, FREQS)

    total_batches = len(CELLS)
    batch_count = 0
    frame_num = 0

    while pc.working():
        result = pc.pyret()
        if result is None:
            continue
        batch_count += 1
        cell_name, order, sweeps, world_id = result

        # ── Accumulate results & compute VS ──
        for freq_key, sweep in sweeps.items():
            row_idx = accumulate_sweep(
                df, row_lookup, cell_name, freq_key, int(order), sweep
            )
            trial_count += len(DELTA_VALS)

            # ── Collect spike times ──
            for delta_val, st in sweep.get("spike_times", {}).items():
                if st is not None and len(st) > 0:
                    for t in st:
                        spike_times_rows.append(
                            {
                                "cell": cell_name,
                                "order": order,
                                "freq": freq_key,
                                "delta": delta_val,
                                "spike_time_ms": t,
                            }
                        )

        # ── Plot ──
        if PLOT:
            if TRACES:
                for freq_key, sweep_data in sweeps.items():
                    for delta_val, trace_list in sweep_data.get("traces", {}).items():
                        accumulate_master_trace(
                            master_traces, cell_name, freq_key, delta_val, trace_list
                        )
                plot_traces(
                    cell_name,
                    sweeps,
                    trace_axes,
                    trace_figs,
                    FREQS,
                    vs=df.loc[df["cell"] == cell_name, ["freq", "vs"]]
                    .set_index("freq")
                    .to_dict()["vs"],
                    only_aps=False,
                )

            plot_itd_curves(
                cell_name,
                df,
                row_lookup,
                itd_columns,
                FREQS,
                ORDERS if len(ORDERS) > 0 else [99],
                master_figs,
                master_axes,
                freq_cmap,
                freq_norm,
            )
            plt.pause(0.1)

        df.to_csv(RESULTS_FILE, index=False, mode="w")
        frame_num += 1

if PLOT and TRACES:
    master_trace_fig, master_trace_ax = plt.subplots(len(FREQS), 2, figsize=(10, 8))
    if len(FREQS) == 1:
        master_trace_ax = np.expand_dims(master_trace_ax, axis=0)
    for cell_name in master_traces:
        for freq_ind, freq_key in enumerate(master_traces[cell_name]):
            for delta_val in master_traces[cell_name][freq_key]:
                x = master_traces[cell_name][freq_key][delta_val]["voltage_soma"]
                x_ax = master_traces[cell_name][freq_key][delta_val]["voltage_axon"]
                master_trace_ax[freq_ind, 0].plot(
                    np.linspace(
                        99, calc_duration(CYCLES, freq_key, DURATION) + 99 + 2, 100000
                    ),
                    x,
                    label=f"Freq: {freq_key}, Delta: {delta_val}",
                )
                master_trace_ax[freq_ind, 1].plot(
                    np.linspace(
                        99, calc_duration(CYCLES, freq_key, DURATION) + 99 + 2, 100000
                    ),
                    x_ax,
                    label=f"Freq: {freq_key}, Delta: {delta_val}",
                )
    master_trace_ax[0, 0].set_title("Soma Voltage")
    master_trace_ax[0, 1].set_title("Axon Voltage")
    for freq_ind in range(len(FREQS)):
        master_trace_ax[freq_ind, 0].set_xlabel("Time (ms)")
        master_trace_ax[freq_ind, 1].set_xlabel("Time (ms)")
        master_trace_ax[freq_ind, 0].set_ylabel("Voltage (mV)")
        master_trace_ax[freq_ind, 1].set_ylabel("Voltage (mV)")
        master_trace_ax[freq_ind, 0].legend()
        master_trace_ax[freq_ind, 1].legend()
    master_trace_fig.savefig(f"{PLOT_FILE_PREFIX}_final_master_traces.pdf")

# ── Final summary ────────────────────────────────────────────────────────────
for cell_name in CELLS:
    for freq in FREQS:
        elapsed = CYCLES * (1 / freq) if not DURATION else DURATION
        rate = df.loc[
            (df["freq"] == freq) & (df["cell"] == cell_name), itd_columns
        ].sum(axis=1).max() / (elapsed * TRIALS)
        print(
            f"Final Rate for Cell: {cell_name}, Freq: {freq} Hz: {rate:.4f} spikes/ms VS: {df.loc[(df['freq'] == freq) & (df['cell'] == cell_name), 'vs'].values[0]:.4f}"
        )

df.to_csv(RESULTS_FILE, index=False, mode="w")
if SIM and spike_times_rows:
    st_df = pd.DataFrame(spike_times_rows)
    st_df.to_csv(SPIKE_TIMES_FILE, index=False)
    print(f"Spike times saved to {SPIKE_TIMES_FILE} ({len(st_df)} spikes)")
if PLOT:
    for name in CELLS:
        master_figs[name].savefig(f"{PLOT_FILE_PREFIX}_final_results_{name}.pdf")
        if SIM and TRACES:
            trace_figs[name].savefig(f"{PLOT_FILE_PREFIX}_final_traces_{name}.pdf")
    plt.show(block=True)
pc.done()

if pc.nhost_world() > 24:
    shutil.rmtree("/spikes")
