import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ufpdf_xfel_scripts.lcls.run import find_nearest

mpl.rcParams.update({"text.usetex": False})


def compute_gr(q, fq, r_min, r_max, step=0.01):
    """FT by hand to preserve the morph squeezes.

    Can't use the raw output from pdfGetter for this reason.

    Parameters
    ----------
    q
      The q-array
    fq
      The fq to transform
    r_min
      The rmin
    r_max
      The rmax
    step
      The grid step size

    Returns
    -------
    """
    r = np.arange(r_min, r_max, step)
    qr = np.outer(q, r)
    integrand = fq[:, None] * np.sin(qr)
    gr = (2 / np.pi) * np.trapezoid(integrand, q, axis=0)
    return r, gr


def plot_delay_scans(scan_dict, run):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(8, 16))
    keys = [key for key in scan_dict.keys()]
    # delay_times_l1 = [delay[4] for delay in scan_dict.values()]
    delay_times_off = [delay[5] for delay in scan_dict.values()]
    delay_times_on = [delay[6] for delay in scan_dict.values()]
    delay_times_l2 = [delay[7] for delay in scan_dict.values()]
    delay_times_int = [delay[8] for delay in scan_dict.values()]
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
    key_to_color_idx = {key: i for i, key in enumerate(keys)}
    for key, delay in scan_dict.items():
        # if key == time_away_t0:
        # on_plot = scan[1]
        # off_plot = scan[2]
        color = colors[key_to_color_idx[key]]
        ax0.plot(delay[0], delay[1], label=key, color=color)
        ax1.plot(delay[0], delay[2], label=key, color=color)
        ax2.plot(delay[0], delay[3], label=key, color=color)
        if run.q_min is not None:
            ax2.axvline(x=run.q_min, color="red")
        if run.q_max is not None:
            ax2.axvline(x=run.q_max, color="red")

    q_vals = list(scan_dict.values())[0][0]
    on_minus_off_matrix = np.array([delay[3] for delay in scan_dict.values()])
    delay_times = np.array([key for key in scan_dict.keys()])
    sort_idx = np.argsort(delay_times)
    on_minus_off_matrix = on_minus_off_matrix[sort_idx]
    delay_times = delay_times[sort_idx]
    extent = [q_vals[0], q_vals[-1], delay_times[0], delay_times[-1]]
    ax3.imshow(
        on_minus_off_matrix,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
    )
    ax3.invert_yaxis()
    if run.q_min is not None:
        ax3.axvline(x=run.q_min, color="red")
    if run.q_max is not None:
        ax3.axvline(x=run.q_max, color="red")
    ax3.set_xlabel("Q [1/A]")
    ax3.set_ylabel("Time scan (ps)")
    ax3.set_title("On - Off")
    ax4.plot(keys, delay_times_off, marker="o", linestyle="-", label="off")
    ax4.plot(keys, delay_times_on, marker="o", linestyle="-", label="on")
    ax5.plot(
        keys, np.sqrt(delay_times_l2), marker="o", linestyle="-", label="diff"
    )
    ax5_twin = ax5.twinx()
    ax5_twin.plot(
        keys,
        delay_times_int,
        marker="o",
        color="red",
        linestyle="-",
        label="diff",
    )
    ax0.set_xlabel("Q [1/A]")
    ax0.set_ylabel("Pump On Intensity [a.u.]")
    ax1.set_xlabel("Q [1/A]")
    ax1.set_ylabel("Pump Off Intensity [a.u.]")
    ax2.set_xlabel("Q [1/A]")
    ax2.set_ylabel("On-Off Intensity [a.u.]")
    ax4.set_xlabel("Time scan (ps)")
    ax4.set_ylabel("Sum intensities")
    ax5.set_xlabel("Time scan (ps)")
    ax5.set_ylabel("RMS")
    ax5.legend()
    ax5_twin.legend()
    ax4.legend()
    ax0.set_title(
        f"sample = {run.sample_name}, run = {run.run_number}, "
        f"qmin = {run.q_min}, qmax = {run.q_max}"
    )
    ax1.set_title(f"run = {run.run_number}")
    ax2.set_title(f"I(q) On - I(q) Off run = {run.run_number}")
    ax5.set_title(
        f"Figure of Merit run = {run.run_number}, run.q_min = {run.q_min}, "
        f"run.q_max = {run.q_max}"
    )
    plt.tight_layout()
    plt.show()


def plot_static_scans(scan_dict, run):
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    keys = sorted(scan_dict.keys())
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
    key_to_color_idx = {key: i for i, key in enumerate(keys)}

    roi_sum_I = []
    roi_l1 = []
    roi_rms = []
    roi_int = []

    for key in keys:
        color = colors[key_to_color_idx[key]]

        q = scan_dict[key][0]
        Iq = scan_dict[key][1]
        diff = scan_dict[key][2]

        ax0.plot(q, Iq, label=str(key), color=color)
        ax1.plot(q, diff, label=str(key), color=color)

        # ROI indices
        if run.q_min is not None:
            i0 = find_nearest(q, run.q_min)
        else:
            i0 = 0
        if run.q_max is not None:
            i1 = find_nearest(q, run.q_max)
        else:
            i1 = len(q) - 1
        if i1 < i0:
            i0, i1 = i1, i0

        sl = slice(i0, i1 + 1)

        roi_sum_I.append(np.nansum(Iq[sl]))
        roi_l1.append(np.nansum(np.abs(diff[sl])))
        roi_rms.append(np.sqrt(np.nanmean(diff[sl] ** 2)))
        roi_int.append(np.nansum(diff[sl]))

    if run.q_min is not None:
        ax0.axvline(x=run.q_min, color="red")
        ax1.axvline(x=run.q_min, color="red")
    if run.q_max is not None:
        ax0.axvline(x=run.q_max, color="red")
        ax1.axvline(x=run.q_max, color="red")

    ax2.plot(keys, roi_sum_I, marker="o", linestyle="-", label="ROI sum I(q)")
    ax2.plot(keys, roi_int, marker="o", linestyle="-", label="ROI ∑ΔI")

    ax0.set_xlabel("Q [1/A]")
    ax0.set_ylabel("Intensity [a.u.]")
    ax1.set_xlabel("Q [1/A]")
    ax1.set_ylabel("I(q) - Iref(q) [a.u.]")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("ROI metric [a.u.]")

    ax0.set_title(
        f"sample = {run.sample_name}, run = {run.run_number}, "
        f"qmin = {run.q_min}, qmax = {run.q_max}"
    )
    ax1.set_title(f"I(q) - I(q)_ref run = {run.run_number}")
    ax2.set_title(
        f"ROI metrics run = {run.run_number}, run.q_min = {run.q_min}, "
        f"run.q_max = {run.q_max}"
    )

    plt.tight_layout()
    plt.show()


def plot_fq_delay_scans(fq_dict, run):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(8, 16))

    delay_keys = sorted(fq_dict.keys())
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(delay_keys))]

    for i, delay_t in enumerate(delay_keys):
        pdata = fq_dict[delay_t]
        color = colors[i]

        q = pdata["q"]
        fq_on = pdata["fq_on"]
        fq_off = pdata["fq_off"]
        diff = fq_on - fq_off

        ax0.plot(q, fq_on, color=color)
        ax1.plot(q, fq_off, color=color)
        ax2.plot(q, diff, color=color)

        if run.q_min is not None:
            ax2.axvline(x=run.q_min, color="red")
        if run.q_max is not None:
            ax2.axvline(x=run.q_max, color="red")

    q_vals = fq_dict[delay_keys[0]]["q"]
    diff_matrix = np.array(
        [fq_dict[d]["fq_on"] - fq_dict[d]["fq_off"] for d in delay_keys]
    )

    delay_times = np.array(delay_keys)
    sort_idx = np.argsort(delay_times)

    diff_matrix = diff_matrix[sort_idx]
    delay_times = delay_times[sort_idx]

    extent = [q_vals[0], q_vals[-1], delay_times[0], delay_times[-1]]

    ax3.imshow(
        diff_matrix,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
    )
    ax3.invert_yaxis()

    if run.q_min is not None:
        ax3.axvline(x=run.q_min, color="red")
    if run.q_max is not None:
        ax3.axvline(x=run.q_max, color="red")

    ax3.set_xlabel("Q [1/A]")
    ax3.set_ylabel("Time scan (ps)")
    ax3.set_title("ΔF(Q)")

    rms_vals = []
    diff_int_vals = []
    sum_on_vals = []
    sum_off_vals = []

    for delay_t in delay_times:
        pdata = fq_dict[delay_t]
        q = pdata["q"]
        diff = pdata["fq_on"] - pdata["fq_off"]

        if run.q_min is not None:
            i0 = np.abs(q - run.q_min).argmin()
        else:
            i0 = 0

        if run.q_max is not None:
            i1 = np.abs(q - run.q_max).argmin()
        else:
            i1 = len(q) - 1

        if i1 < i0:
            i0, i1 = i1, i0

        sl = slice(i0, i1 + 1)

        rms_vals.append(np.sqrt(np.mean(diff[sl] ** 2)))
        diff_int_vals.append(np.sum(diff[sl]))
        sum_on_vals.append(np.sum(pdata["fq_on"][sl]))
        sum_off_vals.append(np.sum(pdata["fq_off"][sl]))

    ax4.plot(delay_times, sum_off_vals, marker="o", label="OFF")
    ax4.plot(delay_times, sum_on_vals, marker="o", label="ON")
    ax4.set_ylabel("Sum F(Q)")
    ax4.legend()

    ax5.plot(delay_times, rms_vals, marker="o", label="sqrt(diff^2)")
    ax5_twin = ax5.twinx()
    ax5_twin.plot(
        delay_times, diff_int_vals, color="red", marker="o", label="diff"
    )
    ax5.set_xlabel("Time scan (ps)")
    ax5.set_ylabel("Metric")
    ax5.legend()
    ax5_twin.legend()

    ax0.set_ylabel("F(Q) ON")
    ax1.set_ylabel("F(Q) OFF")
    ax2.set_ylabel("ΔF(Q)")
    ax2.set_xlabel("Q [1/A]")

    ax0.set_title(
        f"sample = {run.sample_name}, run = {run.run_number}, "
        f"qmin = {run.q_min}, qmax = {run.q_max}"
    )
    ax5.set_title(
        f"Figure of Merit run = {run.run_number}, "
        f"run.q_min = {run.q_min}, run.q_max = {run.q_max}"
    )

    plt.tight_layout()
    plt.show()


def plot_reference_comparison(
    q_target,
    fq_target,
    q_morph,
    fq_morph,
    r_min=0,
    r_max=30,
    r_min_fom=None,
    r_max_fom=None,
):
    r, gr_target = compute_gr(q_target, fq_target, r_min, r_max)
    r, gr_morph = r, gr_morph = compute_gr(q_morph, fq_morph, r_min, r_max)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))

    ax0.plot(q_target, fq_target, linestyle="--", label="Synchrotron F(Q)")
    ax0.plot(q_morph, fq_morph, label="XFEL F(Q)")
    ax0.set_xlabel("Q (1/Å)")
    ax0.set_ylabel("F(Q)")
    ax0.set_title("Reference: F(Q)")
    ax0.legend()

    ax1.plot(r, gr_target, label="G(r) Synchrotron", color="orange")
    ax1.plot(r, gr_morph, label="G(r) XFEL", color="black")

    if r_min_fom is not None:
        ax1.axvline(r_min_fom, color="red")
    if r_max_fom is not None:
        ax1.axvline(r_max_fom, color="red")

    ax1.set_xlabel("r (Å)")
    ax1.set_ylabel("G(r)")
    ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_gr_function(
    gr_delay_dict,
    sample_name,
    run_number,
    r_min_fom=None,
    r_max_fom=None,
):

    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(8, 16))

    delay_keys = sorted(gr_delay_dict.keys())
    cmap = matplotlib.colormaps["viridis"]
    norm = plt.Normalize(min(delay_keys), max(delay_keys))

    for delay_t in delay_keys:
        pdata = gr_delay_dict[delay_t]
        color = cmap(norm(delay_t))

        ax0.plot(pdata["r"], pdata["gr_on"], color=color)
        ax1.plot(pdata["r"], pdata["gr_off"], color=color)
        ax2.plot(pdata["r"], pdata["diff_gr"], color=color)

    if r_min_fom is not None:
        ax2.axvline(r_min_fom, color="red")
    if r_max_fom is not None:
        ax2.axvline(r_max_fom, color="red")

    ax0.set_ylabel("G(r) ON")
    ax1.set_ylabel("G(r) OFF")
    ax2.set_ylabel("ΔG(r)")

    r_vals = gr_delay_dict[delay_keys[0]]["r"]
    diff_matrix = np.array([gr_delay_dict[d]["diff_gr"] for d in delay_keys])

    delay_times = np.array(delay_keys)
    sort_idx = np.argsort(delay_times)

    diff_matrix = diff_matrix[sort_idx]
    delay_times = delay_times[sort_idx]

    extent = [r_vals[0], r_vals[-1], delay_times[0], delay_times[-1]]

    ax3.imshow(
        diff_matrix,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
    )
    ax3.invert_yaxis()

    if r_min_fom is not None:
        ax3.axvline(r_min_fom, color="red")
    if r_max_fom is not None:
        ax3.axvline(r_max_fom, color="red")

    ax3.set_xlabel("r [Å]")
    ax3.set_ylabel("Time scan (ps)")
    ax3.set_title("ΔG(r) contour")

    sum_on = [gr_delay_dict[d]["sum_gr_on"] for d in delay_times]
    sum_off = [gr_delay_dict[d]["sum_gr_off"] for d in delay_times]
    RMS = [gr_delay_dict[d]["RMS"] for d in delay_times]
    diff_int = [gr_delay_dict[d]["diff_int"] for d in delay_times]

    ax4.plot(delay_times, sum_on, marker="o", label="Pump ON")
    ax4.plot(delay_times, sum_off, marker="o", label="Pump OFF")
    ax4.set_ylabel("Integrated G(r)")
    ax4.legend(frameon=False)

    ax5.plot(delay_times, RMS, marker="o", label="RMS")
    ax5.plot(delay_times, diff_int, marker="o", label="Integral ΔG(r)")
    ax5.set_xlabel("Delay time (ps)")
    ax5.set_ylabel("Metric")
    ax5.legend(frameon=False)

    ax0.set_title(f"sample = {sample_name}, run = {run_number}")

    plt.tight_layout()
    plt.show()


def plot_time_resolved_window_map(
    delay_dict,
    run,
    width=0.8,
    n_centers=200,
    metric="rms",  # "rms", "l1", "mean", "peak", "var"
):
    """Fixed-width sliding window. Produces FOM(center, delay) map.

    Parameters
    ----------
    delay_dict : dict
        Can be:
            run_object.morphed_delay_scans
            run_object.fq_delay_scans
            run_object.gr_delay_scans

    run : run object
        Used for title only

    width : float
        Fixed window width

    n_centers : int
        Number of window centers

    metric : str
        One of:
            "rms"   -> sqrt(mean(diff^2))
            "l1"    -> mean(|diff|)
            "mean"  -> mean(diff)
            "peak"  -> max(|diff|)
            "var"   -> variance(diff)
    """

    delay_keys = sorted(delay_dict.keys())
    delay_times = np.array(delay_keys)

    first_entry = delay_dict[delay_keys[0]]

    if isinstance(first_entry, dict):

        # G(r)
        if "r" in first_entry:
            axis_vals = first_entry["r"]

            if "diff_gr" in first_entry:
                diff_matrix = np.array(
                    [delay_dict[d]["diff_gr"] for d in delay_keys]
                )
            else:
                diff_matrix = np.array(
                    [
                        delay_dict[d]["gr_on"] - delay_dict[d]["gr_off"]
                        for d in delay_keys
                    ]
                )

            axis_label = "r [Å]"
            data_label = "G(r)"

        elif "q" in first_entry:
            axis_vals = first_entry["q"]

            diff_matrix = np.array(
                [
                    delay_dict[d]["fq_on"] - delay_dict[d]["fq_off"]
                    for d in delay_keys
                ]
            )

            axis_label = "Q [1/Å]"
            data_label = "F(Q)"

        else:
            raise ValueError("Unknown dictionary structure in delay_dict.")

    # I(q)
    else:
        axis_vals = first_entry[0]
        diff_matrix = np.array([delay_dict[d][3] for d in delay_keys])

        axis_label = "Q [1/Å]"
        data_label = "I(q)"

    sort_idx = np.argsort(delay_times)
    delay_times = delay_times[sort_idx]
    diff_matrix = diff_matrix[sort_idx]

    centers = np.linspace(
        axis_vals.min(),
        axis_vals.max(),
        n_centers,
    )

    half_w = width / 2.0
    FOM_map = np.zeros((len(delay_times), len(centers)))

    for i_c, center in enumerate(centers):

        low = center - half_w
        high = center + half_w

        mask = (axis_vals >= low) & (axis_vals <= high)

        if np.sum(mask) < 3:
            continue

        window_data = diff_matrix[:, mask]

        if metric == "rms":
            values = np.sqrt(np.mean(window_data**2, axis=1))
        elif metric == "l1":
            values = np.mean(np.abs(window_data), axis=1)
        elif metric == "mean":
            values = np.mean(window_data, axis=1)
        elif metric == "peak":
            values = np.max(np.abs(window_data), axis=1)
        elif metric == "var":
            values = np.var(window_data, axis=1)
        else:
            raise ValueError(
                "metric must be 'rms', 'l1', 'mean', 'peak', or 'var'"
            )

        FOM_map[:, i_c] = values

    fig, ax = plt.subplots(figsize=(9, 6))

    X, Y = np.meshgrid(centers, delay_times)

    im = ax.pcolormesh(
        X,
        Y,
        FOM_map,
        shading="auto",
        cmap="viridis",
    )

    ax.set_xlabel(axis_label)
    ax.set_ylabel("Delay (ps)")
    ax.set_title(
        f"{data_label} time-resolved window map\n"
        f"metric={metric}, width={width}, run={run.run_number}"
    )

        ax.set_xlabel(axis_label)
        ax.set_ylabel("Delay (ps)")
        ax.set_title(
            f"{data_label} time-resolved window map\n"
            f"metric={metric}, width={width}, run={run.run_number}"
        )

        plt.colorbar(im, ax=ax, label=metric)
        plt.tight_layout()
        plt.show()


def plot_morph_parameters(
    morph_parameters_dict, exclude_parameters=("xmin", "xmax", "xstep")
):
    """Plot morph fit parameters as a function of delay.

    Parameters
    ----------
    morph_parameters_dict : dict
        Dictionary with structure:
            {
                delay_1: [on_result_dict, off_result_dict],
                delay_2: [on_result_dict, off_result_dict],
                ...
            }

        Each result_dict contains fitted morph parameters
        such as: scale, stretch, shift, Rw, Pearson, etc.

    exclude_parameters : tuple of str
        Parameter names that should not be plotted.
    """
    sorted_delays = np.array(sorted(morph_parameters_dict.keys()))

    first_delay = sorted_delays[0]
    first_on_result = morph_parameters_dict[first_delay][0]
    morph_parameter_names = [
        name
        for name in first_on_result.keys()
        if name not in exclude_parameters
    ]

    parameter_values_on = {}
    parameter_values_off = {}

    for parameter_name in morph_parameter_names:

        on_values = []
        off_values = []

        for delay in sorted_delays:

            on_dict, off_dict = morph_parameters_dict[delay]

            on_value = on_dict.get(parameter_name)
            off_value = off_dict.get(parameter_name)

            on_values.append(np.nan if on_value is None else float(on_value))
            off_values.append(
                np.nan if off_value is None else float(off_value)
            )

        parameter_values_on[parameter_name] = np.array(on_values)
        parameter_values_off[parameter_name] = np.array(off_values)

    parameters_that_vary = []

    for parameter_name in morph_parameter_names:

        combined_values = np.concatenate(
            [
                parameter_values_on[parameter_name],
                parameter_values_off[parameter_name],
            ]
        )

        valid_values = combined_values[~np.isnan(combined_values)]

        if len(valid_values) > 0 and not np.allclose(
            valid_values, valid_values[0]
        ):
            parameters_that_vary.append(parameter_name)

    n_parameters = len(parameters_that_vary)

    ncols = 1
    nrows = n_parameters

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6, 4 * n_parameters),
    )

    axes = np.atleast_1d(axes).flatten()

    for axis, parameter_name in zip(axes, parameters_that_vary):

        axis.plot(
            sorted_delays,
            parameter_values_on[parameter_name],
            marker="o",
            label="ON",
        )

        axis.plot(
            sorted_delays,
            parameter_values_off[parameter_name],
            marker="s",
            label="OFF",
        )

        axis.set_ylabel(parameter_name)
        axis.set_title(f"{parameter_name} vs delay")
        axis.legend()

    axes[-1].set_xlabel("Delay")

    plt.tight_layout()
    plt.show()


def plot_filter_metrics(run, n_bins=80):
    """Plot diode (monitor2) and time_jitter distributions with filter
    results when available.

    Pane 1: diode histogram
      light blue = Unfiltered Shots
      red = Filtered Shots (if diode filter applied)
      dashed lines: mean (black), median (blue), threshold (red)

    Pane 2: timestamp (shifted, s) vs diode scatter
      green = Unfiltered Shots
      red = Filtered Shots (if diode filter applied)

    Pane 3: time_jitter histogram
      light blue = Unfiltered Shots
      red = Filtered Shots (if time filter applied)
      dashed red line = threshold cutoff (positive only)

    Parameters
    ----------
    run : Run
        Run object containing:
          monitor2, time_jitter, timestamp, sample_name, run_number
        And optional filter plotting state:
          plt_filter_pre_monitor2, plt_filter_keep_diode,
          plt_filter_cutoff_diode, plt_filter_pre_time,
          plt_filter_keep_time, plt_filter_cutoff_time,
          plt_filter_pre_timestamp.

    n_bins : int
        Number of bins used for the diode and time_jitter histograms.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure (closed before return).
    """
    diode_filter_enabled = hasattr(run, "plt_filter_pre_monitor2") and hasattr(
        run, "plt_filter_keep_diode"
    )
    if diode_filter_enabled:
        diode_pre = run.plt_filter_pre_monitor2
        diode_keep = run.plt_filter_keep_diode
    else:
        diode_pre = run.monitor2
        diode_keep = np.ones_like(diode_pre, dtype=bool)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(9, 13), constrained_layout=True
    )

    # Pane 1
    diode_pre_finite = diode_pre[np.isfinite(diode_pre)]
    if not diode_pre_finite.size:
        ax0.set_axis_off()
        ax0.text(
            0.5,
            0.5,
            "No finite monitor2 values",
            transform=ax0.transAxes,
            ha="center",
            va="center",
        )
    else:
        _, diode_edges = np.histogram(diode_pre_finite, bins=int(n_bins))
        diode_centers = 0.5 * (diode_edges[:-1] + diode_edges[1:])
        _, _, diode_patches = ax0.hist(
            diode_pre_finite,
            bins=diode_edges,
            edgecolor="black",
            linewidth=0.8,
        )
        for patch in diode_patches:
            patch.set_facecolor("lightblue")
        diode_cutoff = getattr(run, "plt_filter_cutoff_diode", None)
        if diode_filter_enabled:
            diode_filtered = diode_pre[~diode_keep]
            diode_filtered = diode_filtered[np.isfinite(diode_filtered)]
            if diode_filtered.size:
                bad_bins = np.digitize(diode_filtered, diode_edges) - 1
                bad_bins = np.clip(bad_bins, 0, len(diode_centers) - 1)
                for idx in np.unique(bad_bins):
                    diode_patches[int(idx)].set_facecolor("red")
        diode_mean = float(np.mean(diode_pre_finite))
        diode_median = float(np.median(diode_pre_finite))
        ax0.axvline(diode_mean, linestyle="--", color="black", linewidth=1.2)
        ax0.axvline(diode_median, linestyle="--", color="blue", linewidth=1.2)
        if diode_filter_enabled and diode_cutoff is not None:
            ax0.axvline(
                float(diode_cutoff),
                linestyle="--",
                color="red",
                linewidth=1.2,
            )
        q = getattr(run, "i0_percentile_threshold", None)
        if diode_filter_enabled and q is not None:
            thr_pct = f"{float(q):.2f}%"
        else:
            thr_pct = "None"
        ax0.text(
            0.01,
            0.93,
            (
                f"threshold: {thr_pct}\n"
                f"mean: {diode_mean:.4g}\n"
                f"median: {diode_median:.4g}"
            ),
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        unfiltered_patch = Patch(
            facecolor="lightblue",
            edgecolor="black",
            label="Unfiltered Shots",
        )
        filtered_patch = Patch(
            facecolor="red",
            edgecolor="black",
            label="Filtered Shots",
        )
        handles = [unfiltered_patch]
        if diode_filter_enabled:
            handles.append(filtered_patch)
        ax0.legend(handles=handles, loc="upper right", frameon=True)
        ax0.set_title(
            "Diode filter" if diode_filter_enabled else "Diode distribution"
        )
        ax0.set_xlabel("2BmMon Diode Intensity (J)")
        ax0.set_ylabel("Counts")

    # Pane 2
    t_ps = getattr(run, "plt_filter_pre_timestamp", None)
    if t_ps is None:
        t_ps = getattr(run, "timestamp", None)
    use_time = t_ps is not None and len(t_ps) == len(diode_pre)
    if use_time:
        t_s = t_ps.astype(float) * 1e-12
        finite_t = np.isfinite(t_s)
        if finite_t.any():
            t_s = t_s - t_s[finite_t][0]
        else:
            use_time = False
    if not use_time:
        t_s = np.arange(len(diode_pre), dtype=float)
    if diode_filter_enabled:
        h_keep = ax1.scatter(
            t_s[diode_keep],
            diode_pre[diode_keep],
            s=6,
            alpha=0.6,
            color="green",
            label="Unfiltered Shots",
        )
        h_filt = ax1.scatter(
            t_s[~diode_keep],
            diode_pre[~diode_keep],
            s=6,
            alpha=0.6,
            color="red",
            label="Filtered Shots",
        )
        ax1.legend(handles=[h_keep, h_filt], loc="upper right", frameon=True)
    else:
        h_all = ax1.scatter(
            t_s,
            diode_pre,
            s=6,
            alpha=0.6,
            color="green",
            label="Unfiltered Shots",
        )
        ax1.legend(handles=[h_all], loc="upper right", frameon=True)

    ax1.set_xlabel("Time (s)" if use_time else "Index")
    ax1.set_ylabel("2BmMon Diode Intensity (J)")
    sample = getattr(run, "sample_name", "unknown")
    run_number = getattr(run, "run_number", "unknown")
    ax1.set_title(f"sample = {sample}, run = {run_number}")

    # Pane 3
    if hasattr(run, "plt_filter_pre_time"):
        time_pre = run.plt_filter_pre_time
    else:
        time_pre = (
            run.time_jitter
            if getattr(run, "time_jitter", None) is not None
            else None
        )
    if time_pre is None:
        ax2.set_axis_off()
        ax2.text(
            0.5,
            0.5,
            "No time_jitter available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        plt.close(fig)
        return fig
    if diode_filter_enabled and len(time_pre) == len(diode_keep):
        time_to_plot = time_pre[diode_keep]
    else:
        time_to_plot = time_pre
    time_to_plot = time_to_plot[np.isfinite(time_to_plot)]
    if not time_to_plot.size:
        ax2.set_axis_off()
        ax2.text(
            0.5,
            0.5,
            "No finite time_jitter values",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        plt.close(fig)
        return fig
    _, time_edges = np.histogram(time_to_plot, bins=int(n_bins))
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    _, _, time_patches = ax2.hist(
        time_to_plot,
        bins=time_edges,
        edgecolor="black",
        linewidth=0.8,
    )
    for patch in time_patches:
        patch.set_facecolor("lightblue")
    time_cutoff = getattr(run, "plt_filter_cutoff_time", None)
    time_filter_applied = hasattr(run, "plt_filter_keep_time")
    time_filter_applied = time_filter_applied and time_cutoff is not None
    h_cut = None
    if time_filter_applied:
        cut_tt = float(time_cutoff)
        for i, patch in enumerate(time_patches):
            if float(time_centers[i]) > cut_tt:
                patch.set_facecolor("red")
        h_cut = ax2.axvline(
            cut_tt,
            linestyle="--",
            color="red",
            linewidth=1.2,
            label=f"cut={cut_tt:.4g} ps",
        )
    t_mean = float(np.mean(time_to_plot))
    t_median = float(np.median(time_to_plot))
    if time_filter_applied:
        thr_ps = f"{float(time_cutoff):.4g}"
    else:
        thr_ps = "None"
    ax2.text(
        0.01,
        0.93,
        (
            f"threshold (ps): {thr_ps}\n"
            f"mean (ps): {t_mean:.3f}\n"
            f"median (ps): {t_median:.3f}"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    unfiltered_patch = Patch(
        facecolor="lightblue",
        edgecolor="black",
        label="Unfiltered Shots",
    )
    filtered_patch = Patch(
        facecolor="red",
        edgecolor="black",
        label="Filtered Shots",
    )
    legend_handles = [unfiltered_patch]
    if time_filter_applied:
        legend_handles.append(filtered_patch)
        if h_cut is not None:
            legend_handles.append(h_cut)
    ax2.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax2.set_xlabel("Time Offset (ps)")
    ax2.set_ylabel("Counts")
    ax2.set_title(
        "Time offset (filtered)" if time_filter_applied else "Time offset"
    )

    plt.close(fig)
    return fig
