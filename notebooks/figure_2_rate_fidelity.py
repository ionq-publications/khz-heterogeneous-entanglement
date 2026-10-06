"""Plot the entangled state delivery time and the rate-fidelity tradeoff.

Copyright (c) 2026 IonQ, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# %% [markdown]
# # Figure 2: Rate and fidelity

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from analysis.tomography import (
    calculate_histogram_population_and_correlator,
    make_two_qubit_state,
)
from papertools import (
    BAR_EDGE,
    BAR_FILL,
    BAR_STYLE,
    DOUBLE_COLUMN_WIDTH_MM,
    ERRORBAR_STYLE,
    FIGURES_DIR,
    IONQ_ORANGE,
    IONQ_ORANGE_EDGE,
    MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
    load_dataset,
    load_provenance,
    save_figure,
)
from papertools.formatting import format_uncertainty
from papertools.run_ids import ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, RATE_FIDELITY_TRADEOFF

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Entangled state delivery time

# %%
rate_ds = load_dataset(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "rate.h5")
waiting_time_s = rate_ds.readout_timestamp - rate_ds.repetition_start_timestamp
waiting_time_s = waiting_time_s.values.ravel()
assert np.isfinite(waiting_time_s).all(), (
    "Time-to-success data contain non-finite values"
)
mean_waiting_time_s = waiting_time_s.mean()
rate = 1 / mean_waiting_time_s
# Exponential waiting times have SEM(t) = mean(t)/sqrt(N). First-order
# propagation through R = 1/mean(t) gives sigma_R = R/sqrt(N). Statistical error only.
mean_waiting_time_error_s = mean_waiting_time_s / np.sqrt(waiting_time_s.size)
rate_error_hz = rate / np.sqrt(waiting_time_s.size)
waiting_time_ms = waiting_time_s * 1000
main_ds = load_dataset(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "main.h5")
attempts_per_success = main_ds["nof_attempts_this_success"].values.ravel()
assert np.isfinite(attempts_per_success).all(), (
    "Attempts-per-success data contain non-finite values"
)
mean_attempts = attempts_per_success.mean()
# Exponential approximation for independent attempt counts with a fixed
# success probability: SEM(attempts) = mean(attempts)/sqrt(N).
mean_attempts_error = mean_attempts / np.sqrt(attempts_per_success.size)

# %%
WAITING_TIME_PANEL_RATIO = 0.25
WAITING_TIME_WIDTH = DOUBLE_COLUMN_WIDTH_MM * WAITING_TIME_PANEL_RATIO * MM
WAITING_TIME_FIGSIZE = (WAITING_TIME_WIDTH, WAITING_TIME_WIDTH * 5 / 6.5)
WAITING_TIME_BINS = 30
WAITING_TIME_ATTEMPTS_BINS = 30
WAITING_TIME_MEAN_LINEWIDTH = 0.8
WAITING_TIME_INSET_BOUNDS = (0.45, 0.34, 0.46, 0.43)
WAITING_TIME_INSET_FONT_SIZE = SMALL_FONT_SIZE

# %%
waiting_time_counts, waiting_time_edges = np.histogram(
    waiting_time_ms, bins=WAITING_TIME_BINS, range=(0, waiting_time_ms.max())
)
attempt_counts, attempt_edges = np.histogram(
    attempts_per_success,
    bins=WAITING_TIME_ATTEMPTS_BINS,
    range=(0, attempts_per_success.max()),
)
fig, ax = plt.subplots(figsize=WAITING_TIME_FIGSIZE)
ax.stairs(
    waiting_time_counts,
    waiting_time_edges,
    fill=True,
    baseline=0,
    **BAR_STYLE,
    zorder=2,
)
ax.set_ylim(bottom=0)
ax.axvline(
    waiting_time_ms.mean(),
    color=IONQ_ORANGE,
    linestyle="--",
    linewidth=WAITING_TIME_MEAN_LINEWIDTH,
    zorder=3,
)
ax.set_xlim(0, waiting_time_edges[-1])
ax.set_xlabel("Time to success (ms)")
ax.set_ylabel("Occurrences")
ax.text(
    0.94,
    0.94,
    f"Mean time: {format_uncertainty(mean_waiting_time_s * 1e6, mean_waiting_time_error_s * 1e6)} µs",
    transform=ax.transAxes,
    ha="right",
    va="top",
)

waiting_time_inset = ax.inset_axes(
    WAITING_TIME_INSET_BOUNDS, facecolor="white", zorder=5
)
waiting_time_inset.stairs(
    attempt_counts,
    attempt_edges,
    fill=True,
    baseline=0,
    **BAR_STYLE,
    zorder=2,
)
waiting_time_inset.set_yscale("log")
waiting_time_inset.minorticks_off()
waiting_time_inset.set_ylim(bottom=0.8)
waiting_time_inset.set_xlim(0, attempt_edges[-1])
waiting_time_inset.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
waiting_time_inset.tick_params(
    axis="both",
    which="both",
    labelsize=WAITING_TIME_INSET_FONT_SIZE,
    pad=2,
)
waiting_time_inset.set_xlabel(
    "Attempts per success", fontsize=WAITING_TIME_INSET_FONT_SIZE, labelpad=1, x=0.4
)
waiting_time_inset.set_ylabel(
    "Occurrences", fontsize=WAITING_TIME_INSET_FONT_SIZE, labelpad=1
)

save_figure(fig, "figure_2_rate")

lines = [
    "Figure 2: Entangled state delivery time",
    "",
    f"Number of events: {waiting_time_ms.size}",
    f"Mean time: {format_uncertainty(mean_waiting_time_s * 1e6, mean_waiting_time_error_s * 1e6)} µs",
    f"Rate: {format_uncertainty(rate, rate_error_hz)} Hz",
    "",
    f"{'Bin start (ms)':>18} {'Bin end (ms)':>18} {'Occurrences':>11}",
]
for left, right, count in zip(
    waiting_time_edges[:-1], waiting_time_edges[1:], waiting_time_counts
):
    lines.append(f"{left:18.8f} {right:18.8f} {count:11d}")
lines.extend(
    [
        "",
        "Attempts per success",
        f"Number of events: {attempts_per_success.size}",
        f"Mean attempts: {format_uncertainty(mean_attempts, mean_attempts_error)}",
        "",
        f"{'Bin start':>18} {'Bin end':>18} {'Occurrences':>11}",
    ]
)
for left, right, count in zip(attempt_edges[:-1], attempt_edges[1:], attempt_counts):
    lines.append(f"{left:18.8f} {right:18.8f} {count:11d}")
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_2_rate.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Rate-fidelity tradeoff
#
# Fidelity is (1 + XX - YY + ZZ) / 4. Independent correlator errors are
# combined in quadrature; the negative YY coefficient swaps its error sides.
# Rate is the inverse mean time to success, with uncertainty rate / sqrt(N).

# %%
tradeoff_results = {}
for run_id in RATE_FIDELITY_TRADEOFF:
    provenance = load_provenance(run_id)
    with load_dataset(run_id, "main.h5") as tradeoff_main_ds:
        ion_siv_state = make_two_qubit_state(
            state_1=tradeoff_main_ds["ion_state"],
            state_1_high_value=1,
            state_2=tradeoff_main_ds["siv_final_readout_state"],
            state_2_high_value=1,
            name="ion_siv",
            label="Ion-SiV",
        )
        _, _, ion_siv_corr = calculate_histogram_population_and_correlator(
            ion_siv_state, label="Ion-SiV", prefix="ion_siv", rep_dim="success_idx"
        )
    ion_siv_corr = ion_siv_corr.sel(correlator=["XX", "YY", "ZZ"])
    mean = ion_siv_corr["ion_siv_corr_mean"]
    minus = ion_siv_corr["ion_siv_corr_minus_error"]
    plus = ion_siv_corr["ion_siv_corr_plus_error"]
    fidelity = (
        1
        + mean.sel(correlator="XX")
        - mean.sel(correlator="YY")
        + mean.sel(correlator="ZZ")
    ).item() / 4
    # First-order propagation for independent XX, YY, ZZ measurements:
    # sigma_F = sqrt(sigma_XX**2 + sigma_YY**2 + sigma_ZZ**2) / 4.
    # Approximate each asymmetric posterior width as a directional standard
    # error; the minus sign on YY swaps its upper/lower contributions.
    fidelity_minus = (
        np.sqrt(
            minus.sel(correlator="XX") ** 2
            + plus.sel(correlator="YY") ** 2
            + minus.sel(correlator="ZZ") ** 2
        ).item()
        / 4
    )
    fidelity_plus = (
        np.sqrt(
            plus.sel(correlator="XX") ** 2
            + minus.sel(correlator="YY") ** 2
            + plus.sel(correlator="ZZ") ** 2
        ).item()
        / 4
    )
    with load_dataset(run_id, "rate.h5") as tradeoff_rate_ds:
        success_times = (
            tradeoff_rate_ds.readout_timestamp
            - tradeoff_rate_ds.repetition_start_timestamp
        )
        assert np.isfinite(success_times.values).all(), (
            f"Run {run_id}: time-to-success data contain non-finite values"
        )
        sample_count = success_times.size
        rate_hz = 1 / success_times.mean().item()
    tradeoff_results[run_id] = {
        "max_attempts_per_cooling": provenance["metadata"]["max_attempts_per_cooling"],
        "correlator": ion_siv_corr,
        "fidelity": fidelity,
        "fidelity_minus_error": fidelity_minus,
        "fidelity_plus_error": fidelity_plus,
        "rate_hz": rate_hz,
        "rate_error_hz": rate_hz / np.sqrt(sample_count),
        "rate_sample_count": sample_count,
    }

# %%
TRADEOFF_PANEL_RATIO = 0.25
TRADEOFF_WIDTH = DOUBLE_COLUMN_WIDTH_MM * TRADEOFF_PANEL_RATIO * MM
TRADEOFF_FIGSIZE = (TRADEOFF_WIDTH, TRADEOFF_WIDTH * 5 / 6.5)
TRADEOFF_MARKER_SIZE = 3
TRADEOFF_MARKER_EDGE_WIDTH = 0.6
TRADEOFF_XLIM = (4.5, 168)
TRADEOFF_FIDELITY_YLIM = (0.70, 0.90)
TRADEOFF_RATE_YLIM = (0.65, 1.45)

# %%
# Maximum number of attempts until reset (x axis of Fig. 2d).
#
# max_attempts_per_cooling = M = 8N - 1 for XY8-N. In these runs, the first
# attempt of each block could not herald because of a sequence-script error,
# so we plot the M - 1 attempts that can herald. See SM Sec. S8, "Entanglement
# attempt sequence".
tradeoff_attempts = (
    np.array([r["max_attempts_per_cooling"] for r in tradeoff_results.values()]) - 1
)
assert np.array_equal(tradeoff_attempts, [6, 14, 22, 30, 62, 126]), (
    "Unexpected tradeoff run order or attempt counts"
)
tradeoff_fidelity = np.array([r["fidelity"] for r in tradeoff_results.values()])
tradeoff_fidelity_errors = np.array(
    [
        [r["fidelity_minus_error"] for r in tradeoff_results.values()],
        [r["fidelity_plus_error"] for r in tradeoff_results.values()],
    ]
)
tradeoff_rate_khz = np.array([r["rate_hz"] for r in tradeoff_results.values()]) / 1000
tradeoff_rate_error_khz = (
    np.array([r["rate_error_hz"] for r in tradeoff_results.values()]) / 1000
)

fig, ax = plt.subplots(figsize=TRADEOFF_FIGSIZE)
ax.errorbar(
    tradeoff_attempts,
    tradeoff_fidelity,
    yerr=tradeoff_fidelity_errors,
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=2,
)
ax.plot(
    tradeoff_attempts,
    tradeoff_fidelity,
    marker="o",
    linestyle="none",
    color=BAR_FILL,
    markeredgecolor=BAR_EDGE,
    markersize=TRADEOFF_MARKER_SIZE,
    markeredgewidth=TRADEOFF_MARKER_EDGE_WIDTH,
    zorder=3,
)
ax.set_xlabel("Maximum attempts until reset", x=0.45)
ax.set_ylabel("Fidelity", color=BAR_EDGE)
ax.set_ylim(*TRADEOFF_FIDELITY_YLIM)
ax.tick_params(axis="y", colors=BAR_EDGE, right=False)
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64, 128])
ax.xaxis.set_major_formatter(ScalarFormatter())

rate_ax = ax.twinx()
rate_ax.errorbar(
    tradeoff_attempts,
    tradeoff_rate_khz,
    yerr=tradeoff_rate_error_khz,
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=2,
)
rate_ax.plot(
    tradeoff_attempts,
    tradeoff_rate_khz,
    marker="o",
    linestyle="none",
    color=IONQ_ORANGE,
    markeredgecolor=IONQ_ORANGE_EDGE,
    markersize=TRADEOFF_MARKER_SIZE,
    markeredgewidth=TRADEOFF_MARKER_EDGE_WIDTH,
    zorder=3,
)
rate_ax.set_ylabel("Rate (kHz)", color=IONQ_ORANGE)
rate_ax.set_ylim(*TRADEOFF_RATE_YLIM)
rate_ax.tick_params(axis="y", colors=IONQ_ORANGE, left=False, right=True)
ax.set_xlim(*TRADEOFF_XLIM)
save_figure(fig, "figure_2_rate_fidelity_tradeoff")

lines = [
    "Figure 2: Rate-fidelity tradeoff",
    "",
    f"{'Run':>8} {'Maximum attempts until reset':>28} "
    f"{'XX':>12} {'YY':>12} {'ZZ':>12} "
    f"{'Fidelity':>14} {'Lower error':>14} {'Upper error':>14} "
    f"{'Rate (kHz)':>16} {'Rate error (kHz)':>18} {'N':>8}",
]
for run_id, result in tradeoff_results.items():
    mean = result["correlator"]["ion_siv_corr_mean"]
    lines.append(
        f"{run_id:8d} {result['max_attempts_per_cooling'] - 1:28d} "
        f"{mean.sel(correlator='XX').item():12.8f} {mean.sel(correlator='YY').item():12.8f} "
        f"{mean.sel(correlator='ZZ').item():12.8f} {result['fidelity']:14.8f} "
        f"{result['fidelity_minus_error']:14.8f} {result['fidelity_plus_error']:14.8f} "
        f"{result['rate_hz'] / 1000:16.8f} {result['rate_error_hz'] / 1000:18.8f} "
        f"{result['rate_sample_count']:8d}"
    )
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_2_rate_fidelity_tradeoff.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
