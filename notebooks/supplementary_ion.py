"""Plot the ion population sweep, ion readout, and ion-photon correlations.

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
# # Supplementary ion figures

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lmfit.models import ConstantModel, GaussianModel, SineModel
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from scipy.optimize import minimize_scalar
from scipy.stats import norm

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
    SINGLE_COLUMN_WIDTH_MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
    load_dataset,
    save_figure,
)
from papertools.formatting import format_fit_report, format_uncertainty
from papertools.run_ids import (
    ION_493_PHOTON_IMPROVED_EXTINCTION_XX_SWEEP,
    ION_493_PHOTON_IMPROVED_EXTINCTION_ZZ_SWEEP,
    ION_493_PHOTON_XX_PHASE_COMP_OFF,
    ION_493_PHOTON_XX_PHASE_COMP_ON,
    ION_493_PHOTON_XX_SWEEP,
    ION_493_PHOTON_ZZ_SWEEP,
    ION_493_PULSE_AMPLITUDE_SWEEP,
    ION_737_PHOTON_OFF_RESONANT_XX_SWEEP,
    ION_737_PHOTON_OFF_RESONANT_ZZ_SWEEP,
    ION_READOUT_HISTOGRAM,
)

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Ion populations versus 493 nm pulse amplitude

# %%
pulse_ds = load_dataset(ION_493_PULSE_AMPLITUDE_SWEEP, "bright_state_probability.h5")
assert np.isfinite(pulse_ds.to_array().values).all(), (
    "Pulse-sweep probabilities contain non-finite values"
)
amplitude = pulse_ds.amplitude_scale.values
p_plus = pulse_ds.bright_state_probability_p1p3.values
p_minus = pulse_ds.bright_state_probability_m1m3.values
p_both = pulse_ds.bright_state_probability_both.values
error_plus = pulse_ds.bright_state_probability_error_p1p3.values
error_minus = pulse_ds.bright_state_probability_error_m1m3.values
error_both = pulse_ds.bright_state_probability_error_both.values

# The three readout settings are independent measurements, so errors add in quadrature.
populations = {
    "D3/2": (p_both, error_both),
    "S+1/2": (p_minus - p_both, np.sqrt(error_minus**2 + error_both**2)),
    "S-1/2": (p_plus - p_both, np.sqrt(error_plus**2 + error_both**2)),
}

# %%
ION_PULSE_FIGSIZE = (SINGLE_COLUMN_WIDTH_MM * MM, SINGLE_COLUMN_WIDTH_MM * MM * 5 / 6.5)
ION_PULSE_IDEAL_POPULATION = 0.488
ION_PULSE_MARKER_SIZE = 3
ION_PULSE_MARKER_EDGE_WIDTH = 0.6
ION_PULSE_SERIES = (
    ("D3/2", r"$D_{3/2}$", BAR_FILL, BAR_EDGE),
    ("S+1/2", r"$S_{1/2},\ m=+1/2$", IONQ_ORANGE, IONQ_ORANGE_EDGE),
    ("S-1/2", r"$S_{1/2},\ m=-1/2$", "#009e73", "#006646"),
)

# %%
fig, ax = plt.subplots(figsize=ION_PULSE_FIGSIZE)
ax.axhline(
    ION_PULSE_IDEAL_POPULATION,
    color=IONQ_ORANGE,
    linestyle="--",
    linewidth=0.8,
    label=r"Ideal $S_{1/2},\ m=+1/2$",
)
for key, label, color, edge in ION_PULSE_SERIES:
    population, error = populations[key]
    ax.errorbar(
        amplitude,
        population,
        yerr=error,
        fmt="o",
        label=label,
        color=color,
        markeredgecolor=edge,
        markersize=ION_PULSE_MARKER_SIZE,
        markeredgewidth=ION_PULSE_MARKER_EDGE_WIDTH,
        **ERRORBAR_STYLE,
        zorder=3,
    )
for population in (0, 1):
    ax.axhline(population, color="0.65", linestyle="--", linewidth=0.5, zorder=0)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("493 nm pulse amplitude scale (a.u.)")
ax.set_ylabel("Population")
ax.legend(loc="center left", frameon=True)
save_figure(fig, "supplementary_ion_493_pulse_amplitude_sweep")

lines = [
    "Supplementary ion: 493 nm pulse amplitude sweep",
    "",
    f"{'Amplitude':>12}"
    + "".join(f"{key:>14} {key + ' error':>14}" for key in populations),
]
for i, scale in enumerate(amplitude):
    lines.append(
        f"{scale:12.8f}"
        + "".join(
            f"{population[i]:14.8f} {error[i]:14.8f}"
            for population, error in populations.values()
        )
    )
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_ion_493_pulse_amplitude_sweep.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Ion readout histogram

# %%
readout_ds = load_dataset(ION_READOUT_HISTOGRAM, "ion_roi.h5")
photoelectrons = readout_ds.ion_roi_photoelectrons
assert np.isfinite(photoelectrons.values).all(), (
    "Readout photoelectrons contain non-finite values"
)
dark_readout = photoelectrons.sel(pump_polarization=1).values
bright_readout = photoelectrons.sel(pump_polarization=-1).values

# %%
ION_READOUT_FIGSIZE = (DOUBLE_COLUMN_WIDTH_MM * MM, 75 * MM)
ION_READOUT_BIN_WIDTH = 1.0
ION_READOUT_YLIM = (0.8, 1e4)
ION_READOUT_FIT_LINEWIDTH = 0.8
ION_READOUT_FILL_ALPHA = 0.4

# %%
readout_edges = np.arange(
    np.floor(photoelectrons.min().item()),
    np.ceil(photoelectrons.max().item()) + ION_READOUT_BIN_WIDTH,
    ION_READOUT_BIN_WIDTH,
)
readout_centers = (readout_edges[:-1] + readout_edges[1:]) / 2
dark_counts, _ = np.histogram(dark_readout, bins=readout_edges)
bright_counts, _ = np.histogram(bright_readout, bins=readout_edges)
readout_shots = dark_readout.size + bright_readout.size
readout_density = (dark_counts + bright_counts) / (
    readout_shots * ION_READOUT_BIN_WIDTH
)

# Probability density: w * N(x; mu_dark, sigma_dark) + (1-w) * N(x; mu_bright, sigma_bright).
# Fit the weight, both means, and both widths; component areas sum to one.
# Means and widths are in photoelectrons; the weight is dimensionless.
readout_model = GaussianModel(prefix="dark_") + GaussianModel(prefix="bright_")
readout_params = readout_model.make_params(
    dark_amplitude={"value": 0.5, "min": 0, "max": 1},
    bright_amplitude={"expr": "1 - dark_amplitude"},
    dark_center=np.median(dark_readout),
    dark_sigma=dark_readout.std(),
    bright_center=np.median(bright_readout),
    bright_sigma=bright_readout.std(),
)

readout_fit = readout_model.fit(readout_density, readout_params, x=readout_centers)
if not readout_fit.success:
    raise RuntimeError(f"Readout Gaussian fit failed: {readout_fit.message}")
dark_mean = readout_fit.params["dark_center"].value
dark_sigma = readout_fit.params["dark_sigma"].value
bright_mean = readout_fit.params["bright_center"].value
bright_sigma = readout_fit.params["bright_sigma"].value
# Equal prior probabilities: minimize [P(dark above threshold) + P(bright below threshold)] / 2.
threshold_result = minimize_scalar(
    lambda threshold: (
        (
            norm.sf(threshold, loc=dark_mean, scale=dark_sigma)
            + norm.cdf(threshold, loc=bright_mean, scale=bright_sigma)
        )
        / 2
    ),
    bounds=(dark_mean, bright_mean),
    method="bounded",
)
if not threshold_result.success:
    raise RuntimeError(f"Readout threshold search failed: {threshold_result.message}")
readout_threshold = threshold_result.x

# %%
readout_fig, readout_axes = plt.subplots(1, 2, figsize=ION_READOUT_FIGSIZE)
ax = readout_axes[0]
for counts, label, color, edge in (
    (dark_counts, r"$\sigma^+$ pump (dark)", BAR_FILL, BAR_EDGE),
    (bright_counts, r"$\sigma^-$ pump (bright)", IONQ_ORANGE, IONQ_ORANGE_EDGE),
):
    ax.stairs(
        counts,
        readout_edges,
        fill=True,
        facecolor=to_rgba(color, ION_READOUT_FILL_ALPHA),
        edgecolor=edge,
        linewidth=0.6,
        label=label,
    )
readout_fit_x = np.linspace(readout_edges[0], readout_edges[-1], 1000)
# Convert fitted probability density to expected shots per bin: density * shots * bin width.
ax.plot(
    readout_fit_x,
    readout_fit.eval(x=readout_fit_x) * readout_shots * ION_READOUT_BIN_WIDTH,
    color="black",
    linewidth=ION_READOUT_FIT_LINEWIDTH,
    label="Bimodal Gaussian fit",
)
ax.axvline(
    readout_threshold,
    color="0.3",
    linestyle=":",
    linewidth=0.8,
    label=f"Threshold: {readout_threshold:.1f} PE",
)
ax.set_yscale("log")
ax.minorticks_off()
ax.set_xlim(readout_edges[0], readout_edges[-1])
ax.set_ylim(*ION_READOUT_YLIM)
ax.set_xlabel("Ion photoelectrons")
ax.set_ylabel("Occurrences")
ax.legend(loc="upper right", frameon=False)

lines = [
    "Supplementary ion readout histogram",
    f"Dark-preparation shots: {dark_readout.size}",
    f"Bright-preparation shots: {bright_readout.size}",
    f"Threshold: {readout_threshold:.8f} photoelectrons",
    "",
    format_fit_report(readout_fit),
    "",
    f"{'Bin left (PE)':>16} {'Bin right (PE)':>16} {'Dark shots':>12} {'Bright shots':>14} {'Fit shots':>16}",
]
for left, right, dark, bright, fitted in zip(
    readout_edges[:-1],
    readout_edges[1:],
    dark_counts,
    bright_counts,
    readout_fit.best_fit * readout_shots * ION_READOUT_BIN_WIDTH,
):
    lines.append(f"{left:16.8f} {right:16.8f} {dark:12d} {bright:14d} {fitted:16.8f}")
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_ion_readout_histogram.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Ion readout assignment matrix

# %%
assignment_counts = np.array(
    [
        [
            np.count_nonzero(values <= readout_threshold),
            np.count_nonzero(values > readout_threshold),
        ]
        for values in (dark_readout, bright_readout)
    ]
)
assignment_shots = assignment_counts.sum(axis=1, keepdims=True)
assignment_probability = assignment_counts / assignment_shots
# Keep error bars nonzero at p = 0 or 1 without changing the measured probability.
assignment_probability_for_error = (assignment_counts + 0.5) / (assignment_shots + 1)
assignment_error = np.sqrt(
    assignment_probability_for_error
    * (1 - assignment_probability_for_error)
    / assignment_shots
)
assert np.allclose(assignment_probability.sum(axis=1), 1)

ION_ASSIGNMENT_LABELS = ("Sigma plus (σ+)", "Sigma minus (σ−)")

ax = readout_axes[1]
row_colors = np.array([to_rgba(BAR_FILL, 0.4), to_rgba(IONQ_ORANGE, 0.4)])
ax.imshow(np.repeat(row_colors[:, None, :], 2, axis=1), interpolation="nearest")
ax.axhline(0.5, color="white", lw=1)
ax.axvline(0.5, color="white", lw=1)
ax.set(
    xticks=[0, 1],
    xticklabels=["Dark", "Bright"],
    yticks=[0, 1],
    yticklabels=ION_ASSIGNMENT_LABELS,
    xlabel="Detected state",
    title="Ion readout percentages",
)
ax.tick_params(top=False, right=False, length=0)
for row in range(2):
    for column in range(2):
        p = assignment_probability[row, column]
        text = format_uncertainty(100 * p, 100 * assignment_error[row, column])
        ax.text(column, row, f"{text} %", ha="center", va="center")
for label, panel in zip("ab", readout_axes):
    panel.text(
        -0.13,
        1.04,
        label,
        transform=panel.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=0.3, foreground="black")],
    )
save_figure(readout_fig, "supplementary_ion_readout")

lines = [
    "Supplementary ion readout assignment matrix",
    "",
    "pump_polarization detected_state count shots probability standard_error",
]
for row, polarization in enumerate((1, -1)):
    for column, state in enumerate(("dark", "bright")):
        lines.append(
            f"{polarization:+d} {state} {assignment_counts[row, column]} {assignment_shots[row, 0]} "
            f"{assignment_probability[row, column]:.8f} {assignment_error[row, column]:.8f}"
        )
(FIGURES_DIR / "txt" / "supplementary_ion_readout_assignment.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## Ion-photon correlations

# %%
ION_PHOTON_FIGSIZE = (
    SINGLE_COLUMN_WIDTH_MM * MM,
    SINGLE_COLUMN_WIDTH_MM * MM * 5 / 6.5,
)
ION_PHOTON_MARKER_SIZE = 3
ION_PHOTON_MARKER_EDGE_WIDTH = 0.6
ION_PHOTON_FIT_LINEWIDTH = 0.8
ION_PHOTON_YLIM = (0, 1)
ION_PHOTON_SWEEPS = (
    (
        737,
        "ZZ",
        ION_737_PHOTON_OFF_RESONANT_ZZ_SWEEP,
        90,
        "Basis converter output HWP angle (deg)",
    ),
    (
        737,
        "XX",
        ION_737_PHOTON_OFF_RESONANT_XX_SWEEP,
        180,
        "Ion Z-rotation angle (deg)",
    ),
    (493, "ZZ", ION_493_PHOTON_ZZ_SWEEP, 90, "Waveplate angle (deg)"),
    (493, "XX", ION_493_PHOTON_XX_SWEEP, 180, "Ion Z-rotation angle (deg)"),
    (
        493,
        "ZZ",
        ION_493_PHOTON_IMPROVED_EXTINCTION_ZZ_SWEEP,
        90,
        "Waveplate angle (deg)",
    ),
    (
        493,
        "XX",
        ION_493_PHOTON_IMPROVED_EXTINCTION_XX_SWEEP,
        180,
        "Ion Z-rotation angle (deg)",
    ),
)
ION_PHOTON_DETECTORS = (
    (1, "o", BAR_FILL, BAR_EDGE),
    (2, "o", IONQ_ORANGE, IONQ_ORANGE_EDGE),
)

# %%
ion_photon_results = {}
for wavelength_nm, basis, run_id, period, xlabel in ION_PHOTON_SWEEPS:
    ds = load_dataset(run_id, "main.h5")
    assert np.isin(ds.ion_state.values, [0, 1]).all(), (
        f"Run {run_id}: invalid or missing ion states"
    )
    assert np.isin(ds.which_spcm_clicked.values, [1, 2]).all(), (
        f"Run {run_id}: invalid or missing detector outcomes"
    )
    (sweep_dim,) = (dim for dim in ds.ion_state.dims if dim != "success_idx")
    sweep = ds[sweep_dim].values
    angle_scale = 2 if basis == "XX" else 1
    fit_sweep = np.linspace(sweep.min(), sweep.max(), 1000)
    fig, ax = plt.subplots(figsize=ION_PHOTON_FIGSIZE)
    detector_results = {}
    for detector, marker, facecolor, edgecolor in ION_PHOTON_DETECTORS:
        selected = ds.which_spcm_clicked == detector
        counts = selected.sum("success_idx").values
        successes = (selected & (ds.ion_state == 1)).sum("success_idx").values
        assert np.all(counts > 0), (
            f"Run {run_id}, detector {detector}: empty sweep setting"
        )
        probability = successes / counts
        # Keep error bars nonzero at p = 0 or 1 without changing the measured probability.
        probability_for_error = (successes + 0.5) / (counts + 1)
        probability_error = np.sqrt(
            probability_for_error * (1 - probability_for_error) / counts
        )

        sine_model = SineModel()
        model = sine_model + ConstantModel()
        params = sine_model.guess(probability, x=sweep)
        params.add("c", value=probability.mean())
        params["frequency"].set(value=2 * np.pi / period, vary=False)
        params["shift"].set(min=-np.inf, max=np.inf)

        fit = model.fit(
            probability,
            params,
            x=sweep,
            weights=1 / probability_error,
            scale_covar=False,
        )
        if not fit.success:
            raise RuntimeError(f"Run {run_id}, detector {detector}: {fit.message}")
        assert fit.params["amplitude"].stderr is not None, (
            "Sine-fit amplitude uncertainty is unavailable"
        )
        contrast = 2 * fit.params["amplitude"].value
        contrast_error = 2 * fit.params["amplitude"].stderr
        detector_results[detector] = {
            "counts": counts,
            "probability": probability,
            "error": probability_error,
            "fit": fit,
            "contrast": contrast,
            "contrast_error": contrast_error,
        }
        ax.errorbar(
            angle_scale * sweep,
            probability,
            yerr=probability_error,
            fmt="none",
            **ERRORBAR_STYLE,
            zorder=2,
        )
        ax.plot(
            angle_scale * sweep,
            probability,
            marker=marker,
            linestyle="none",
            label=f"Detector {detector}",
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markersize=ION_PHOTON_MARKER_SIZE,
            markeredgewidth=ION_PHOTON_MARKER_EDGE_WIDTH,
            zorder=3,
        )
        fitted_probability = fit.eval(x=fit_sweep)
        ax.plot(
            angle_scale * fit_sweep,
            fitted_probability,
            color=edgecolor,
            linewidth=ION_PHOTON_FIT_LINEWIDTH,
        )
        peak_angle = angle_scale * fit_sweep[np.argmax(fitted_probability)]
        ax.text(
            peak_angle,
            0.73,
            f"C = {format_uncertainty(100 * contrast, 100 * contrast_error)} %",
            ha="center",
            color=edgecolor,
            fontsize=SMALL_FONT_SIZE,
        )
    ion_photon_results[run_id] = detector_results
    ax.set_xlabel(xlabel)
    if basis == "XX":
        ax.set_xticks(np.arange(0, 361, 90))
    ax.set_ylabel("Ion-state probability P(1|Detector)")
    ax.set_ylim(*ION_PHOTON_YLIM)
    ax.set_yticks(np.linspace(0, 1, 5))
    title = f"Ion-{wavelength_nm} nm photon {basis}"
    if wavelength_nm == 737:
        title += ", off-resonant from the SiV"
    improved_extinction = run_id in (
        ION_493_PHOTON_IMPROVED_EXTINCTION_ZZ_SWEEP,
        ION_493_PHOTON_IMPROVED_EXTINCTION_XX_SWEEP,
    )
    if improved_extinction:
        title += ", improved extinction"
    ax.set_title(title)
    legend_location = "upper left" if wavelength_nm == 493 else "upper right"
    if improved_extinction:
        legend_location = "center" if basis == "ZZ" else "upper right"
    ax.legend(loc=legend_location, frameon=False)
    output_name = f"supplementary_ion_{wavelength_nm}_photon_{basis.lower()}_sweep"
    if improved_extinction:
        output_name += "_improved_extinction"
    save_figure(fig, output_name)

    lines = [
        f"Supplementary {title}",
    ]
    for detector, result in detector_results.items():
        lines.extend(
            [
                "",
                f"Detector {detector}",
                f"Contrast = {format_uncertainty(100 * result['contrast'], 100 * result['contrast_error'])} %",
                "",
                format_fit_report(result["fit"]),
                "",
                f"{'Recorded angle (deg)':>20} {'Plot angle (deg)':>18} "
                f"{'Clicks':>10} {'P(ion=1)':>14} {'Standard error':>16}",
            ]
        )
        for angle, count, probability, error in zip(
            sweep, result["counts"], result["probability"], result["error"]
        ):
            lines.append(
                f"{angle:20.8f} {angle_scale * angle:18.8f} {count:10d} {probability:14.8f} {error:16.8f}"
            )
    (FIGURES_DIR / "txt" / f"{output_name}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

# %% [markdown]
# ## Supplementary Ion-493 nm photon XX compensation comparison

# %%
photon_compensation_scans = {
    "No compensation": load_dataset(ION_493_PHOTON_XX_PHASE_COMP_OFF, "main.h5"),
    "Compensation": load_dataset(ION_493_PHOTON_XX_PHASE_COMP_ON, "main.h5"),
}

# %%
PHOTON_COMP_SPCM_TICK_S = 0.5e-9
PHOTON_COMP_ARRIVAL_BIN_TICKS = 8
PHOTON_COMP_ARRIVAL_BIN_WIDTH_S = (
    PHOTON_COMP_ARRIVAL_BIN_TICKS * PHOTON_COMP_SPCM_TICK_S
)

spcm_ticks = np.concatenate(
    [ds.spcm_timestamp.values.ravel() for ds in photon_compensation_scans.values()]
)
assert np.isfinite(spcm_ticks).all() and np.all(spcm_ticks >= 0)
assert np.all(spcm_ticks == np.floor(spcm_ticks))
first_bin = int(spcm_ticks.min() // PHOTON_COMP_ARRIVAL_BIN_TICKS)
last_bin = int(spcm_ticks.max() // PHOTON_COMP_ARRIVAL_BIN_TICKS)
arrival_edges = np.arange(first_bin, last_bin + 2) * PHOTON_COMP_ARRIVAL_BIN_WIDTH_S
arrival_times = (
    np.arange(first_bin, last_bin + 1) + 0.5
) * PHOTON_COMP_ARRIVAL_BIN_WIDTH_S
bin_axis = xr.DataArray(
    np.arange(first_bin, last_bin + 1),
    dims="arrival_time",
    coords={"arrival_time": arrival_times},
)
photon_compensation_results = {}
for label, ds in photon_compensation_scans.items():
    assert np.isin(ds.ion_state.values, [0, 1]).all()
    assert np.isin(ds.which_spcm_clicked.values, [1, 2]).all()
    assert np.isfinite(ds.laser_1762_phase.values).all()
    state = make_two_qubit_state(
        state_1=ds.ion_state,
        state_1_high_value=1,
        state_2=ds.which_spcm_clicked,
        state_2_high_value=2,
        name="ion_photon",
        label="Ion-493 nm photon",
    )
    binned_state = state.where(
        ds.spcm_timestamp // PHOTON_COMP_ARRIVAL_BIN_TICKS == bin_axis
    )
    histogram, _, correlator = calculate_histogram_population_and_correlator(
        binned_state,
        label="Ion-493 nm photon",
        prefix="ion_photon",
        rep_dim="success_idx",
    )
    assert histogram.sum().item() == ds.ion_state.size
    photon_compensation_results[label] = (histogram, correlator)

# %%
PHOTON_COMP_FIGSIZE = (SINGLE_COLUMN_WIDTH_MM * MM, 123 * MM)
PHOTON_COMP_HIST_HEIGHT_RATIO = 0.30
PHOTON_COMP_HEATMAP_HEIGHT_RATIO = 0.70
PHOTON_COMP_HIST_BIN_TICKS = 2
PHOTON_COMP_CMAP = "RdBu"
PHOTON_COMP_MIN_EVENTS = 10
PHOTON_COMP_PULSE_DURATION_S = 1 / 79.3e6
PHOTON_COMP_DECAY_TIME_S = 7.9e-9
PHOTON_COMP_MODEL_DELAY_S = 3.5e-9

# %%
photon_edge_ticks = np.arange(
    first_bin * PHOTON_COMP_ARRIVAL_BIN_TICKS,
    (last_bin + 1) * PHOTON_COMP_ARRIVAL_BIN_TICKS + PHOTON_COMP_HIST_BIN_TICKS,
    PHOTON_COMP_HIST_BIN_TICKS,
)
photon_arrival_counts, _ = np.histogram(
    spcm_ticks,
    bins=photon_edge_ticks,
)
photon_edges = photon_edge_ticks * PHOTON_COMP_SPCM_TICK_S
photon_centers = (photon_edges[:-1] + photon_edges[1:]) / 2
photon_arrival_normalized = photon_arrival_counts / photon_arrival_counts.max()
# Model: uniform pulse convolved with exponential decay,
# shifted by a fixed delay and normalized to its analytic peak.
photon_model_time_s = np.unique(
    np.r_[
        np.linspace(0, photon_edges[-1], 1000),
        PHOTON_COMP_MODEL_DELAY_S,
        PHOTON_COMP_MODEL_DELAY_S + PHOTON_COMP_PULSE_DURATION_S,
    ]
)
photon_model_elapsed_s = np.maximum(photon_model_time_s - PHOTON_COMP_MODEL_DELAY_S, 0)
photon_model = (
    -np.expm1(
        -np.minimum(photon_model_elapsed_s, PHOTON_COMP_PULSE_DURATION_S)
        / PHOTON_COMP_DECAY_TIME_S
    )
    * np.exp(
        -np.maximum(photon_model_elapsed_s - PHOTON_COMP_PULSE_DURATION_S, 0)
        / PHOTON_COMP_DECAY_TIME_S
    )
    / -np.expm1(-PHOTON_COMP_PULSE_DURATION_S / PHOTON_COMP_DECAY_TIME_S)
)
fig = plt.figure(figsize=PHOTON_COMP_FIGSIZE, layout="constrained")
grid = fig.add_gridspec(
    3,
    2,
    height_ratios=(
        PHOTON_COMP_HIST_HEIGHT_RATIO,
        PHOTON_COMP_HEATMAP_HEIGHT_RATIO,
        PHOTON_COMP_HEATMAP_HEIGHT_RATIO,
    ),
    width_ratios=(1, 0.045),
)
photon_hist_ax = fig.add_subplot(grid[0, 0])
heatmap_axes = [fig.add_subplot(grid[row, 0], sharex=photon_hist_ax) for row in (1, 2)]
colorbar_ax = fig.add_subplot(grid[1:, 1])
photon_hist_ax.tick_params(axis="x", labelbottom=False)
photon_hist_ax.stairs(
    photon_arrival_normalized, photon_edges * 1e9, fill=True, label="Data", **BAR_STYLE
)
photon_hist_ax.plot(
    photon_model_time_s * 1e9,
    photon_model,
    color=IONQ_ORANGE,
    linewidth=1.0,
    label="Model",
)
photon_hist_ax.set_ylim(0, 1.1)
photon_hist_ax.set_yticks([0, 0.5, 1])
photon_hist_ax.set_ylabel("Occurrences (norm.)")
photon_hist_ax.legend(loc="upper right", frameon=False)
photon_hist_ax.set_title("Photon shape")
photon_comp_cmap = plt.get_cmap(PHOTON_COMP_CMAP).copy()
photon_comp_cmap.set_bad("white")
for ax, (label, (histogram, correlator)) in zip(
    heatmap_axes, photon_compensation_results.items()
):
    display_mask = histogram.sum("state_idx") >= PHOTON_COMP_MIN_EVENTS
    values = (
        correlator["ion_photon_corr_mean"]
        .where(display_mask)
        .assign_coords(
            arrival_time=arrival_times * 1e9,
            laser_1762_phase=2 * correlator.laser_1762_phase,
        )
    )
    image = values.plot.pcolormesh(
        ax=ax,
        x="arrival_time",
        y="laser_1762_phase",
        add_colorbar=False,
        cmap=photon_comp_cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(
        xlabel="",
        ylabel="Ion Z-rotation angle (deg)",
        yticks=np.arange(0, 361, 90),
        title=label,
    )
heatmap_axes[0].tick_params(axis="x", labelbottom=False)
heatmap_axes[1].set_xlabel("Photon arrival time (ns)")
photon_hist_ax.set_xlim(arrival_edges[0] * 1e9, arrival_edges[-1] * 1e9)
fig.colorbar(image, cax=colorbar_ax, label="Ion-493 nm photon XX correlator")
save_figure(fig, "supplementary_ion_493_photon_xx_phase_compensation")

lines = [
    "Supplementary Ion-493 nm photon XX phase compensation comparison",
]
for label, (histogram, correlator) in photon_compensation_results.items():
    lines.extend(
        [
            "",
            label,
            "AOM_phase_deg ion_rotation_deg arrival_ns left_ns right_ns shots mean lower_error upper_error displayed",
        ]
    )
    for phase in correlator.laser_1762_phase.values:
        for i, arrival in enumerate(arrival_times):
            values = correlator.sel(laser_1762_phase=phase, arrival_time=arrival)
            shots = (
                histogram.sel(laser_1762_phase=phase, arrival_time=arrival).sum().item()
            )
            lines.append(
                f"{phase:.8f} {2 * phase:.8f} {arrival * 1e9:.8f} "
                f"{arrival_edges[i] * 1e9:.8f} {arrival_edges[i + 1] * 1e9:.8f} {shots:d} "
                f"{values['ion_photon_corr_mean'].item():.8f} "
                f"{values['ion_photon_corr_minus_error'].item():.8f} "
                f"{values['ion_photon_corr_plus_error'].item():.8f} {int(shots >= PHOTON_COMP_MIN_EVENTS)}"
            )
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
lines.extend(
    [
        "",
        "Pooled photon arrival histogram (1 ns bins, peak-normalized)",
        f"Total occurrences: {photon_arrival_counts.sum()}",
        "",
        f"{'Arrival time (ns)':>18} {'Bin left (ns)':>15} {'Bin right (ns)':>15} "
        f"{'Occurrences':>12} {'Normalized':>14}",
    ]
)
for center, left, right, count, normalized in zip(
    photon_centers,
    photon_edges[:-1],
    photon_edges[1:],
    photon_arrival_counts,
    photon_arrival_normalized,
):
    lines.append(
        f"{center * 1e9:18.8f} {left * 1e9:15.8f} {right * 1e9:15.8f} {count:12d} {normalized:14.8f}"
    )
(
    FIGURES_DIR / "txt" / "supplementary_ion_493_photon_xx_phase_compensation.txt"
).write_text("\n".join(lines) + "\n", encoding="utf-8")
