"""Plot the SiV reflection, readout, spin control, photon correlations, and reset.

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
# # Supplementary SiV reflection and readout

# %%
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import ConstantModel, ExponentialModel, SineModel
from matplotlib import patheffects

from papertools import (
    BAR_EDGE,
    BAR_FILL,
    DOUBLE_COLUMN_WIDTH_MM,
    ERRORBAR_STYLE,
    FIGURES_DIR,
    IONQ_ORANGE,
    IONQ_ORANGE_EDGE,
    MM,
    apply_publication_style,
    load_dataset,
    load_provenance,
    save_figure,
)
from papertools.formatting import format_fit_report, format_uncertainty
from papertools.run_ids import (
    ION_SIV_FULL_TOMOGRAPHY_FIGURE_1,
    SIV_CAVITY_REFLECTION,
    SIV_ELECTRON_RABI,
    SIV_LINEWIDTH_REFLECTION,
    SIV_MID_CIRCUIT_RESET,
    SIV_NO_RESET,
    SIV_ODMR_NUCLEAR_DOWN,
    SIV_ODMR_NUCLEAR_UP,
    SIV_PHOTON_HV_POPULATIONS,
    SIV_PHOTON_ZZ_WAVEPLATE_SWEEP,
    SIV_REFLECTION_BLUE,
    SIV_REFLECTION_ORANGE,
)

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Cavity-SiV reflection

# %%
SIV_FREQUENCY_OFFSET_HZ = 406e12

# There is a second SiV in the scan, that we exclude from plotting
# because it's not used in this work and does not affect the fitting.
SIV_BROAD_RANGE_GHZ = (430, 575)

# The dark count rates stored in the datasets are not correct.
# These have been measured right before the dataset acquisition and
# are used to correct the data.
SIV_SIGNAL_DARK_RATE_HZ = 170
SIV_REFERENCE_DARK_RATE_HZ = 1800


def siv_reflection(frequency, amplitude, slope, eta, kappa, f_cav, f_siv, g, gamma):
    """Return the intensity reflection of one SiV coupled to a cavity.

    All frequencies and rates are ordinary frequencies (angular frequency / 2pi).

    Args:
        frequency: Laser frequency relative to a fixed offset, in GHz.
        amplitude: Off-resonant reflection level, in the units of the data.
        slope: Linear baseline slope, in 1/GHz.
        eta: Cavity input coupling ratio kappa_in / kappa, dimensionless.
        kappa: Total cavity energy decay rate, in GHz.
        f_cav: Cavity resonance frequency relative to the same offset, in GHz.
        f_siv: SiV optical transition frequency relative to the same offset, in GHz.
        g: SiV-cavity coupling rate, in GHz.
        gamma: SiV optical transition linewidth, in GHz.

    Returns:
        Reflected intensity, in the units of ``amplitude``.
    """
    denominator = (
        1j * (frequency - f_cav)
        + kappa / 2
        + g**2 / (1j * (frequency - f_siv) + gamma / 2)
    )
    baseline = amplitude * (1 + slope * frequency)
    return baseline * abs(1 - eta * kappa / denominator) ** 2


broad_ds = load_dataset(SIV_CAVITY_REFLECTION, "dataset_0.h5").isel(rep=0)
broad_ds = broad_ds.sel(
    laser_frequency=slice(
        SIV_FREQUENCY_OFFSET_HZ + SIV_BROAD_RANGE_GHZ[0] * 1e9,
        SIV_FREQUENCY_OFFSET_HZ + SIV_BROAD_RANGE_GHZ[1] * 1e9,
    )
)
broad_frequency = (broad_ds.laser_frequency.values - SIV_FREQUENCY_OFFSET_HZ) / 1e9
assert np.isfinite(broad_frequency).all()
assert all(
    np.isfinite(broad_ds[name].values).all()
    for name in ("rate", "norm_rate", "corr_rate")
)
ignored = broad_ds.corr_rate.values == 0
fit_mask = ~ignored
broad_metadata = load_provenance(SIV_CAVITY_REFLECTION)["metadata"]
signal_correction = (
    broad_metadata["spcm_dark_count_rate_device"] - SIV_SIGNAL_DARK_RATE_HZ
)
reference_correction = (
    broad_metadata["spcm_dark_count_rate_normalization"] - SIV_REFERENCE_DARK_RATE_HZ
)
signal_rate = broad_ds.rate.values + signal_correction
reference_rate = broad_ds.norm_rate.values + reference_correction
initial_reference = (
    broad_metadata["initial_normalization_countrate"] + reference_correction
)
assert np.any(fit_mask) and initial_reference > 0
assert np.all(signal_rate[fit_mask] > 0) and np.all(reference_rate[fit_mask] > 0)

broad_rate = np.zeros(broad_frequency.size)
broad_rate[fit_mask] = (
    signal_rate[fit_mask] * initial_reference / reference_rate[fit_mask]
)
broad_scale = broad_rate.max()
broad_reflection = broad_rate / broad_scale

model = Model(siv_reflection)
linewidth_ds = load_dataset(SIV_LINEWIDTH_REFLECTION, "dataset_0.h5").isel(rep=0)
linewidth_frequency = (
    linewidth_ds.laser_frequency.values - SIV_FREQUENCY_OFFSET_HZ
) / 1e9
linewidth_rate = linewidth_ds.corr_rate.values
assert np.isfinite(linewidth_frequency).all() and np.isfinite(linewidth_rate).all()
assert np.all(linewidth_rate >= 0)
linewidth_mask = linewidth_rate > 0
linewidth_scale = linewidth_rate.max()
linewidth_reflection = linewidth_rate / linewidth_scale
linewidth_parameters = model.make_params(
    amplitude=1,
    slope=0,
    eta=0.5,
    kappa=30,
    f_cav=470,
    f_siv=525.4,
    g=4.4,
    gamma=0.1,
)
linewidth_parameters["amplitude"].set(min=0)
linewidth_parameters["slope"].set(min=-0.001, max=0.001)
linewidth_parameters["eta"].set(min=0, max=1)
linewidth_parameters["kappa"].set(min=1, max=100)
linewidth_parameters["f_cav"].set(min=450, max=490)
linewidth_parameters["f_siv"].set(min=523, max=528)
linewidth_parameters["g"].set(min=0.1, max=10)
linewidth_parameters["gamma"].set(min=0.001, max=1)
linewidth_fit = model.fit(
    linewidth_reflection[linewidth_mask],
    linewidth_parameters,
    frequency=linewidth_frequency[linewidth_mask],
    weights=1 / np.sqrt(linewidth_reflection[linewidth_mask]),
)
assert linewidth_fit.success, linewidth_fit.message
assert linewidth_fit.covar is not None, "Linewidth fit covariance is unavailable"
gamma_calibration = linewidth_fit.uvars["gamma"]

parameters = model.make_params(
    amplitude=1,
    slope=0,
    eta=0.5,
    kappa=40,
    f_cav=505,
    f_siv=525,
    g=3,
    gamma=0.1,
)
parameters["amplitude"].set(min=0)
parameters["slope"].set(min=-0.005, max=0.005)
parameters["eta"].set(min=0, max=1)
parameters["kappa"].set(min=1, max=100)
parameters["f_cav"].set(min=490, max=515)
parameters["f_siv"].set(min=522, max=528)
parameters["g"].set(min=0.1, max=10)
parameters["gamma"].set(value=gamma_calibration.nominal_value, vary=False)
broad_fit = model.fit(
    broad_reflection[fit_mask],
    parameters,
    frequency=broad_frequency[fit_mask],
    weights=1 / np.sqrt(broad_reflection[fit_mask]),
)
assert broad_fit.success, broad_fit.message

# Refit the broad scan with gamma sampled from its fitted Gaussian uncertainty.
# Add the variance across refits to the average squared fit error to include both uncertainty sources.

SIV_GAMMA_MC_SAMPLES = 1000
SIV_GAMMA_MC_SEED = 0

generator = np.random.default_rng(SIV_GAMMA_MC_SEED)
assert gamma_calibration.nominal_value > gamma_calibration.std_dev > 0
sampled_gamma = generator.normal(
    gamma_calibration.nominal_value, gamma_calibration.std_dev, SIV_GAMMA_MC_SAMPLES
)
assert np.all(sampled_gamma > 0)
cqed_samples = []
cqed_conditional_variances = []
for gamma in sampled_gamma:
    sample_parameters = broad_fit.params.copy()
    sample_parameters["gamma"].set(value=gamma)
    sample_fit = model.fit(
        broad_reflection[fit_mask],
        sample_parameters,
        frequency=broad_frequency[fit_mask],
        weights=1 / np.sqrt(broad_reflection[fit_mask]),
    )
    assert sample_fit.success, sample_fit.message
    assert sample_fit.covar is not None, "Monte Carlo refit covariance is unavailable"
    p = sample_fit.uvars
    quantities = (
        p["eta"] * p["kappa"],
        p["kappa"],
        SIV_FREQUENCY_OFFSET_HZ / 1e12 + p["f_cav"] / 1000,
        SIV_FREQUENCY_OFFSET_HZ / 1e12 + p["f_siv"] / 1000,
        p["gamma"] * 1000,
        p["g"],
        4 * p["g"] ** 2 / (p["kappa"] * p["gamma"]),
    )
    cqed_samples.append([q.nominal_value for q in quantities])
    cqed_conditional_variances.append([q.std_dev**2 for q in quantities])
cqed_gamma_variance = np.var(cqed_samples, axis=0, ddof=1)
cqed_scan_variance = np.mean(cqed_conditional_variances, axis=0)
cqed_errors = np.sqrt(cqed_gamma_variance + cqed_scan_variance)
# Quote the independent gamma estimate directly, rather than its sampled spread.
cqed_errors[4] = gamma_calibration.std_dev * 1000
assert np.isfinite(cqed_errors).all()

# %% [markdown]
# ## Narrow reflection scans

# %%
narrow_scans = []
for run_id, initial_frequency in (
    (SIV_REFLECTION_BLUE, 525.7),
    (SIV_REFLECTION_ORANGE, 526.5),
):
    ds = load_dataset(run_id, "dataset_0.h5")
    assert np.isfinite(ds.corr_rate.values).all()
    frequency = (ds.laser_frequency.values - SIV_FREQUENCY_OFFSET_HZ) / 1e9
    rate = ds.corr_rate.median("rep", skipna=False).values
    assert np.isfinite(frequency).all() and np.all(rate >= 0)
    parameters = broad_fit.params.copy()
    for parameter in parameters.values():
        parameter.set(vary=False)
    parameters["f_siv"].set(
        value=initial_frequency, min=frequency.min(), max=frequency.max(), vary=True
    )
    parameters["amplitude"].set(value=rate.max(), vary=True)
    fit = model.fit(rate, parameters, frequency=frequency)
    assert fit.success, fit.message
    narrow_scans.append((run_id, frequency, rate, fit))

# %% [markdown]
# ## SiV readout distribution

# %%
readout = load_dataset(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "main.h5")
counts = readout.siv_final_readout_counts.values.ravel()
assert (
    np.isfinite(counts).all()
    and np.all(counts >= 0)
    and np.all(counts == np.floor(counts))
)
readout_metadata = load_provenance(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1)["metadata"]
threshold = readout_metadata["dr2.siv.readout.hi_trsh"]
readout_duration_us = readout_metadata["dr2.siv.readout.duration"] * 1e6
assert threshold == int(threshold)
photon_number = np.arange(int(counts.max()) + 1)
probability = np.bincount(counts.astype(int)) / counts.size
assert np.isclose(probability.sum(), 1)

# %%
SIV_FIGSIZE = (DOUBLE_COLUMN_WIDTH_MM * MM, 0.78 * DOUBLE_COLUMN_WIDTH_MM * MM)
SIV_FIT_POINTS = 4000
SIV_FIT_LINEWIDTH = 0.8
SIV_BROAD_MARKER_SIZE = 1.6
SIV_NARROW_COLORS = ((BAR_FILL, BAR_EDGE), (IONQ_ORANGE, IONQ_ORANGE_EDGE))
SIV_LINEWIDTH_INSET_POSITION = (0.04, 0.16, 0.33, 0.48)

fig, axes = plt.subplot_mosaic([["a", "a"], ["b", "c"]], figsize=SIV_FIGSIZE)
ax = axes["a"]
broad_grid = np.linspace(*SIV_BROAD_RANGE_GHZ, SIV_FIT_POINTS)
ax.plot(
    broad_frequency[fit_mask],
    broad_reflection[fit_mask],
    ".",
    color=BAR_FILL,
    markersize=SIV_BROAD_MARKER_SIZE,
    markeredgewidth=0,
    alpha=0.7,
    label="Data",
)
ax.plot(
    broad_grid,
    broad_fit.eval(frequency=broad_grid),
    color=BAR_EDGE,
    lw=SIV_FIT_LINEWIDTH,
    label="Fit",
)
ax.plot(
    broad_frequency[ignored],
    broad_reflection[ignored],
    "x",
    color="0.7",
    markeredgecolor="0.7",
    markersize=2,
    markeredgewidth=0.4,
    label="Ignored (measurement issue)",
)
ax.legend(
    loc="lower right", frameon=False, fontsize=7, borderaxespad=1.2, handlelength=1.5
)
ax.set(
    xlim=SIV_BROAD_RANGE_GHZ,
    ylim=(-0.05, 1.03),
    ylabel="Background-corrected and normalized reflection",
    xlabel="Laser frequency (GHz) + 406 THz",
)
inset = ax.inset_axes(SIV_LINEWIDTH_INSET_POSITION)
linewidth_grid = np.unique(
    np.r_[
        np.linspace(
            linewidth_frequency.min(), linewidth_frequency.max(), SIV_FIT_POINTS
        ),
        np.linspace(523, 528, SIV_FIT_POINTS),
    ]
)
inset.plot(
    linewidth_frequency[linewidth_mask],
    linewidth_reflection[linewidth_mask],
    ".",
    color=BAR_FILL,
    markersize=1.6,
    markeredgewidth=0,
)
inset.plot(
    linewidth_grid, linewidth_fit.eval(frequency=linewidth_grid), color=BAR_EDGE, lw=0.7
)
inset.plot(
    linewidth_frequency[~linewidth_mask],
    linewidth_reflection[~linewidth_mask],
    "x",
    color="0.7",
    markeredgecolor="0.7",
    markersize=2,
    markeredgewidth=0.4,
)
inset.set(
    xlim=(linewidth_frequency.min(), linewidth_frequency.max()),
    ylim=(-0.05, 1.05),
    xticks=[400, 450, 500, 550],
)
inset.tick_params(labelsize=5, length=2, pad=1)

ax = axes["b"]
for (run_id, frequency, rate, fit), (fill, edge) in zip(
    narrow_scans, SIV_NARROW_COLORS
):
    grid = np.linspace(frequency.min(), frequency.max(), SIV_FIT_POINTS)
    fitted = fit.eval(frequency=grid) / fit.params["amplitude"].value
    ax.plot(
        frequency,
        rate / fit.params["amplitude"].value,
        "o",
        color=fill,
        markeredgecolor=edge,
        markeredgewidth=0.3,
        markersize=2,
        label="Spin down data" if run_id == SIV_REFLECTION_BLUE else "Spin up data",
    )
    ax.plot(
        grid,
        fitted,
        color=edge,
        lw=SIV_FIT_LINEWIDTH,
        label="Spin down fit" if run_id == SIV_REFLECTION_BLUE else "Spin up fit",
    )
    if run_id == SIV_REFLECTION_BLUE:
        laser_frequency = grid[np.argmin(fitted)]
        ax.axvline(
            laser_frequency,
            color=IONQ_ORANGE_EDGE,
            lw=0.6,
            linestyle="--",
            label="Optimal laser frequency",
        )
ax.set(
    ylim=(-0.05, 1.03),
    ylabel="Background-corrected and normalized reflection",
    xlabel="Laser frequency (GHz) + 406 THz",
)
ax.legend(
    loc="upper right", frameon=False, fontsize=7, borderaxespad=1.2, handlelength=1.5
)

ax = axes["c"]
for mask, color, label in (
    (photon_number < threshold, BAR_FILL, "Assigned spin down state"),
    (photon_number >= threshold, IONQ_ORANGE, "Assigned spin up state"),
):
    ax.bar(
        photon_number[mask],
        probability[mask],
        width=1,
        linewidth=0,
        color=color,
        label=label,
    )
ax.axvline(
    threshold - 0.5,
    color=IONQ_ORANGE_EDGE,
    lw=0.6,
    linestyle="--",
    label="Optimal threshold",
)
ax.legend(
    loc="upper right", frameon=False, fontsize=7, borderaxespad=1.2, handlelength=1.5
)
ax.set(
    xlabel=f"Number of photons collected in a {readout_duration_us:g} µs readout",
    ylabel="Probability",
    xlim=(-1, photon_number[-1] + 1),
)
for label, ax in axes.items():
    ax.text(
        -0.13 if label != "a" else -0.06,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=0.3, foreground="black")],
    )
save_figure(fig, "supplementary_siv_reflection_readout")

# %%
lines = [
    "Supplementary SiV reflection and readout",
    "Frequency unit: GHz above 406 THz",
    "",
    f"Normalization: {broad_scale:.8f} counts/s",
    format_fit_report(broad_fit),
    "",
    "Broad scan: frequency_GHz reflection fitted_reflection included_in_fit",
]
lines.extend(
    f"{x:.8f} {y:.8f} {f:.8f} {int(included)}"
    for x, y, f, included in zip(
        broad_frequency,
        broad_reflection,
        broad_fit.eval(frequency=broad_frequency),
        fit_mask,
    )
)
for run_id, frequency, rate, fit in narrow_scans:
    lines.extend(
        [
            "",
            format_fit_report(fit),
            "frequency_GHz reflection fitted_reflection",
        ]
    )
    scale = fit.params["amplitude"].value
    lines.extend(
        f"{x:.8f} {y / scale:.8f} {f / scale:.8f}"
        for x, y, f in zip(frequency, rate, fit.best_fit)
    )
lines.extend(
    [
        "",
        f"Shots: {counts.size}; high threshold: {int(threshold)} photons",
        f"Readout duration: {readout_duration_us:g} µs",
        "photon_number probability",
    ]
)
lines.extend(f"{n:d} {p:.8f}" for n, p in zip(photon_number, probability))
assert broad_fit.covar is not None, "Cavity-QED parameter covariance is unavailable"
# Central values are from the nominal fit, not the Monte Carlo average.
p = broad_fit.params.valuesdict()
cqed_values = (
    p["eta"] * p["kappa"],
    p["kappa"],
    SIV_FREQUENCY_OFFSET_HZ / 1e12 + p["f_cav"] / 1000,
    SIV_FREQUENCY_OFFSET_HZ / 1e12 + p["f_siv"] / 1000,
    p["gamma"] * 1000,
    p["g"],
    4 * p["g"] ** 2 / (p["kappa"] * p["gamma"]),
)
cqed_rows = (
    (r"$\kappa_\mathrm{in}/2\pi$", "GHz"),
    (r"$\kappa/2\pi$", "GHz"),
    (r"$\omega_\mathrm{cav}/2\pi$", "THz"),
    (r"$\omega_\mathrm{SiV}/2\pi$", "THz"),
    (r"$\gamma/2\pi$", "MHz"),
    (r"$g/2\pi$", "GHz"),
    (r"Cooperativity $C = 4g^2/(\kappa\gamma)$", ""),
)
lines.extend(
    [
        "",
        "LaTeX cavity-QED parameter table",
        r"\begin{table}[tb]",
        r"  \caption{Fitted cavity QED parameters.}",
        r"  \label{tab:SI:siv_cqed}",
        r"  \centering",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    Parameter & Fitted value \\",
        r"    \midrule",
    ]
)
for (label, unit), central_value, error in zip(cqed_rows, cqed_values, cqed_errors):
    value = format_uncertainty(central_value, error)
    if unit == "THz":
        number, uncertainty = value.split("(", 1)
        integer, fraction = number.split(".")
        value = (
            integer
            + "."
            + r"\,".join(fraction[i : i + 3] for i in range(0, len(fraction), 3))
            + "("
            + uncertainty
        )
    lines.append(f"    {label} & {value}" + (f"~{unit}" if unit else "") + r" \\")
lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
lines.extend(
    [
        "",
        f"Normalization: {linewidth_scale:.12g} counts/s",
        format_fit_report(linewidth_fit),
        "frequency_GHz reflection fitted_reflection included_in_fit",
    ]
)
lines.extend(
    f"{x:.8f} {y:.8f} {f:.8f} {int(included)}"
    for x, y, f, included in zip(
        linewidth_frequency,
        linewidth_reflection,
        linewidth_fit.eval(frequency=linewidth_frequency),
        linewidth_mask,
    )
)
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_siv_reflection_readout.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## Electron-spin ODMR and Rabi oscillations


# %%
def odmr_probability(x, center, omega, duration, amplitude, offset):
    """Return the spin transition probability after a rectangular microwave pulse.

    Args:
        x: Microwave frequency relative to a fixed offset, in MHz.
        center: Spin transition frequency relative to the same offset, in MHz.
        omega: Rabi frequency (angular Rabi frequency / 2pi), in MHz.
        duration: Pulse duration, in µs.
        amplitude: Peak transition probability above ``offset``, dimensionless.
        offset: Background transition probability, dimensionless.

    Returns:
        Transition probability, dimensionless.
    """
    effective_frequency = np.sqrt(omega**2 + (x - center) ** 2)
    return (
        offset
        + amplitude
        * (omega / effective_frequency) ** 2
        * np.sin(np.pi * duration * effective_frequency) ** 2
    )


spin_scans = []
for run_id, coordinate in (
    (SIV_ODMR_NUCLEAR_UP, "frequency"),
    (SIV_ODMR_NUCLEAR_DOWN, "frequency"),
    (SIV_ELECTRON_RABI, "duration"),
):
    ds = load_dataset(run_id, "dataset_0.h5")
    for name in ("first_readout_state", "second_readout_state"):
        assert np.isin(ds[name].values, [0, 1]).all(), (
            f"Run {run_id}: invalid readout states"
        )
    transition = ds.second_readout_state != ds.first_readout_state
    shots = ds.sizes["rep"]
    nof_transitions = transition.sum("rep").values
    probability = nof_transitions / shots
    # Keep error bars nonzero at p = 0 or 1 without changing the measured probability.
    probability_for_error = (nof_transitions + 0.5) / (shots + 1)
    error = np.sqrt(probability_for_error * (1 - probability_for_error) / shots)
    x = ds[coordinate].values
    assert np.isfinite(x).all() and np.all(np.diff(x) > 0)
    metadata = load_provenance(run_id)["metadata"]
    if coordinate == "frequency":
        # A fixed frequency origin avoids fitting with large absolute values.
        x = (x - 17.1e9) / 1e6
        duration_us = metadata["mw_pulse_duration"] * 1e6
        spin_model = Model(odmr_probability)
        params = spin_model.make_params(
            center=x[np.argmax(probability)],
            omega=1 / (2 * duration_us),
            duration=duration_us,
            amplitude=0.95,
            offset=0.001,
        )
        params["center"].set(min=x.min(), max=x.max())
        params["omega"].set(min=0.01, max=20)
        params["duration"].set(vary=False)
        params["offset"].set(min=0, max=1)
        params.add("contrast", value=0.95, min=0, max=1)
        params["amplitude"].set(expr="(1-offset)*contrast")
    else:
        x = x * 1e9
        spin_model = SineModel() + ConstantModel()
        params = spin_model.make_params(
            amplitude=0.49, frequency=2 * np.pi / 37.5, shift=-np.pi / 2, c=0.5
        )
        params["frequency"].set(min=2 * np.pi / 60, max=2 * np.pi / 20)
        params["shift"].set(min=-2 * np.pi, max=2 * np.pi)
        params.add("minimum", value=0.002, min=0, max=1)
        params.add("contrast", value=0.98, min=0, max=1)
        params["amplitude"].set(expr="contrast*(1-minimum)/2")
        params["c"].set(expr="minimum+amplitude")
    fit = spin_model.fit(probability, params, x=x, weights=1 / error, scale_covar=False)
    assert fit.success, fit.message
    spin_scans.append((run_id, x, nof_transitions, shots, probability, error, fit))

# %%
SIV_SPIN_FIGSIZE = (DOUBLE_COLUMN_WIDTH_MM * MM, 68 * MM)
SIV_SPIN_MARKER_SIZE = 2
SIV_SPIN_FIT_POINTS = 4000
SIV_SPIN_LABELS = ("Nuclear spin up", "Nuclear spin down", "Data")
SIV_SPIN_COLORS = (
    (BAR_FILL, BAR_EDGE),
    (IONQ_ORANGE, IONQ_ORANGE_EDGE),
    (BAR_FILL, BAR_EDGE),
)

fig, axes = plt.subplots(1, 2, figsize=SIV_SPIN_FIGSIZE, sharey=True)
lines = ["Supplementary SiV electron-spin ODMR and Rabi oscillations"]
for index, (
    (run_id, x, nof_transitions, shots, probability, error, fit),
    label,
    (fill, edge),
) in enumerate(zip(spin_scans, SIV_SPIN_LABELS, SIV_SPIN_COLORS)):
    ax = axes[0 if index < 2 else 1]
    grid = np.linspace(x.min(), x.max(), SIV_SPIN_FIT_POINTS)
    plot_x = 17.1 + x / 1000 if index < 2 else x
    plot_grid = 17.1 + grid / 1000 if index < 2 else grid
    ax.errorbar(plot_x, probability, yerr=error, fmt="none", **ERRORBAR_STYLE, zorder=2)
    ax.plot(
        plot_x,
        probability,
        "o",
        color=fill,
        markeredgecolor=edge,
        markeredgewidth=0.3,
        markersize=SIV_SPIN_MARKER_SIZE,
        label=label,
        zorder=3,
    )
    ax.plot(
        plot_grid,
        fit.eval(x=grid),
        color=edge,
        lw=0.8,
        label="Fit" if index == 2 else f"{label} fit",
    )
    if index < 2:
        lines.append(
            "Fit center and omega in MHz; center is relative to 17.1 GHz. Duration in microseconds."
        )
    else:
        lines.append("Sine-model frequency in rad/ns; shift in rad.")
    lines.extend(
        [
            format_fit_report(fit),
            "frequency_GHz" if index < 2 else "duration_ns",
            "coordinate nof_transitions shots probability standard_error fitted_probability",
        ]
    )
    lines.extend(
        f"{v:.9f} {k:d} {shots:d} {p:.8f} {e:.8f} {f:.8f}"
        for v, k, p, e, f in zip(
            plot_x,
            nof_transitions,
            probability,
            error,
            fit.best_fit,
        )
    )
axes[0].set(
    xlabel="Frequency (GHz)", ylabel="Transition probability", ylim=(-0.04, 1.07)
)
axes[0].set_xticks([17.10, 17.15, 17.20, 17.25, 17.30])
axes[1].set_xlabel("Duration (ns)")
for label, ax in zip("ab", axes):
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=0.3, foreground="black")],
    )
axes[0].legend(loc="upper left", frameon=False, fontsize=7, borderaxespad=1.0)
axes[1].legend(loc="upper right", frameon=False, fontsize=7, borderaxespad=1.0)
save_figure(fig, "supplementary_siv_odmr_rabi")
(FIGURES_DIR / "txt" / "supplementary_siv_odmr_rabi.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## Detector-conditioned SiV-photon correlations

# %%
photon_results = {}
for run_id in (SIV_PHOTON_ZZ_WAVEPLATE_SWEEP, SIV_PHOTON_HV_POPULATIONS):
    ds = load_dataset(run_id, "main.h5")
    assert np.isin(ds.siv_final_readout_state.values, [0, 1]).all()
    assert np.isin(ds.which_spcm_clicked.values, [1, 2]).all()
    angles = ds.bc1_output_half.values
    assert np.isfinite(angles).all()
    if run_id == SIV_PHOTON_HV_POPULATIONS:
        assert angles.size == 1
    for detector in (1, 2):
        selected = ds.which_spcm_clicked == detector
        counts = selected.sum("success_idx").values
        successes = (
            (selected & (ds.siv_final_readout_state == 1)).sum("success_idx").values
        )
        assert np.all(counts > 0)
        probability = successes / counts
        # Keep error bars nonzero at p = 0 or 1 without changing the measured probability.
        probability_for_error = (successes + 0.5) / (counts + 1)
        error = np.sqrt(probability_for_error * (1 - probability_for_error) / counts)
        fit = None
        if run_id == SIV_PHOTON_ZZ_WAVEPLATE_SWEEP:
            sine = SineModel()
            params = sine.guess(probability, x=angles)
            params.add("c", value=probability.mean())
            params["frequency"].set(value=2 * np.pi / 90, vary=False)
            fit = (sine + ConstantModel()).fit(
                probability,
                params,
                x=angles,
                weights=1 / error,
                scale_covar=False,
            )
            assert fit.success, fit.message
        photon_results[run_id, detector] = (
            angles,
            counts,
            successes,
            probability,
            error,
            fit,
        )

detector_1_fit = photon_results[SIV_PHOTON_ZZ_WAVEPLATE_SWEEP, 1][-1]
hv_angle = (-np.pi / 2 - detector_1_fit.params["shift"].value) / (2 * np.pi / 90)
sweep_start = photon_results[SIV_PHOTON_ZZ_WAVEPLATE_SWEEP, 1][0].min()
hv_angle += 90 * np.round((sweep_start - hv_angle) / 90)
diagonal_angle = hv_angle + 22.5

# %%
SIV_PHOTON_FIGSIZE = (DOUBLE_COLUMN_WIDTH_MM * MM, 78 * MM)
SIV_PHOTON_COLORS = ((BAR_FILL, BAR_EDGE), (IONQ_ORANGE, IONQ_ORANGE_EDGE))
SIV_PHOTON_MARKER_SIZE = 3

fig, axes = plt.subplots(1, 2, figsize=SIV_PHOTON_FIGSIZE)
lines = [
    "Supplementary detector-conditioned SiV-photon correlations",
    f"Detector-1 fitted H/V minimum: {hv_angle:.8f} deg; +/- basis angle: {diagonal_angle:.8f} deg",
]
for detector, (fill, edge) in zip((1, 2), SIV_PHOTON_COLORS):
    angles, counts, successes, probability, error, fit = photon_results[
        SIV_PHOTON_ZZ_WAVEPLATE_SWEEP, detector
    ]
    axes[0].errorbar(
        angles, probability, yerr=error, fmt="none", **ERRORBAR_STYLE, zorder=2
    )
    axes[0].plot(
        angles,
        probability,
        "o",
        color=fill,
        markeredgecolor=edge,
        markersize=SIV_PHOTON_MARKER_SIZE,
        markeredgewidth=0.5,
        label=f"Detector {detector} data",
        zorder=3,
    )
    grid = np.linspace(angles.min(), angles.max(), 1000)
    axes[0].plot(
        grid, fit.eval(x=grid), color=edge, lw=0.8, label=f"Detector {detector} fit"
    )
    lines.extend(
        [
            "",
            "Shift in radians.",
            format_fit_report(fit),
            "angle_deg detector_count state_1_count probability standard_error fitted_probability",
        ]
    )
    lines.extend(
        f"{a:.8f} {n:d} {k:d} {p:.8f} {e:.8f} {f:.8f}"
        for a, n, k, p, e, f in zip(
            angles,
            counts,
            successes,
            probability,
            error,
            fit.best_fit,
        )
    )
    fixed_angles, n, k, p, e, _ = photon_results[SIV_PHOTON_HV_POPULATIONS, detector]
    positions = np.array([0, 1]) + 2 * (detector - 1)
    populations = [1 - p.item(), p.item()]
    axes[1].bar(
        positions, populations, width=0.7, color=fill, edgecolor=edge, linewidth=0.6
    )
    axes[1].errorbar(
        positions, populations, yerr=e.item(), fmt="none", **ERRORBAR_STYLE
    )
    lines.extend(
        [
            "",
            f"Waveplate angle: {fixed_angles.item():g} deg; clicks: {n.item()}; state-1 events: {k.item()}",
            f"P(0): {format_uncertainty(populations[0], e.item())}; P(1): {format_uncertainty(populations[1], e.item())}",
        ]
    )
    for position, value in zip(positions, populations):
        axes[1].text(
            position,
            value + e.item() + 0.025,
            f"{format_uncertainty(100 * value, 100 * e.item())} %",
            ha="center",
            va="bottom",
            fontsize=5,
        )
axes[0].axvline(
    hv_angle,
    ymax=0.84,
    color="0.5",
    linestyle="--",
    lw=0.8,
    label="Detector 1: H; detector 2: V",
)
axes[0].axvline(
    diagonal_angle,
    ymax=0.84,
    color=IONQ_ORANGE_EDGE,
    linestyle="--",
    lw=0.8,
    label="Detector 1: +; detector 2: −",
)
axes[0].set(
    xlabel="Half-waveplate angle (deg)",
    ylabel="SiV-state probability P(1|Detector)",
    ylim=(-0.04, 1.25),
)
axes[0].set_yticks(np.linspace(0, 1, 6))
axes[0].legend(loc="upper center", ncol=3, frameon=False, fontsize=6, borderaxespad=1.0)
axes[1].set(
    xticks=np.arange(4),
    xticklabels=["P(0|H)", "P(1|H)", "P(0|V)", "P(1|V)"],
    ylabel="Conditional probability",
    ylim=(0, 1.15),
)
for label, ax in zip("ab", axes):
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=0.3, foreground="black")],
    )
save_figure(fig, "supplementary_siv_photon_correlations")
(FIGURES_DIR / "txt" / "supplementary_siv_photon_correlations.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

# %% [markdown]
# ## Mid-circuit reset and fast readout

# %%
reset_results = []
for run_id in (SIV_NO_RESET, SIV_MID_CIRCUIT_RESET):
    ds = load_dataset(run_id, "dataset_0.h5")
    metadata = load_provenance(run_id)["metadata"]
    assert metadata["initial_electron_state"] == 1
    assert metadata["do_mid_circuit_correction"] == (run_id == SIV_MID_CIRCUIT_RESET)
    assert np.isin(ds.final_readout_state.values, [0, 1]).all()
    gates = ds.attempts_before_long_read.values
    assert (
        np.isfinite(gates).all() and np.all(gates >= 0) and np.all(np.diff(gates) > 0)
    )
    shots = ds.sizes["rep"]
    successes = (ds.final_readout_state == 1).sum("rep").values
    fidelity = successes / shots
    # Keep error bars nonzero at p = 0 or 1 without changing the measured probability.
    probability_for_error = (successes + 0.5) / (shots + 1)
    error = np.sqrt(probability_for_error * (1 - probability_for_error) / shots)
    model = ExponentialModel() + ConstantModel()
    params = model.make_params(
        amplitude=fidelity[0] - fidelity[-1],
        decay=12 if run_id == SIV_MID_CIRCUIT_RESET else 160,
        c=fidelity[-1],
    )
    params["amplitude"].set(min=0, max=1)
    params["decay"].set(min=0.001)
    params["c"].set(min=0, max=1)
    fit = model.fit(fidelity, params, x=gates, weights=1 / error, scale_covar=False)
    assert fit.success, fit.message
    reset_results.append((run_id, gates, shots, successes, fidelity, error, fit))

fast_counts = load_dataset(
    ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "main.h5"
).siv_init_readout_counts.values.ravel()
assert np.isfinite(fast_counts).all() and np.all(fast_counts >= 0)
assert np.all(fast_counts == np.floor(fast_counts))
fast_metadata = load_provenance(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1)["metadata"]
fast_threshold = fast_metadata["dr2.siv.fast_readout.hi_trsh"]
fast_duration_us = fast_metadata["dr2.siv.fast_readout.duration"] * 1e6
assert fast_threshold == int(fast_threshold) and fast_duration_us > 0
fast_photon_number = np.arange(int(fast_counts.max()) + 1)
fast_probability = np.bincount(fast_counts.astype(int)) / fast_counts.size
assert np.isclose(fast_probability.sum(), 1)

# %%
SIV_RESET_FIGSIZE = (DOUBLE_COLUMN_WIDTH_MM * MM, 70 * MM)
SIV_RESET_MARKER_SIZE = 3
SIV_RESET_COLORS = ((BAR_FILL, BAR_EDGE), (IONQ_ORANGE, IONQ_ORANGE_EDGE))
SIV_RESET_LABELS = ("Simulated decoherence", "Mid-circuit reset")

fig, axes = plt.subplots(1, 2, figsize=SIV_RESET_FIGSIZE)
lines = ["Supplementary SiV mid-circuit reset and fast readout"]
for (run_id, gates, shots, successes, fidelity, error, fit), label, (fill, edge) in zip(
    reset_results,
    SIV_RESET_LABELS,
    SIV_RESET_COLORS,
):
    axes[0].errorbar(
        gates, fidelity, yerr=error, fmt="none", **ERRORBAR_STYLE, zorder=2
    )
    axes[0].plot(
        gates,
        fidelity,
        "o",
        color=fill,
        markeredgecolor=edge,
        markeredgewidth=0.5,
        markersize=SIV_RESET_MARKER_SIZE,
        label=label,
        zorder=3,
    )
    if run_id == SIV_NO_RESET:
        grid = np.linspace(gates.min(), gates.max(), 1000)
        axes[0].plot(grid, fit.eval(x=grid), color=edge, lw=0.8, label="No-reset fit")
    else:
        plateau = fit.params["c"]
        axes[0].axhline(
            plateau.value,
            color=edge,
            lw=0.8,
            linestyle="--",
            label=f"Reset plateau: F = {format_uncertainty(plateau.value, plateau.stderr)}",
        )
    lines.extend(
        [
            "",
            format_fit_report(fit),
            "spin_photon_gates shots state_1_count fidelity standard_error fitted_fidelity",
        ]
    )
    lines.extend(
        f"{n:d} {shots:d} {k:d} {p:.8f} {e:.8f} {f:.8f}"
        for n, k, p, e, f in zip(
            gates,
            successes,
            fidelity,
            error,
            fit.best_fit,
        )
    )
axes[0].set(xlabel="Number of spin-photon gates", ylabel="Fidelity", ylim=(0.5, 1.03))
axes[0].legend(loc="center right", frameon=False, fontsize=7, borderaxespad=1.0)

for mask, color, label in (
    (fast_photon_number < fast_threshold, BAR_FILL, "Assigned spin down state"),
    (fast_photon_number >= fast_threshold, IONQ_ORANGE, "Assigned spin up state"),
):
    axes[1].bar(
        fast_photon_number[mask],
        fast_probability[mask],
        width=1,
        linewidth=0,
        color=color,
        label=label,
    )
axes[1].axvline(
    fast_threshold - 0.5,
    color=IONQ_ORANGE_EDGE,
    lw=0.8,
    linestyle="--",
    label="Readout threshold",
)
axes[1].set(
    xlabel=f"Number of photons collected in a {fast_duration_us:g} µs readout",
    ylabel="Probability",
)
axes[1].legend(loc="upper right", frameon=False, fontsize=6, borderaxespad=1.0)
for label, ax in zip("ab", axes):
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        path_effects=[patheffects.withStroke(linewidth=0.3, foreground="black")],
    )
save_figure(fig, "supplementary_siv_mid_circuit_reset")
lines.extend(
    [
        "",
        f"Shots: {fast_counts.size}; readout: {fast_duration_us:g} µs; threshold: {int(fast_threshold)} photons",
        "photon_number probability",
    ]
)
lines.extend(f"{n:d} {p:.8f}" for n, p in zip(fast_photon_number, fast_probability))
(FIGURES_DIR / "txt" / "supplementary_siv_mid_circuit_reset.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
