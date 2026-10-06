"""Plot the Ion/SiV/Photon correlator sweeps.

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
# # Figure 3: Ion-SiV sweeps

# %%
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ConstantModel, SineModel
from scipy.optimize import brentq

from analysis.tomography import (
    calculate_histogram_population_and_correlator,
    make_two_qubit_state,
)
from papertools import (
    BAR_EDGE,
    BAR_FILL,
    DOUBLE_COLUMN_WIDTH_MM,
    ERRORBAR_STYLE,
    FIGURES_DIR,
    IONQ_ORANGE,
    MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
    load_dataset,
    save_figure,
)
from papertools.formatting import format_fit_report
from papertools.run_ids import (
    ION_SIV_XX_SWEEP_PHASE_COMP_ON,
    ION_SIV_ZZ_SWEEP,
    SIV_PHOTON_BASIS_CONVERTER_OUTPUT_HWP_SWEEP,
)

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Load data: Photon-arrival-time phase compensation on

# %%
main_ds = load_dataset(ION_SIV_XX_SWEEP_PHASE_COMP_ON, "main.h5")

# %% [markdown]
# ## Two-qubit populations and correlator

# %%
ion_siv_state = make_two_qubit_state(
    state_1=main_ds["ion_state"],
    state_1_high_value=1,
    state_2=main_ds["siv_final_readout_state"],
    state_2_high_value=1,
    name="ion_siv",
    label="Ion-SiV",
)

ion_siv_state_hist, ion_siv_population, ion_siv_corr = (
    calculate_histogram_population_and_correlator(
        ion_siv_state, label="Ion-SiV", prefix="ion_siv", rep_dim="success_idx"
    )
)

# %% [markdown]
# ## Ion-SiV XX correlator sweep
#
# Both qubits are measured in the X basis.
# We sweep the phase of a Z rotation applied to the ion before readout and
# select the phase that maximizes XX for the target Bell state |Phi+>.
# The 1762 AOM is double pass, so its phase is imprinted twice:
# the ion Z-rotation angle is twice the 1762 AOM phase.

# %%
XX_SWEEP_PANEL_RATIO = 0.25
XX_SWEEP_WIDTH = DOUBLE_COLUMN_WIDTH_MM * XX_SWEEP_PANEL_RATIO * MM
XX_SWEEP_FIGSIZE = (XX_SWEEP_WIDTH, XX_SWEEP_WIDTH * 5 / 6.5)
XX_SWEEP_MARKER_SIZE = 3
XX_SWEEP_MARKER_EDGE_WIDTH = 0.6
XX_SWEEP_YLIM = (-1.02, 1.02)
XX_SWEEP_AOM_PERIOD = 180
XX_SWEEP_FIT_LINEWIDTH = 0.8

# %%
mean = ion_siv_corr["ion_siv_corr_mean"].squeeze()
minus_error = ion_siv_corr["ion_siv_corr_minus_error"].squeeze()
plus_error = ion_siv_corr["ion_siv_corr_plus_error"].squeeze()
(phase_dim,) = mean.dims
aom_phase = mean[phase_dim].values
readout_angle = 2 * aom_phase

# %%
# SineModel uses angular frequency; the period is fixed in 1762 AOM degrees.
sine_model = SineModel()
fit_model = sine_model + ConstantModel()
fit_params = sine_model.guess(mean.values, x=aom_phase)
fit_params.add("c", value=mean.mean().item())
fit_params["frequency"].set(value=2 * np.pi / XX_SWEEP_AOM_PERIOD, vary=False)
fit_params["shift"].set(min=-np.inf, max=np.inf)
# Approximate the asymmetric correlator errors by sigma = (lower + upper)/2
# and assume independent Gaussian residuals.
# Parameter errors use the local least-squares covariance; scale_covar=False
# treats sigma as absolute, without reduced-chi-square rescaling.
fit_result = fit_model.fit(
    mean.values,
    fit_params,
    x=aom_phase,
    weights=1 / ((minus_error.values + plus_error.values) / 2),
    scale_covar=False,
)
if not fit_result.success:
    raise RuntimeError(f"Sine fit failed: {fit_result.message}")
fit_aom_phase = np.linspace(aom_phase.min(), aom_phase.max(), 1000)
xx_optimal_aom_phase = (
    (np.pi / 2 - fit_result.params["shift"].value) % (2 * np.pi)
) / fit_result.params["frequency"].value
xx_optimal_correlator = fit_result.eval(x=xx_optimal_aom_phase)

# %%
fig, ax = plt.subplots(figsize=XX_SWEEP_FIGSIZE)
ax.errorbar(
    readout_angle,
    mean.values,
    yerr=[minus_error.values, plus_error.values],
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=2,
)
(line,) = mean.assign_coords({phase_dim: readout_angle}).plot(
    ax=ax,
    marker="o",
    linestyle="none",
    color=BAR_FILL,
    markersize=XX_SWEEP_MARKER_SIZE,
    markeredgecolor=BAR_EDGE,
    markeredgewidth=XX_SWEEP_MARKER_EDGE_WIDTH,
    zorder=3,
)
ax.plot(
    2 * fit_aom_phase,
    fit_result.eval(x=fit_aom_phase),
    color=BAR_EDGE,
    linewidth=XX_SWEEP_FIT_LINEWIDTH,
    zorder=2,
)
ax.set_ylim(*XX_SWEEP_YLIM)
ax.set_xlabel("Ion Z-rotation angle (deg)")
ax.set_xticks(np.arange(0, 361, 90))
ax.set_ylabel("Ion-SiV XX Correlator")
ax.plot(
    2 * xx_optimal_aom_phase,
    xx_optimal_correlator,
    marker="o",
    markersize=5,
    markerfacecolor="none",
    markeredgecolor=IONQ_ORANGE,
    markeredgewidth=0.8,
    linestyle="none",
    zorder=4,
)
ax.annotate(
    "Optimal",
    xy=(2 * xx_optimal_aom_phase, xx_optimal_correlator),
    xytext=(0, -24),
    textcoords="offset points",
    fontsize=SMALL_FONT_SIZE,
    color="black",
    ha="center",
    va="top",
    arrowprops={
        "arrowstyle": "simple,head_width=0.4,tail_width=0.12",
        "color": IONQ_ORANGE,
        "lw": 0.5,
        "shrinkB": 4,
        "relpos": (0.5, 1),
    },
    zorder=4,
)
ax.set_title("")
save_figure(fig, "figure_3_xx_sweep")

lines = [
    "Figure 3: Ion-SiV XX sweep, phase compensation on",
    "",
    f"{'1762 phase (deg)':>18} {'Ion phase (deg)':>20} "
    f"{'Mean':>14} {'Lower error':>14} {'Upper error':>14}",
]
for phase, angle, y, lower, upper in zip(
    aom_phase, readout_angle, mean.values, minus_error.values, plus_error.values
):
    lines.append(f"{phase:18.8f} {angle:20.8f} {y:14.8f} {lower:14.8f} {upper:14.8f}")
lines.extend(
    [
        "",
        f"Fitted maximum: {xx_optimal_correlator:.8f} at AOM phase {xx_optimal_aom_phase:.8f} deg "
        f"(ion Z-rotation angle {2 * xx_optimal_aom_phase:.8f} deg)",
        "Frequency is in radians per AOM degree; shift is in radians.",
        "",
        format_fit_report(fit_result),
    ]
)
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_3_xx_sweep.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Load data: Ion-SiV ZZ sweep

# %%
zz_main_ds = load_dataset(ION_SIV_ZZ_SWEEP, "main.h5")

# %%
zz_ion_siv_state = make_two_qubit_state(
    state_1=zz_main_ds["ion_state"],
    state_1_high_value=1,
    state_2=zz_main_ds["siv_final_readout_state"],
    state_2_high_value=1,
    name="ion_siv",
    label="Ion-SiV",
)
zz_ion_siv_state_hist, zz_ion_siv_population, zz_ion_siv_corr = (
    calculate_histogram_population_and_correlator(
        zz_ion_siv_state, label="Ion-SiV", prefix="ion_siv", rep_dim="success_idx"
    )
)

# %% [markdown]
# ## Ion-SiV ZZ correlator sweep

# %%
ZZ_SWEEP_PANEL_RATIO = 0.25
ZZ_SWEEP_WIDTH = DOUBLE_COLUMN_WIDTH_MM * ZZ_SWEEP_PANEL_RATIO * MM
ZZ_SWEEP_FIGSIZE = (ZZ_SWEEP_WIDTH, ZZ_SWEEP_WIDTH * 5 / 6.5)
ZZ_SWEEP_MARKER_SIZE = 3
ZZ_SWEEP_MARKER_EDGE_WIDTH = 0.6
ZZ_SWEEP_YLIM = (-1.02, 1.02)
ZZ_SWEEP_PERIOD = 90
ZZ_SWEEP_FIT_LINEWIDTH = 0.8

# %%
zz_mean = zz_ion_siv_corr["ion_siv_corr_mean"].squeeze()
zz_minus_error = zz_ion_siv_corr["ion_siv_corr_minus_error"].squeeze()
zz_plus_error = zz_ion_siv_corr["ion_siv_corr_plus_error"].squeeze()
(zz_angle_dim,) = zz_mean.dims
hwp_angle = zz_mean[zz_angle_dim].values

zz_sine_model = SineModel()
zz_fit_model = zz_sine_model + ConstantModel()
zz_fit_params = zz_sine_model.guess(zz_mean.values, x=hwp_angle)
zz_fit_params.add("c", value=zz_mean.mean().item())
zz_fit_params["frequency"].set(value=2 * np.pi / ZZ_SWEEP_PERIOD, vary=False)
zz_fit_params["shift"].set(min=-np.inf, max=np.inf)
# Same weights and covariance treatment as the XX fit.
zz_fit_result = zz_fit_model.fit(
    zz_mean.values,
    zz_fit_params,
    x=hwp_angle,
    weights=1 / ((zz_minus_error.values + zz_plus_error.values) / 2),
    scale_covar=False,
)
if not zz_fit_result.success:
    raise RuntimeError(f"ZZ sine fit failed: {zz_fit_result.message}")
fit_hwp_angle = np.linspace(hwp_angle.min(), hwp_angle.max(), 1000)
zz_optimal_hwp_angle = (
    (np.pi / 2 - zz_fit_result.params["shift"].value) % (2 * np.pi)
) / zz_fit_result.params["frequency"].value
zz_optimal_correlator = zz_fit_result.eval(x=zz_optimal_hwp_angle)

# %%
fig, ax = plt.subplots(figsize=ZZ_SWEEP_FIGSIZE)
ax.errorbar(
    hwp_angle,
    zz_mean.values,
    yerr=[zz_minus_error.values, zz_plus_error.values],
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=2,
)
zz_mean.plot(
    ax=ax,
    marker="o",
    linestyle="none",
    color=BAR_FILL,
    markersize=ZZ_SWEEP_MARKER_SIZE,
    markeredgecolor=BAR_EDGE,
    markeredgewidth=ZZ_SWEEP_MARKER_EDGE_WIDTH,
    zorder=3,
)
ax.plot(
    fit_hwp_angle,
    zz_fit_result.eval(x=fit_hwp_angle),
    color=BAR_EDGE,
    linewidth=ZZ_SWEEP_FIT_LINEWIDTH,
    zorder=2,
)
ax.set_ylim(*ZZ_SWEEP_YLIM)
ax.set_xlabel("Basis Converter Input HWP Angle (deg)", x=0.35)
ax.set_xticks([0, 30, 60, 90])
ax.set_ylabel("Ion-SiV ZZ Correlator")
ax.plot(
    zz_optimal_hwp_angle,
    zz_optimal_correlator,
    marker="o",
    markersize=5,
    markerfacecolor="none",
    markeredgecolor=IONQ_ORANGE,
    markeredgewidth=0.8,
    linestyle="none",
    zorder=4,
)
ax.annotate(
    "Optimal",
    xy=(zz_optimal_hwp_angle, zz_optimal_correlator),
    xytext=(0, -24),
    textcoords="offset points",
    fontsize=SMALL_FONT_SIZE,
    color="black",
    ha="center",
    va="top",
    arrowprops={
        "arrowstyle": "simple,head_width=0.4,tail_width=0.12",
        "color": IONQ_ORANGE,
        "lw": 0.5,
        "shrinkB": 4,
        "relpos": (0.5, 1),
    },
    zorder=4,
)
ax.set_title("")
save_figure(fig, "figure_3_zz_sweep")

lines = [
    "Figure 3: Ion-SiV ZZ sweep",
    "",
    f"{'HWP angle (deg)':>18} {'Mean':>14} {'Lower error':>14} {'Upper error':>14}",
]
for angle, y, lower, upper in zip(
    hwp_angle, zz_mean.values, zz_minus_error.values, zz_plus_error.values
):
    lines.append(f"{angle:18.8f} {y:14.8f} {lower:14.8f} {upper:14.8f}")
lines.extend(
    [
        "",
        f"Fitted maximum: {zz_optimal_correlator:.8f} at HWP angle {zz_optimal_hwp_angle:.8f} deg",
        "Frequency is in radians per HWP degree; shift is in radians.",
        "",
        format_fit_report(zz_fit_result),
    ]
)
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_3_zz_sweep.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Figure 3: SiV-photon sweep

# %%
siv_photon_main_ds = load_dataset(
    SIV_PHOTON_BASIS_CONVERTER_OUTPUT_HWP_SWEEP, "main.h5"
)

# %%
siv_photon_state = make_two_qubit_state(
    state_1=siv_photon_main_ds["siv_final_readout_state"],
    state_1_high_value=1,
    state_2=siv_photon_main_ds["which_spcm_clicked"],
    state_2_high_value=2,
    name="siv_photon",
    label="SiV-Photon",
)
siv_photon_state_hist, siv_photon_population, siv_photon_corr = (
    calculate_histogram_population_and_correlator(
        siv_photon_state, label="SiV-Photon", prefix="siv_photon", rep_dim="success_idx"
    )
)

# %%
SIV_PHOTON_SWEEP_PANEL_RATIO = 0.25
SIV_PHOTON_SWEEP_WIDTH = DOUBLE_COLUMN_WIDTH_MM * SIV_PHOTON_SWEEP_PANEL_RATIO * MM
SIV_PHOTON_SWEEP_FIGSIZE = (SIV_PHOTON_SWEEP_WIDTH, SIV_PHOTON_SWEEP_WIDTH * 5 / 6.5)
SIV_PHOTON_SWEEP_MARKER_SIZE = 3
SIV_PHOTON_SWEEP_MARKER_EDGE_WIDTH = 0.6
SIV_PHOTON_SWEEP_YLIM = (-1.02, 1.02)
SIV_PHOTON_SWEEP_PERIOD = 90
SIV_PHOTON_SWEEP_FIT_LINEWIDTH = 0.8

# %%
siv_photon_mean = siv_photon_corr["siv_photon_corr_mean"].squeeze()
siv_photon_minus_error = siv_photon_corr["siv_photon_corr_minus_error"].squeeze()
siv_photon_plus_error = siv_photon_corr["siv_photon_corr_plus_error"].squeeze()
(siv_photon_angle_dim,) = siv_photon_mean.dims
output_hwp_angle = siv_photon_mean[siv_photon_angle_dim].values

siv_photon_sine_model = SineModel()
siv_photon_fit_model = siv_photon_sine_model + ConstantModel()
siv_photon_fit_params = siv_photon_sine_model.guess(
    siv_photon_mean.values, x=output_hwp_angle
)
siv_photon_fit_params.add("c", value=siv_photon_mean.mean().item())
siv_photon_fit_params["frequency"].set(
    value=2 * np.pi / SIV_PHOTON_SWEEP_PERIOD, vary=False
)
siv_photon_fit_params["shift"].set(min=-np.inf, max=np.inf)
# Same weights and covariance treatment as the XX fit.
siv_photon_fit_result = siv_photon_fit_model.fit(
    siv_photon_mean.values,
    siv_photon_fit_params,
    x=output_hwp_angle,
    weights=1 / ((siv_photon_minus_error.values + siv_photon_plus_error.values) / 2),
    scale_covar=False,
)
if not siv_photon_fit_result.success:
    raise RuntimeError(f"SiV-photon sine fit failed: {siv_photon_fit_result.message}")
fit_output_hwp_angle = np.linspace(output_hwp_angle.min(), output_hwp_angle.max(), 1000)
siv_photon_sweep_center = (output_hwp_angle.min() + output_hwp_angle.max()) / 2
siv_photon_zero_angle = brentq(
    lambda angle: siv_photon_fit_result.eval(x=angle),
    siv_photon_sweep_center - SIV_PHOTON_SWEEP_PERIOD / 4,
    siv_photon_sweep_center + SIV_PHOTON_SWEEP_PERIOD / 4,
)

# %%
fig, ax = plt.subplots(figsize=SIV_PHOTON_SWEEP_FIGSIZE)
ax.errorbar(
    output_hwp_angle,
    siv_photon_mean.values,
    yerr=[siv_photon_minus_error.values, siv_photon_plus_error.values],
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=2,
)
siv_photon_mean.plot(
    ax=ax,
    marker="o",
    linestyle="none",
    color=BAR_FILL,
    markersize=SIV_PHOTON_SWEEP_MARKER_SIZE,
    markeredgecolor=BAR_EDGE,
    markeredgewidth=SIV_PHOTON_SWEEP_MARKER_EDGE_WIDTH,
    zorder=3,
)
ax.plot(
    fit_output_hwp_angle,
    siv_photon_fit_result.eval(x=fit_output_hwp_angle),
    color=BAR_EDGE,
    linewidth=SIV_PHOTON_SWEEP_FIT_LINEWIDTH,
    zorder=2,
)
ax.set_ylim(*SIV_PHOTON_SWEEP_YLIM)
ax.set_xlabel("Basis Converter Output HWP Angle (deg)", x=0.35)
ax.set_xticks([0, 30, 60, 90])
ax.set_ylabel("SiV-Photon ZZ Correlator", y=0.45)
ax.plot(
    siv_photon_zero_angle,
    0,
    marker="o",
    markersize=5,
    markerfacecolor="none",
    markeredgecolor=IONQ_ORANGE,
    markeredgewidth=0.8,
    linestyle="none",
    zorder=4,
)
ax.annotate(
    "Optimal",
    xy=(siv_photon_zero_angle, 0),
    xytext=(14, 12),
    textcoords="offset points",
    fontsize=SMALL_FONT_SIZE,
    color="black",
    ha="left",
    va="center",
    arrowprops={
        "arrowstyle": "simple,head_width=0.4,tail_width=0.12",
        "color": IONQ_ORANGE,
        "lw": 0.5,
        "shrinkB": 4,
        "relpos": (0, 0.5),
    },
    zorder=4,
)
ax.set_title("")
save_figure(fig, "figure_3_siv_photon_sweep")

lines = [
    "Figure 3: SiV-photon sweep",
    "",
    f"{'Output HWP (deg)':>18} {'Mean':>14} {'Lower error':>14} {'Upper error':>14}",
]
for angle, y, lower, upper in zip(
    output_hwp_angle,
    siv_photon_mean.values,
    siv_photon_minus_error.values,
    siv_photon_plus_error.values,
):
    lines.append(f"{angle:18.8f} {y:14.8f} {lower:14.8f} {upper:14.8f}")
lines.extend(
    [
        "",
        f"Middle zero crossing: {siv_photon_zero_angle:.8f} deg",
        "Frequency is in radians per HWP degree; shift is in radians.",
        "",
        format_fit_report(siv_photon_fit_result),
    ]
)
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_3_siv_photon_sweep.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
