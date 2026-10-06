"""Plot the Ion-SiV tomography correlators and reconstructed density matrix.

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
# # Figure 1: Tomography

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis import density_matrix_mle
from analysis.tomography import (
    calculate_histogram_population_and_correlator,
    make_two_qubit_state,
)
from papertools import (
    BAR_STYLE,
    DENSITY_MATRIX_COLORS,
    DOUBLE_COLUMN_WIDTH_MM,
    ERRORBAR_STYLE,
    FIGURES_DIR,
    FLOOR_COLOR,
    MM,
    SECONDARY_TEXT,
    SINGLE_COLUMN_WIDTH_MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
    load_dataset,
    save_figure,
)
from papertools.formatting import format_uncertainty
from papertools.run_ids import ION_SIV_FULL_TOMOGRAPHY_FIGURE_1

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Load data

# %%
main_ds = load_dataset(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "main.h5")

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
# ## Supplementary tomography correlators

# %%
CORRELATOR_PANEL_RATIO = 1.0
CORRELATOR_WIDTH = SINGLE_COLUMN_WIDTH_MM * CORRELATOR_PANEL_RATIO * MM
CORRELATOR_FIGSIZE = (CORRELATOR_WIDTH, CORRELATOR_WIDTH * 5 / 6.5)
CORRELATOR_ZERO_COLOR = "0.65"
CORRELATOR_ZERO_LINEWIDTH = 0.5
CORRELATOR_YLIM = (-1.2, 1.2)
CORRELATOR_LABEL_OFFSET_PT = 4

# %%
fig, ax = plt.subplots(figsize=CORRELATOR_FIGSIZE)
ax.bar(
    ion_siv_corr.correlator,
    ion_siv_corr["ion_siv_corr_mean"],
    **BAR_STYLE,
    zorder=2,
)
ax.errorbar(
    ion_siv_corr.correlator,
    ion_siv_corr["ion_siv_corr_mean"],
    yerr=[
        ion_siv_corr["ion_siv_corr_minus_error"],
        ion_siv_corr["ion_siv_corr_plus_error"],
    ],
    fmt="none",
    **ERRORBAR_STYLE,
    zorder=3,
)
ax.axhline(
    0, color=CORRELATOR_ZERO_COLOR, linewidth=CORRELATOR_ZERO_LINEWIDTH, zorder=1
)
for index, (value, lower, upper) in enumerate(
    zip(
        ion_siv_corr["ion_siv_corr_mean"].values,
        ion_siv_corr["ion_siv_corr_minus_error"].values,
        ion_siv_corr["ion_siv_corr_plus_error"].values,
    )
):
    positive = value >= 0
    ax.annotate(
        format_uncertainty(value, lower, upper),
        xy=(index, value + upper if positive else value - lower),
        xytext=(
            0,
            CORRELATOR_LABEL_OFFSET_PT if positive else -CORRELATOR_LABEL_OFFSET_PT,
        ),
        textcoords="offset points",
        ha="center",
        va="bottom" if positive else "top",
        fontsize=SMALL_FONT_SIZE,
    )
ax.set_ylim(*CORRELATOR_YLIM)
ax.set_yticks(np.linspace(-1, 1, 5))
ax.set_ylabel("Correlator Value")
ax.set_xlabel("Ion-SiV Correlator")

save_figure(fig, "supplementary_tomography_correlators")

lines = [
    "Supplementary tomography correlators",
    "",
    f"{'Correlator':<12} {'Value':>14} {'Lower error':>14} {'Upper error':>14}",
]
for correlator in ion_siv_corr.correlator.values:
    values = ion_siv_corr.sel(correlator=correlator)
    lines.append(
        f"{correlator:<12} {values['ion_siv_corr_mean'].item():14.8f} "
        f"{values['ion_siv_corr_minus_error'].item():14.8f} "
        f"{values['ion_siv_corr_plus_error'].item():14.8f}"
    )
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_tomography_correlators.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

# %% [markdown]
# ## Density matrix reconstruction

# %%
# make_two_qubit_state encodes state = ion + 2 * siv: bit 0b01 is the ion and
# bit 0b10 is the SiV. The outcome string lists the ion bit first, to match
# the Ion-SiV order of the measurement settings.
tomography_result = {
    ion_siv_state_hist.correlator[i].item(): {
        f"{state & 0b01}{(state & 0b10) >> 1}": ion_siv_state_hist[i, state].item()
        for state in range(4)
    }
    for i in range(ion_siv_state_hist.correlator.size)
}

# %%
print("Density matrix reconstruction will take approximately 30 seconds on a laptop...")
result = density_matrix_mle.reconstruct_density_matrix(
    tomography_result, num_bootstrap_samples=1000
)

# %% [markdown]
# ## Figure 1: Real part of the density matrix

# %%
DENSITY_MATRIX_PANEL_RATIO = 0.25
DENSITY_MATRIX_WIDTH = DOUBLE_COLUMN_WIDTH_MM * DENSITY_MATRIX_PANEL_RATIO * MM
DENSITY_MATRIX_FIGSIZE = (DENSITY_MATRIX_WIDTH, DENSITY_MATRIX_WIDTH * 5 / 6.5)
DENSITY_MATRIX_BASIS_LABELS = ("00", "01", "10", "11")
DENSITY_MATRIX_ZLIM = (0, 0.5)
DENSITY_MATRIX_VIEW_ANGLES = (28, -55)
DENSITY_MATRIX_FOCAL_LENGTH = 0.7
DENSITY_MATRIX_AXES_POSITION = (-0.035, 0.10, 1.0, 0.92)
DENSITY_MATRIX_BOX_ASPECT = (1, 1, 0.65)
DENSITY_MATRIX_ZOOM = 1.2
DENSITY_MATRIX_BAR_WIDTH = 0.7
DENSITY_MATRIX_FONT_SIZE = SMALL_FONT_SIZE
DENSITY_MATRIX_XY_TICK_PAD = -5
DENSITY_MATRIX_FACE_SHADING = (0.95, 1.0, 1.0, 0.72, 1.0, 0.72)

# %%
rows, columns = (
    index.ravel() for index in np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
)
heights = result.density_matrix.real[rows, columns]
colors = LinearSegmentedColormap.from_list("density_matrix_blue", DENSITY_MATRIX_COLORS)
normalize = plt.Normalize(*DENSITY_MATRIX_ZLIM, clip=True)
bar_colors = colors(normalize(heights))
# Matplotlib face order: bottom, top, -y, +y, -x, +x.
face_colors = np.repeat(bar_colors[:, None, :], 6, axis=1)
face_colors[:, :, :3] *= np.array(DENSITY_MATRIX_FACE_SHADING)[None, :, None]
face_colors[:, 1, :3] = 0.8 * bar_colors[:, :3] + 0.2

fig, ax = plt.subplots(
    figsize=DENSITY_MATRIX_FIGSIZE,
    subplot_kw={"projection": "3d", "computed_zorder": False},
    layout="none",
)
ax.set_position(DENSITY_MATRIX_AXES_POSITION)
floor_tiles = [
    [
        (row + x, column + y, 0)
        for x, y in (
            (-0.08, -0.08),
            (DENSITY_MATRIX_BAR_WIDTH + 0.08, -0.08),
            (DENSITY_MATRIX_BAR_WIDTH + 0.08, DENSITY_MATRIX_BAR_WIDTH + 0.08),
            (-0.08, DENSITY_MATRIX_BAR_WIDTH + 0.08),
        )
    ]
    for row, column in zip(rows, columns)
]
ax.add_collection3d(
    Poly3DCollection(floor_tiles, facecolor=FLOOR_COLOR, alpha=0.6, zorder=2)
)
for mask, zorder in ((heights < 0, 1), (heights >= 0, 3)):
    ax.bar3d(
        rows[mask],
        columns[mask],
        np.zeros_like(heights[mask]),
        DENSITY_MATRIX_BAR_WIDTH,
        DENSITY_MATRIX_BAR_WIDTH,
        heights[mask],
        color=face_colors[mask].reshape(-1, 4),
        edgecolor=(1, 1, 1, 0.55),
        linewidth=0.075,
        shade=False,
        zorder=zorder,
    )
ax.set_zlim(*DENSITY_MATRIX_ZLIM)
ax.view_init(*DENSITY_MATRIX_VIEW_ANGLES)
ax.set_proj_type("persp", focal_length=DENSITY_MATRIX_FOCAL_LENGTH)
ax.set_box_aspect(DENSITY_MATRIX_BOX_ASPECT, zoom=DENSITY_MATRIX_ZOOM)
ax.set_xticks(
    np.arange(4) + DENSITY_MATRIX_BAR_WIDTH / 2,
    [rf"$\langle {label}∣$" for label in DENSITY_MATRIX_BASIS_LABELS],
)
ax.set_yticks(
    np.arange(4) + DENSITY_MATRIX_BAR_WIDTH / 2,
    [rf"$∣{label}\rangle$" for label in DENSITY_MATRIX_BASIS_LABELS],
)
ax.set_zticks([])
ax.grid(False)
ax.tick_params(axis="both", labelsize=DENSITY_MATRIX_FONT_SIZE, colors=SECONDARY_TEXT)
ax.tick_params(axis="x", pad=DENSITY_MATRIX_XY_TICK_PAD)
ax.tick_params(axis="y", pad=DENSITY_MATRIX_XY_TICK_PAD)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((1, 1, 1, 0))
    axis.line.set_visible(False)
    axis._axinfo["tick"].update(inward_factor=0, outward_factor=0)
for row, column in ((0, 0), (0, 3), (3, 0), (3, 3)):
    value = result.density_matrix.real[row, column]
    label_x, label_y, _ = proj3d.proj_transform(
        row + DENSITY_MATRIX_BAR_WIDTH / 2,
        column + DENSITY_MATRIX_BAR_WIDTH / 2,
        value,
        ax.get_proj(),
    )
    ax.annotate(
        f"{value:.2f}",
        (label_x, label_y),
        xytext=(6, 4) if (row, column) == (3, 0) else (0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=DENSITY_MATRIX_FONT_SIZE,
        zorder=10,
        annotation_clip=False,
    )
fidelity_error_percent = result.fidelity_error * 100

save_figure(fig, "figure_1_tomography")

with load_dataset(ION_SIV_FULL_TOMOGRAPHY_FIGURE_1, "rate.h5") as rate_ds:
    success_times = rate_ds.readout_timestamp - rate_ds.repetition_start_timestamp
    assert np.isfinite(success_times.values).all(), (
        "Tomography time-to-success data contain non-finite values"
    )
    rate_sample_count = success_times.size
    measured_rate_hz = 1 / success_times.mean().item()
    # All correlator settings share one continuous clock, so the starts in
    # correlator-major order are in time order.
    starts = rate_ds.repetition_start_timestamp.transpose(
        "correlator", "success_idx"
    ).values.ravel()
# Exponential waiting times have SEM(t) = mean(t)/sqrt(N). First-order
# propagation through R = 1/mean(t) gives sigma_R = R/sqrt(N). Statistical error only.
measured_rate_error_hz = measured_rate_hz / np.sqrt(rate_sample_count)

# The wall-clock rate also includes the fixed time between a readout and the
# next repetition start. Each interval between consecutive repetition starts is
# one full cycle, so R = (N - 1) / (t_start[N - 1] - t_start[0]) = 1 / mean(T).
# The error propagates the measured standard error of the mean cycle time T:
# sigma_R = R * std(T) / (mean(T) * sqrt(N - 1)). Statistical error only.
assert np.isfinite(starts).all() and np.all(np.diff(starts) > 0)
cycle_times = np.diff(starts)
wall_clock_rate_hz = 1 / cycle_times.mean()
wall_clock_rate_error_hz = (
    wall_clock_rate_hz
    * cycle_times.std(ddof=1)
    / (cycle_times.mean() * np.sqrt(cycle_times.size))
)

lines = ["Figure 1: Density matrix"]
for part, matrix in (
    ("Real part", result.density_matrix.real),
    ("Imaginary part", result.density_matrix.imag),
    ("Real-part standard deviations", result.real_error),
    ("Imaginary-part standard deviations", result.imaginary_error),
):
    lines.extend(
        [
            "",
            part,
            f"{'Bra / Ket':<12}"
            + "".join(
                f"{'|' + label + '>':>14}" for label in DENSITY_MATRIX_BASIS_LABELS
            ),
        ]
    )
    for label, row in zip(DENSITY_MATRIX_BASIS_LABELS, matrix):
        lines.append(
            f"{'<' + label + '|':<12}" + "".join(f"{value:14.8f}" for value in row)
        )
lines.extend(
    [
        "",
        f"Fidelity = {format_uncertainty(result.fidelity * 100, fidelity_error_percent)} %",
        f"Rate = {format_uncertainty(measured_rate_hz / 1000, measured_rate_error_hz / 1000)} kHz",
        f"Wall clock rate = {format_uncertainty(wall_clock_rate_hz / 1000, wall_clock_rate_error_hz / 1000)} kHz",
        f"Number of events: {rate_sample_count}",
    ]
)
lines.extend(["", "LaTeX"])
for part, matrix, errors in (
    ("Re", result.density_matrix.real, result.real_error),
    ("Im", result.density_matrix.imag, result.imaginary_error),
):
    lines.extend(
        [
            r"\[",
            r"\resizebox{\columnwidth}{!}{$",
            rf"\operatorname{{{part}}}[\rho] = 10^{{-3}}",
            r"\begin{pmatrix}",
        ]
    )
    for row, row_errors in zip(matrix, errors):
        entries = [
            format_uncertainty(1000 * value, 1000 * error) if error > 1e-12 else "0(0)"
            for value, error in zip(row, row_errors)
        ]
        lines.append(" & ".join(entries) + r" \\")
    lines.extend([r"\end{pmatrix}", r"$}", r"\]", ""])
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "figure_1_tomography.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
