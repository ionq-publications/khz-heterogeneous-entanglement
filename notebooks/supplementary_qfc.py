"""Plot the quantum frequency conversion efficiency and transmitted pump power.

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
# # Supplementary quantum frequency conversion

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lmfit import Model

from papertools import (
    BAR_EDGE,
    BAR_FILL,
    FIGURES_DIR,
    IONQ_ORANGE,
    IONQ_ORANGE_EDGE,
    MM,
    SINGLE_COLUMN_WIDTH_MM,
    apply_publication_style,
    load_dataset,
    save_figure,
)
from papertools.formatting import format_uncertainty
from papertools.run_ids import (
    QFC_ACW_NOISE,
    QFC_ACW_SIGNAL,
    QFC_CW_NOISE,
    QFC_CW_SIGNAL,
)

np.random.seed(list(b"IonQ-R&D"))
apply_publication_style()

# %% [markdown]
# ## Signal and noise power sweeps

# %%
qfc_data = {}
for direction, signal_run, noise_run in (
    ("CW", QFC_CW_SIGNAL, QFC_CW_NOISE),
    ("ACW", QFC_ACW_SIGNAL, QFC_ACW_NOISE),
):
    signal = load_dataset(signal_run, "dataset_0.h5")
    noise = load_dataset(noise_run, "dataset_0.h5")
    assert signal.ppower.equals(noise.ppower), (
        f"{direction}: signal and noise pump settings differ"
    )
    assert np.isfinite(signal.outputpower.values).all(), (
        f"Run {signal_run}: non-finite output power"
    )
    assert np.isfinite(noise.outputpower.values).all(), (
        f"Run {noise_run}: non-finite output power"
    )
    qfc_data[direction] = xr.Dataset(
        {
            "signal_power": signal.outputpower,
            "noise_power": noise.outputpower,
            "net_output_power": signal.outputpower - noise.outputpower,
        }
    )
    qfc_data[direction].net_output_power.attrs.update(
        long_name="Noise-subtracted output power",
        units="W",
    )
    qfc_data[direction].attrs.update(signal_run_id=signal_run, noise_run_id=noise_run)

# %% [markdown]
# ## Conversion efficiency and transmitted pump power

# %%
QFC_INPUT_SIGNAL_POWER_W = 100e-6
QFC_INPUT_WAVELENGTH_NM = 493
QFC_OUTPUT_WAVELENGTH_NM = 737
QFC_TRANSMITTED_PUMP_RANGE_W = (0.012, 0.260)


def conversion_efficiency(
    pump_power_w: np.ndarray, maximum_efficiency: float, coupling_per_w: float
) -> np.ndarray:
    """Return eta_max * sin(sqrt(kappa * P))**2 for pump power P in watts.

    The coupling coefficient kappa has units of 1/W. It combines the normalized
    efficiency and squared interaction length, which cannot be fitted separately.
    """
    return maximum_efficiency * np.sin(np.sqrt(coupling_per_w * pump_power_w)) ** 2


qfc_fits = {}
qfc_model = Model(conversion_efficiency)
for direction, ds in qfc_data.items():
    pump_setting = ds.ppower.values
    pump_low, pump_high = QFC_TRANSMITTED_PUMP_RANGE_W
    pump_power_w = pump_low + (pump_setting - pump_setting.min()) * (
        (pump_high - pump_low) / (pump_setting.max() - pump_setting.min())
    )
    efficiency = ds.net_output_power.values / (
        QFC_INPUT_SIGNAL_POWER_W * QFC_INPUT_WAVELENGTH_NM / QFC_OUTPUT_WAVELENGTH_NM
    )
    ds["transmitted_pump_power"] = ("ppower", pump_power_w, {"units": "W"})
    ds["efficiency"] = ("ppower", efficiency, {"units": ""})
    initial_coupling = np.pi**2 / (4 * pump_power_w[np.argmax(efficiency)])
    parameters = qfc_model.make_params(
        maximum_efficiency={"value": efficiency.max(), "min": 0, "max": 1},
        coupling_per_w={"value": initial_coupling, "min": 0},
    )
    fit = qfc_model.fit(efficiency, parameters, pump_power_w=pump_power_w)
    assert fit.success, fit.message
    assert fit.covar is not None, f"{direction}: fit covariance is unavailable"
    qfc_fits[direction] = fit

# %%
QFC_FIGSIZE = (SINGLE_COLUMN_WIDTH_MM * MM, SINGLE_COLUMN_WIDTH_MM * MM * 5 / 6.5)
QFC_MARKER_SIZE = 3
QFC_MARKER_EDGE_WIDTH = 0.6
QFC_FIT_LINEWIDTH = 0.8
QFC_FIT_POINTS = 100
QFC_COLORS = (("CW", BAR_FILL, BAR_EDGE), ("ACW", IONQ_ORANGE, IONQ_ORANGE_EDGE))

# %%
fit_pump_power_w = np.linspace(*QFC_TRANSMITTED_PUMP_RANGE_W, QFC_FIT_POINTS)
fig, ax = plt.subplots(figsize=QFC_FIGSIZE)
for direction, fill_color, edge_color in QFC_COLORS:
    ds = qfc_data[direction]
    ax.plot(
        ds.transmitted_pump_power.values * 1000,
        ds.efficiency.values,
        marker="o",
        linestyle="none",
        markersize=QFC_MARKER_SIZE,
        color=fill_color,
        markeredgecolor=edge_color,
        markeredgewidth=QFC_MARKER_EDGE_WIDTH,
        label=f"{direction} efficiency",
        zorder=3,
    )
    ax.plot(
        fit_pump_power_w * 1000,
        qfc_fits[direction].eval(pump_power_w=fit_pump_power_w),
        color=edge_color,
        linewidth=QFC_FIT_LINEWIDTH,
        label=f"{direction} fit",
        zorder=2,
    )
ax.set_xlabel("Transmitted pump power (mW)")
ax.set_ylabel("Conversion efficiency")
ax.set_xlim(left=0)
ax.legend(loc="lower right", frameon=False)
save_figure(fig, "supplementary_qfc_conversion_efficiency")

lines = ["Supplementary QFC conversion efficiency"]
for direction, ds in qfc_data.items():
    fit = qfc_fits[direction]
    maximum_efficiency = fit.params["maximum_efficiency"]
    coupling = fit.params["coupling_per_w"]
    lines.extend(
        [
            "",
            direction,
            f"Maximum efficiency: {format_uncertainty(maximum_efficiency.value, maximum_efficiency.stderr)}",
            f"Coupling kappa: {format_uncertainty(coupling.value, coupling.stderr)} 1/W",
            "",
            f"{'Pump setting (W)':>18} {'Pump transmitted (mW)':>23} {'Signal (microW)':>18} "
            f"{'Noise (microW)':>18} {'Net output (microW)':>21} {'Efficiency':>14}",
        ]
    )
    for setting, power, signal, noise, net, efficiency in zip(
        ds.ppower.values,
        ds.transmitted_pump_power.values,
        ds.signal_power.values,
        ds.noise_power.values,
        ds.net_output_power.values,
        ds.efficiency.values,
    ):
        lines.append(
            f"{setting:18.8f} {power * 1000:23.8f} {signal * 1e6:18.8f} "
            f"{noise * 1e6:18.8f} {net * 1e6:21.8f} {efficiency:14.8f}"
        )
    lines.extend(["", f"{'Pump transmitted (mW)':>23} {'Fitted efficiency':>20}"])
    for power, efficiency in zip(
        fit_pump_power_w, fit.eval(pump_power_w=fit_pump_power_w)
    ):
        lines.append(f"{power * 1000:23.8f} {efficiency:20.8f}")
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_qfc_conversion_efficiency.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
