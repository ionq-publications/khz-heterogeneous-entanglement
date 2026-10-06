"""Export figures with matching PDF and PNG layouts.

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

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

FIGURES_DIR = Path(__file__).resolve().parents[2] / "output"
"""Repository output directory, independent of the working directory."""


def save_figure(fig: Figure, name: str) -> None:
    """Save a filename stem to output/pdf and output/png at the canvas size.

    Draw once and freeze the layout before exporting. PDF text stays editable;
    PNG uses 300 dpi. The figure stays open for further notebook adjustments.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    with matplotlib.rc_context({"savefig.bbox": None, "pdf.fonttype": 42}):
        for extension in ("pdf", "png"):
            directory = FIGURES_DIR / extension
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(directory / f"{name}.{extension}", dpi=300)
