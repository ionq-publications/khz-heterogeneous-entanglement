# Quantum Networking at the Speed of Quantum Computation: Kilohertz Entanglement in a Heterogeneous Quantum System

**Authors**: Lukas Hartung\*, Pierre Barral\*, Noah Glachman\*, George Toh\*, Jacob H. Davidson\*, Sagnik Saha\*, Alexander Chuang\*, Matthew C. Cambria\*, Madison Sutula\*, Alexander Abulnaga, Julia Brevoord, John Chiaverini, Ian Counts, Aos Dabbagh, Christian Dangel, Skylar Deckoff-Jones, Chawina De-Eknamkul, Jonathan Dietz, Riley Forst, Prithvi Gundlapalli, Jeonghoon Ha, Michael Haas, Ezekiel Meulbroek, Andrea Mucchietto, Daniel Riedel, Jonah Sachs, Jeremy Sage, Mikhail Shalaev, Harriet Shi, Denis Sukachev, Yichao Yu, Christopher Monroe, Nicholas Mondrik, Mihir Bhaskar, Bart Machielse, Matteo Pompili, Carsten Robens and David Levonian

\* These authors contributed equally to this work, and are listed in a random order.


This repository contains the data and analysis notebooks to reproduce the figures in the paper.

**arXiv**: [2610.10705](https://arxiv.org/abs/2610.10705)

## Setup

Requires [uv](https://docs.astral.sh/uv/). 

## Generating all figures and text exports (raw data)
From the root of this repository, run:

```bash
uv sync
uv run --locked python -m papertools.generate_all_figures
```

## License

Copyright (c) 2026 IonQ, Inc. Licensed under the [Apache License, Version 2.0](LICENSE).
