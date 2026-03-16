# Telemetry-Based Server Selection in the Quantum Internet via Cross-Layer Runtime Estimation

This public snapshot packages the simulation code, notebook, and released CSV data used to reproduce the paper figures.

## Scope

This public snapshot includes source data and notebook workflows for:

- Fig. 3: three-parameter sweep of `T_exe`
- Fig. 4: `T_max` vs. `T_exe` scatter
- Fig. 5: even-regime regret
- Fig. 6: bottleneck-dominated regret
- Fig. 7: jitter robustness
- Fig. 8: PB-VQE operating map
- Fig. 9: Sobol dominant-factor map
- Appendix Fig. 11: full bottleneck curves

## Repository Layout

- `data/`: released CSV inputs used by the notebook and CLI workflows
- `sim/src/`: simulator source, plotting notebook, and local export location for regenerated figures
- `sim/src/data/_cache_rrci*`: cache directories created locally by notebook `compute` cells; these may be absent from the submission snapshot

## Environment

Tested environment:

- Python: `3.9.6`
- NetSquid: `1.1.7`
- NumPy: `1.25.2`
- pandas: `2.0.3`
- SciPy: `1.9.3`
- SALib: `1.5.1`
- matplotlib: `3.9.2`
- seaborn: `0.13.2`
- OS: `macOS 14.5` on `arm64`

This is sufficient for the notebook-based workflows documented in this README. It is not intended to be a full lockfile for every auxiliary script under `sim/src/`.

## Quick Start

### A. Replot figures that read released CSVs directly

Use this path if you only want to redraw figures that consume the released CSVs directly, without rebuilding notebook caches.
This path does not require NetSquid.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab sim/src/experiment.ipynb
```

Run the notebook cells for:

- Fig. 3: `[Paper Fig. 3] visualize (Texe 3-parameter sweep)`
- Fig. 4: `[Paper Fig. 4] visualize (Tmax vs Texe scatter)`
- Fig. 8: `[Paper Fig. 8] visualize (PB-VQE operating map)`
- Fig. 9: `[Paper Fig. 9] visualize (time_density single CSV)`

If the submission snapshot omits pre-rendered PDFs/SVGs and notebook cache CSVs, that is expected. This path regenerates the supported figures locally from the released CSV inputs.

### B. Rebuild notebook caches or rerun simulations

Use this path if you want to reproduce cache-backed plots or regenerate raw simulation outputs.
For cache rebuilds from the released CSVs, `requirements.txt` is sufficient.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `sim/src/experiment.ipynb` and run the `compute` cells for Fig. 5, Fig. 6 / Appendix Fig. 11, and Fig. 7 before the corresponding `visualize` cells.

If you want to rerun raw simulations from scratch, install NetSquid in the same environment:

```bash
pip install netsquid==1.1.7
```

If your NetSquid installation is provided through a licensed or mirrored channel, install the same version there.

Then use one of these routes:

- raw data regeneration: run `sim/src/density_random.py` and `sim/src/run_sense_time_density.py`
- weight recomputation: run `sim/src/run_global_time_sobol.py`

## Experiment Settings

### Even Regime

- `distance ~ loguniform(100, 1000)` km
- `entanglement_speed_factor ~ loguniform(200, 2000)` Hz
- `gate_speed_factor ~ loguniform(0.263, 2.63)`
- `shots = 10`

### Bottleneck-Dominated Regimes

- `Tcc` bottleneck:
  - `distance ~ loguniform(1000, 10000)` km
  - `entanglement_speed_factor ~ loguniform(200, 2000)` Hz
  - `gate_speed_factor ~ loguniform(0.263, 2.63)`
  - `shots = 10`
- `Tent` bottleneck:
  - `distance ~ loguniform(100, 1000)` km
  - `entanglement_speed_factor ~ loguniform(20, 200)` Hz
  - `gate_speed_factor ~ loguniform(0.263, 2.63)`
  - `shots = 10`
- `Tsrv` bottleneck:
  - `distance ~ loguniform(100, 1000)` km
  - `entanglement_speed_factor ~ loguniform(200, 2000)` Hz
  - `gate_speed_factor ~ loguniform(2.63, 26.3)`
  - `shots = 10`

### Jitter Robustness

- base ranges are the same as the even regime
- `shots = 10` and `shots = 1`
- low jitter: `cc_jitter_mean_ns = 72000`
- high jitter: `cc_jitter_mean_ns = 2900000`

### Common Study Settings

- candidate pool size: `10000`
- candidate subset size: `M = 1..100`
- sampled instances per `M`: `1500`
- bootstrap samples: `1500`
- seed stride: `997`

## Reproducing Results

The released snapshot supports two levels of reproduction:

- replot directly from released CSVs: use Quick Start A for Fig. 3, Fig. 4, Fig. 8, and Fig. 9
- rebuild notebook caches or rerun raw simulations: use Quick Start B for Fig. 5, Fig. 6, Fig. 7, Appendix Fig. 11, and full reruns

Pre-rendered figure files and notebook cache CSVs are treated as derived outputs and may be omitted from the submission snapshot.

## Data Artifacts

Released data in this snapshot is organized as follows:

- random candidate pools and per-run metrics:
  - `data/dr_even_*.csv`
  - `data/dr_j72k*.csv`
  - `data/dr_j2p9m*.csv`

- Sobol dominant-factor input data:
  - `data/time_density_*.csv`


This public snapshot does not bundle every intermediate file produced by exploratory scripts. Notebook caches under `sim/src/data/_cache_rrci*` and exported PDF/SVG figures are derived outputs that can be regenerated locally.

## Weighted-Sum Calibration

The weighted selector uses fixed constants in `sim/src/experiment.ipynb`.
The values below are released example constants used by the notebook panels in this snapshot, not universal weights for every regime or downstream reuse.

- `shots=10`: `w_ent=0.03876629357684344`, `w_cc=0.43750676409161815`, `w_srv=0.5237269423315385`
- `shots=1`: `w_ent=0.05503439277181065`, `w_cc=0.43988882263334483`, `w_srv=0.5050767845948445`

The notebook also contains bottleneck-specific constants for the bottleneck panels. Those are panel-specific examples as well.

To recompute the even-regime weights, run:

```bash
cd sim/src
python run_global_time_sobol.py \
  --gate_mode ionq_aria_factor \
  --dist_range 100 1000 \
  --ent_speed_range 200 2000 \
  --factor_range 0.263 2.63 \
  --shots 10 \
  --num_runs 1 \
  --base_seed 42 \
  --N 128 \
  --output_dir ../../data/weights/even10

python run_global_time_sobol.py \
  --gate_mode ionq_aria_factor \
  --dist_range 100 1000 \
  --ent_speed_range 200 2000 \
  --factor_range 0.263 2.63 \
  --shots 1 \
  --num_runs 1 \
  --base_seed 42 \
  --N 128 \
  --output_dir ../../data/weights/even1
```

`run_global_time_sobol.py` prints `Weights: {...}` and also writes normalized values to `*_sobol_indices.csv` as `ST_norm`.

## Limitations

- `Tcli` is fixed rather than sampled
- `distance`, `entanglement_speed_factor`, and `gate_speed_factor` are sampled independently
- the main NetSquid simulation loop does not include explicit queueing
- final figures are exported from a notebook rather than a standalone plotting CLI

## License

Unless otherwise noted, code in this repository is licensed under the MIT License. See LICENSE.
Released non-code artifacts in `data/` are also provided under the MIT License.
