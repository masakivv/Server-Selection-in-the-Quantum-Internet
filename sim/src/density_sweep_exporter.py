from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qs.density import DEFAULT_CONFIG, DEFAULT_THETA, SimulationConfig, single_angle_metrics

EXCLUDED_DENSITY_KEYS = {
    "zz_rho",
    "xx_rho",
    "zz_expectations",
    "xx_expectations",
}

SWEEPABLE_PARAMS = ("distance", "gate_speed_factor", "entanglement_speed_factor")


def _logspace_inclusive(low: float, high: float, n: int) -> np.ndarray:
    if low <= 0 or high <= 0:
        raise ValueError(f"logspace bounds must be > 0 (got low={low}, high={high})")
    if n <= 1:
        return np.array([float(low)])
    return np.logspace(math.log10(low), math.log10(high), int(n))


def _predict_latency_ms(
    *,
    gate_speed_factor: float,
    distance_km: float,
    entanglement_rate_hz: float,
) -> Dict[str, float]:
    Tsrv = 9.48 * float(gate_speed_factor)
    Tcc = 0.025 * float(distance_km)
    Tent = 5000.0 / float(entanglement_rate_hz)
    Tmax = Tsrv + Tcc + Tent
    Tmin = max(Tsrv, Tcc, Tent)
    return {
        "Tsrv_est_ms": Tsrv,
        "Tcc_est_ms": Tcc,
        "Tent_est_ms": Tent,
        "Tmax_est_ms": Tmax,
        "Tmin_est_ms": Tmin,
    }


def _build_config(
    *,
    num_runs: int,
    shots: int,
    distance: float,
    gate_speed_factor: float,
    client_gate_speed_factor: float,
    entanglement_speed_factor: float,
    ext_stochastic: bool,
    ext_seed: Optional[int],
    ext_min_delay_ns: float,
) -> SimulationConfig:

    return SimulationConfig(
        num_runs=int(num_runs),
        shots=int(shots),
        dephase_rate=float(DEFAULT_CONFIG.dephase_rate),
        client_fidelity=float(DEFAULT_CONFIG.client_fidelity),
        distance=float(distance),
        T1=float(DEFAULT_CONFIG.T1),
        client_T1=float(DEFAULT_CONFIG.client_T1),
        T2_ratio=float(DEFAULT_CONFIG.T2_ratio),
        sge=float(DEFAULT_CONFIG.sge),
        dge=float(DEFAULT_CONFIG.dge),
        gate_speed_factor=float(gate_speed_factor),
        client_gate_speed_factor=float(client_gate_speed_factor),
        entanglement_fidelity=float(DEFAULT_CONFIG.entanglement_fidelity),
        entanglement_speed_factor=float(entanglement_speed_factor),
        cc_jitter_mean_ns=float(DEFAULT_CONFIG.cc_jitter_mean_ns),
        ext_stochastic=bool(ext_stochastic),
        ext_seed=ext_seed,
        ext_min_delay_ns=float(ext_min_delay_ns),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic sweeps for density-based simulations (CSV export).")

    p.add_argument("--output", type=str, default="data/density_sweep.csv", help="Output CSV path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")

    p.add_argument("--points", type=int, default=25, help="Number of sweep points per (anchor, param)")
    p.add_argument("--replicates", type=int, default=10, help="Number of independent seeds per sweep point")

    p.add_argument("--base-seed", type=int, default=42, help="Base seed")
    p.add_argument("--seed-stride", type=int, default=997, help="Seed stride")
    p.add_argument(
        "--classical-rng-seed-offset",
        type=int,
        default=100000,
        help="Offset for classical RNGs (rng_ab=seed+offset, rng_ba=seed+offset+1)",
    )

    p.add_argument("--theta", type=float, default=float(DEFAULT_THETA), help="Single angle theta")

    p.add_argument("--shots", type=int, default=10, help="shots per run")
    p.add_argument("--num-runs", type=int, default=int(DEFAULT_CONFIG.num_runs), help="num_runs in SimulationConfig")


    p.add_argument(
        "--client-gate-speed-factor",
        type=float,
        default=float(DEFAULT_CONFIG.client_gate_speed_factor),
        help="Client gate speed factor (fixed unless --client-coupled is set)",
    )
    p.add_argument(
        "--client-coupled",
        action="store_true",
        help="If set, client_gate_speed_factor := gate_speed_factor at each point",
    )


    p.add_argument("--even-gsf-min", type=float, default=0.263, help="Even gate_speed_factor min (≈ κ_min)")
    p.add_argument("--even-gsf-max", type=float, default=2.63, help="Even gate_speed_factor max (≈ κ_max)")
    p.add_argument("--even-dist-min", type=float, default=100.0, help="Even distance min [km]")
    p.add_argument("--even-dist-max", type=float, default=1000.0, help="Even distance max [km]")
    p.add_argument("--even-ent-rate-min", type=float, default=200.0, help="Even entanglement rate min [Hz]")
    p.add_argument("--even-ent-rate-max", type=float, default=2000.0, help="Even entanglement rate max [Hz]")


    p.add_argument(
        "--sweep-params",
        type=str,
        default="distance,gate_speed_factor,entanglement_speed_factor",
        help=f"Comma-separated sweep params from {SWEEPABLE_PARAMS}",
    )
    p.add_argument(
        "--anchors",
        type=str,
        default="light,heavy",
        help="Comma-separated anchors: light,heavy (default: light,heavy)",
    )
    p.add_argument(
        "--ent-sweep-mode",
        type=str,
        choices=("time", "rate"),
        default="time",
        help=(
            "How to define entanglement_speed_factor sweep range. "
            "'time': sweep Tent multiplicatively -> R in [R_min/10, 10*R_max]. "
            "'rate': naive sweep R in [0.1*R_light, 10*R_heavy] (often collapses to Even range)."
        ),
    )


    p.add_argument("--ext-stochastic", action="store_true", help="Use stochastic external entanglement triggering")
    p.add_argument("--ext-seed", type=int, default=None, help="External entanglement RNG seed")
    p.add_argument(
        "--ext-min-delay-ns",
        type=float,
        default=float(DEFAULT_CONFIG.ext_min_delay_ns),
        help="Min delay [ns] for external entanglement exponential samples",
    )

    return p


def _parse_csv_list(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def run_sweeps(args: argparse.Namespace) -> pd.DataFrame:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    sweep_params = _parse_csv_list(args.sweep_params)
    for sp in sweep_params:
        if sp not in SWEEPABLE_PARAMS:
            raise ValueError(f"Unknown sweep param '{sp}'. Choose from {SWEEPABLE_PARAMS}.")

    anchors_req = _parse_csv_list(args.anchors)
    for a in anchors_req:
        if a not in ("light", "heavy"):
            raise ValueError("anchors must be a subset of {'light','heavy'}")


    gsf_min = float(args.even_gsf_min)
    gsf_max = float(args.even_gsf_max)
    dist_min = float(args.even_dist_min)
    dist_max = float(args.even_dist_max)
    r_min = float(args.even_ent_rate_min)
    r_max = float(args.even_ent_rate_max)


    anchor_points: Dict[str, Dict[str, float]] = {

        "light": {
            "gate_speed_factor": gsf_min,
            "distance": dist_min,
            "entanglement_speed_factor": r_max,
        },

        "heavy": {
            "gate_speed_factor": gsf_max,
            "distance": dist_max,
            "entanglement_speed_factor": r_min,
        },
    }


    sweep_ranges: Dict[str, Tuple[float, float]] = {}
    sweep_ranges["distance"] = (0.1 * dist_min, 10.0 * dist_max)
    sweep_ranges["gate_speed_factor"] = (0.1 * gsf_min, 10.0 * gsf_max)

    if args.ent_sweep_mode == "time":


        sweep_ranges["entanglement_speed_factor"] = (r_min / 10.0, 10.0 * r_max)
    else:

        sweep_ranges["entanglement_speed_factor"] = (0.1 * r_max, 10.0 * r_min)


    sweep_values_map: Dict[str, np.ndarray] = {}
    for sp in sweep_params:
        low, high = sweep_ranges[sp]
        sweep_values_map[sp] = _logspace_inclusive(low, high, int(args.points))

    records: List[Dict[str, float]] = []


    run_counter = 0

    for anchor_name in anchors_req:
        base_params = anchor_points[anchor_name]


        for rep in range(int(args.replicates)):
            seed = int(args.base_seed + run_counter * args.seed_stride)
            run_counter += 1

            rng_ab = np.random.RandomState(seed + int(args.classical_rng_seed_offset))
            rng_ba = np.random.RandomState(seed + int(args.classical_rng_seed_offset) + 1)

            gsf = float(base_params["gate_speed_factor"])
            dist = float(base_params["distance"])
            ent_rate = float(base_params["entanglement_speed_factor"])
            cli_gsf = gsf if args.client_coupled else float(args.client_gate_speed_factor)

            config = _build_config(
                num_runs=int(args.num_runs),
                shots=int(args.shots),
                distance=dist,
                gate_speed_factor=gsf,
                client_gate_speed_factor=cli_gsf,
                entanglement_speed_factor=ent_rate,
                ext_stochastic=bool(args.ext_stochastic),
                ext_seed=args.ext_seed,
                ext_min_delay_ns=float(args.ext_min_delay_ns),
            )

            energy, metrics = single_angle_metrics(
                float(args.theta),
                config,
                seed,
                rng_ab=rng_ab,
                rng_ba=rng_ba,
            )

            filtered_metrics = {k: v for k, v in metrics.items() if k not in EXCLUDED_DENSITY_KEYS}
            filtered_metrics.pop("energy", None)

            record = {
                "experiment_kind": "anchor",
                "experiment_anchor": anchor_name,
                "experiment_sweep_param": "",
                "sweep_index": -1,
                "sweep_value": float("nan"),
                "replicate_index": rep,
                "seed": seed,
                "theta": float(args.theta),
                "num_runs": int(args.num_runs),
                "shots": int(args.shots),
                "distance": dist,
                "gate_speed_factor": gsf,
                "client_gate_speed_factor": cli_gsf,
                "entanglement_speed_factor": ent_rate,
                "ent_sweep_mode": str(args.ent_sweep_mode),
                **_predict_latency_ms(
                    gate_speed_factor=gsf,
                    distance_km=dist,
                    entanglement_rate_hz=ent_rate,
                ),
                **filtered_metrics,
                "energy": energy,
            }
            records.append(record)


        for sp in sweep_params:
            values = sweep_values_map[sp]
            low, high = sweep_ranges[sp]

            for i, v in enumerate(values):
                params = dict(base_params)
                params[sp] = float(v)

                gsf = float(params["gate_speed_factor"])
                dist = float(params["distance"])
                ent_rate = float(params["entanglement_speed_factor"])
                cli_gsf = gsf if args.client_coupled else float(args.client_gate_speed_factor)

                for rep in range(int(args.replicates)):
                    seed = int(args.base_seed + run_counter * args.seed_stride)
                    run_counter += 1

                    rng_ab = np.random.RandomState(seed + int(args.classical_rng_seed_offset))
                    rng_ba = np.random.RandomState(seed + int(args.classical_rng_seed_offset) + 1)

                    config = _build_config(
                        num_runs=int(args.num_runs),
                        shots=int(args.shots),
                        distance=dist,
                        gate_speed_factor=gsf,
                        client_gate_speed_factor=cli_gsf,
                        entanglement_speed_factor=ent_rate,
                        ext_stochastic=bool(args.ext_stochastic),
                        ext_seed=args.ext_seed,
                        ext_min_delay_ns=float(args.ext_min_delay_ns),
                    )

                    try:
                        energy, metrics = single_angle_metrics(
                            float(args.theta),
                            config,
                            seed,
                            rng_ab=rng_ab,
                            rng_ba=rng_ba,
                        )
                    except OverflowError:
                        debug_params = {
                            "anchor": anchor_name,
                            "sweep_param": sp,
                            "sweep_value": float(v),
                            "distance": dist,
                            "gate_speed_factor": gsf,
                            "entanglement_speed_factor": ent_rate,
                            "client_gate_speed_factor": cli_gsf,
                            "theta": float(args.theta),
                        }
                        print(f"OverflowError at seed={seed} params={debug_params}")
                        raise

                    filtered_metrics = {k: v for k, v in metrics.items() if k not in EXCLUDED_DENSITY_KEYS}
                    filtered_metrics.pop("energy", None)

                    record = {
                        "experiment_kind": "sweep",
                        "experiment_anchor": anchor_name,
                        "experiment_sweep_param": sp,
                        "sweep_index": int(i),
                        "sweep_value": float(v),
                        "sweep_low": float(low),
                        "sweep_high": float(high),
                        "replicate_index": rep,
                        "seed": seed,
                        "theta": float(args.theta),
                        "num_runs": int(args.num_runs),
                        "shots": int(args.shots),
                        "distance": dist,
                        "gate_speed_factor": gsf,
                        "client_gate_speed_factor": cli_gsf,
                        "entanglement_speed_factor": ent_rate,
                        "ent_sweep_mode": str(args.ent_sweep_mode),
                        **_predict_latency_ms(
                            gate_speed_factor=gsf,
                            distance_km=dist,
                            entanglement_rate_hz=ent_rate,
                        ),
                        **filtered_metrics,
                        "energy": energy,
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


def main() -> None:
    args = build_parser().parse_args()
    run_sweeps(args)


if __name__ == "__main__":
    main()
