"""
Global Sobol sensitivity analysis for total execution time.

This is a simplified, single-pass Sobol run over the full parameter ranges
without segmenting. It reuses the density-based simulator and exports both
sampled metrics and Sobol indices.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample.sobol import sample as sobol_sample

from qs.density import DEFAULT_CONFIG, DEFAULT_THETA, SimulationConfig, single_angle_metrics

Number = float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_float(v: Number) -> str:
    v = float(v)
    s = f"{v:.0e}"
    if "e" not in s:
        return s.replace(".", "_")
    base, exp = s.split("e")
    base = base.rstrip(".0") or "0"
    sign = ""
    if exp.startswith("-"):
        sign = "-"
        exp = exp[1:]
    elif exp.startswith("+"):
        exp = exp[1:]
    exp = exp.lstrip("0") or "0"
    return f"{base}e{sign}{exp}"


def _fmt_range(r: Tuple[Number, Number]) -> str:
    lo, hi = r
    return f"{_fmt_float(lo)}-{_fmt_float(hi)}"


def _build_output_name(
    *,
    gate_mode: str,
    dist_range: Tuple[Number, Number],
    ent_speed_range: Tuple[Number, Number],
    gate_range: Tuple[Number, Number],
    gate_label: str,
    client_range: Tuple[Number, Number] | None,
    N: int,
) -> str:
    parts = [
        "global_sobol",
        gate_mode,
        f"dist-{_fmt_range(dist_range)}",
        f"ent-{_fmt_range(ent_speed_range)}",
        f"{gate_label}-{_fmt_range(gate_range)}",
        f"N{int(N)}",
    ]
    if client_range is not None:
        parts.insert(-1, f"client-{_fmt_range(client_range)}")
    return "_".join(parts)


def compute_time_weights(Si: Dict[str, np.ndarray], names: Iterable[str]) -> Dict[str, float]:
    """Normalize ST to use later as weights."""
    name_list = list(names)
    st = np.array(Si.get("ST", []), dtype=float)
    total = np.sum(st)
    if not np.isfinite(total) or total <= 0:
        return {"w_ent": 0.0, "w_cc": 0.0, "w_srv": 0.0, "w_cli": 0.0}
    norm = st / total
    idx = {name: i for i, name in enumerate(name_list)}
    w_cli = 0.0
    if "client_gate_speed_factor" in idx and idx["client_gate_speed_factor"] < len(norm):
        w_cli = float(norm[idx["client_gate_speed_factor"]])
    return {
        "w_ent": float(norm[idx.get("entanglement_speed_factor", 0)]),
        "w_cc": float(norm[idx.get("distance", 0)]),
        "w_srv": float(norm[idx.get("gate_time_ns", idx.get("gate_speed_factor", 0))]),
        "w_cli": w_cli,
    }


def run_global_time_sobol(
    *,
    dist_range: Tuple[Number, Number] = (30.0, 3000.0),
    ent_speed_range: Tuple[Number, Number] = (3.0, 300.0),
    gate_time_ns_range: Tuple[Number, Number] = (1e4, 1e6),
    gate_speed_range: Tuple[Number, Number] = (1e0, 1e2),
    factor_range: Tuple[Number, Number] = (0.5, 2.0),
    gate_mode: str = "absolute",  
    client_gate_speed_range: Tuple[Number, Number] | None = None,
    client_independent: bool = False,
    N: int = 256,
    theta: float = DEFAULT_THETA,
    output_dir: str = "data",
    num_runs: int = DEFAULT_CONFIG.num_runs,
    shots: int = DEFAULT_CONFIG.shots,
    base_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    _ensure_dir(output_dir)


    client_independent = False
    client_gate_speed_range = (1.0, 1.0)

    if gate_mode == "fixed":
        param_names = ["distance", "entanglement_speed_factor", "gate_speed_factor"]
        bounds = [
            [np.log10(dist_range[0]), np.log10(dist_range[1])],
            [np.log10(ent_speed_range[0]), np.log10(ent_speed_range[1])],
            [np.log10(gate_speed_range[0]), np.log10(gate_speed_range[1])],
        ]
        timing_mode = "factor"
        gate_label = "gs"
    elif gate_mode == "ionq_aria_factor":
        param_names = ["distance", "entanglement_speed_factor", "gate_speed_factor"]
        bounds = [
            [np.log10(dist_range[0]), np.log10(dist_range[1])],
            [np.log10(ent_speed_range[0]), np.log10(ent_speed_range[1])],
            [np.log10(factor_range[0]), np.log10(factor_range[1])],
        ]
        timing_mode = "factor"
        gate_label = "ariafactor"
    else:
        param_names = ["distance", "entanglement_speed_factor", "gate_time_ns"]
        bounds = [
            [np.log10(dist_range[0]), np.log10(dist_range[1])],
            [np.log10(ent_speed_range[0]), np.log10(ent_speed_range[1])],
            [np.log10(gate_time_ns_range[0]), np.log10(gate_time_ns_range[1])],
        ]
        timing_mode = "absolute"
        gate_label = "gt"

    client_range = None

    base_config: SimulationConfig = replace(
        DEFAULT_CONFIG,
        num_runs=num_runs,
        shots=shots,
    )

    problem = {"num_vars": len(param_names), "names": param_names, "bounds": bounds}
    X_log = sobol_sample(problem, N, calc_second_order=True)
    X = 10 ** X_log

    rows = []
    total_times = []

    for idx, row in enumerate(X, start=1):
        distance, ent_speed, gate_var, *client_var_opt = row

        if gate_mode in ("fixed", "ionq_aria_factor"):
            client_factor = 1.0
            cfg = replace(
                base_config,
                distance=float(distance),
                entanglement_speed_factor=float(ent_speed),
                gate_speed_factor=float(gate_var),
                client_gate_speed_factor=client_factor,
                timing_mode="factor",
                gate_time_ns=None,
            )
        else:
            cfg = replace(
                base_config,
                distance=float(distance),
                entanglement_speed_factor=float(ent_speed),
                gate_speed_factor=1.0,
                client_gate_speed_factor=1.0,
                timing_mode="absolute",
                gate_time_ns=float(gate_var),
            )

        _, metrics = single_angle_metrics(
            theta=float(theta),
            config=cfg,
            seed=base_seed + idx,
        )
        total_times.append(metrics["total_time"])
        rows.append(
            {
                "distance": float(distance),
                "entanglement_speed_factor": float(ent_speed),
                "gate_speed_factor" if gate_mode in ("fixed", "ionq_aria_factor") else "gate_time_ns": float(gate_var),
                **(
                    {"client_gate_speed_factor": client_factor}
                    if gate_mode in ("fixed", "ionq_aria_factor") and client_independent
                    else {}
                ),
                "total_time": float(metrics["total_time"]),
            }
        )

    total_array = np.array(total_times)
    Si = sobol.analyze(problem, total_array, calc_second_order=True, print_to_console=False)

    df = pd.DataFrame(rows)

    out_name = _build_output_name(
        gate_mode=gate_mode,
        dist_range=dist_range,
        ent_speed_range=ent_speed_range,
        gate_range=(
            gate_speed_range
            if gate_mode == "fixed"
            else factor_range
            if gate_mode == "ionq_aria_factor"
            else gate_time_ns_range
        ),
        gate_label=gate_label,
        client_range=client_gate_speed_range if client_independent and gate_mode in ("fixed", "ionq_aria_factor") else None,
        N=N,
    )
    sample_path = os.path.join(output_dir, f"{out_name}.csv")
    df.to_csv(sample_path, index=False)

    sobol_df = pd.DataFrame(
        {
            "name": param_names,
            "S1": Si["S1"],
            "ST": Si["ST"],
        }
    )
    sobol_df["ST_norm"] = sobol_df["ST"] / sobol_df["ST"].sum() if sobol_df["ST"].sum() != 0 else 0.0
    sobol_path = os.path.join(output_dir, f"{out_name}_sobol_indices.csv")
    sobol_df.to_csv(sobol_path, index=False)

    return df, Si


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single global Sobol analysis for time.")
    parser.add_argument("--gate_mode", choices=["absolute", "fixed", "ionq_aria_factor"], default="absolute")
    parser.add_argument("--dist_range", nargs=2, type=float, default=[30.0, 3000.0])
    parser.add_argument("--ent_speed_range", nargs=2, type=float, default=[3.0, 300.0])
    parser.add_argument(
        "--gate_time_ns_range",
        nargs=2,
        type=float,
        default=[1e4, 1e6],
        help="Absolute gate time range in ns (only when gate_mode=absolute).",
    )
    parser.add_argument(
        "--gate_speed_range",
        nargs=2,
        type=float,
        default=[1e0, 1e2],
        help="Factor range (gate/measure slower when larger) for gate_mode=fixed.",
    )
    parser.add_argument(
        "--factor_range",
        nargs=2,
        type=float,
        default=[0.5, 2.0],
        help="Factor range (gate/measure slower when larger) for gate_mode=ionq_aria_factor.",
    )
    parser.add_argument(
        "--client_gate_speed_range",
        nargs=2,
        type=float,
        help="Range for client_gate_speed_factor when --client_independent is set (defaults to server range).",
    )
    parser.add_argument(
        "--client_independent",
        action="store_true",
        help="Sample client_gate_speed_factor independently (factor/aria modes only).",
    )
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_runs", type=int, default=DEFAULT_CONFIG.num_runs)
    parser.add_argument("--shots", type=int, default=DEFAULT_CONFIG.shots)
    parser.add_argument("--base_seed", type=int, default=42)

    args = parser.parse_args()

    df, Si = run_global_time_sobol(
        gate_mode=args.gate_mode,
        dist_range=tuple(args.dist_range),
        ent_speed_range=tuple(args.ent_speed_range),
        gate_time_ns_range=tuple(args.gate_time_ns_range),
        gate_speed_range=tuple(args.gate_speed_range),
        factor_range=tuple(args.factor_range),
        client_gate_speed_range=tuple(args.client_gate_speed_range) if args.client_gate_speed_range else None,
        client_independent=args.client_independent,
        N=args.N,
        theta=args.theta,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        shots=args.shots,
        base_seed=args.base_seed,
    )

    name_srv = "gate_speed_factor" if args.gate_mode in ("fixed", "ionq_aria_factor") else "gate_time_ns"
    names = ["distance", "entanglement_speed_factor", name_srv]
    if args.client_independent and args.gate_mode in ("fixed", "ionq_aria_factor"):
        names.append("client_gate_speed_factor")
    weights = compute_time_weights(Si, names)
    print("Saved Sobol samples and indices to", args.output_dir)
    print("Weights:", weights)
