"""Random sampling exporter for density-based simulations."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from qs.density import (
    DEFAULT_CONFIG,
    DEFAULT_THETA,
    SimulationConfig,
    single_angle_metrics,
)

EXCLUDED_DENSITY_KEYS = {
    "zz_rho",
    "xx_rho",
    "zz_expectations",
    "xx_expectations",
}

CONFIG_PARAM_TYPES: Mapping[str, type] = {
    "num_runs": int,
    "shots": int,
    "dephase_rate": float,
    "client_fidelity": float,
    "distance": float,
    "T1": float,
    "client_T1": float,
    "T2_ratio": float,
    "sge": float,
    "dge": float,
    "gate_speed_factor": float,
    "client_gate_speed_factor": float,
    "entanglement_fidelity": float,
    "entanglement_speed_factor": float,
}

ALL_PARAM_TYPES: Mapping[str, type] = {
    **CONFIG_PARAM_TYPES,
    "theta": float,
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    mode: str
    values: Tuple[float, ...]
    dtype: type

    def sample(self, rng: random.Random) -> float:
        if self.mode == "constant":
            value = self.values[0]
        elif self.mode == "uniform":
            low, high = self.values
            value = rng.uniform(low, high)
        elif self.mode == "loguniform":
            low, high = self.values
            if low <= 0 or high <= 0:
                raise ValueError(f"{self.name}: loguniform bounds must be > 0 (got {low}, {high})")
            log_low = math.log(low)
            log_high = math.log(high)
            value = math.exp(rng.uniform(log_low, log_high))
        elif self.mode == "choice":
            value = rng.choice(self.values)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}' for {self.name}")

        if self.dtype is int:
            return int(round(value))
        return float(value)

    def describe(self) -> str:
        if self.mode == "constant":
            return f"constant({self.values[0]})"
        formatted = ",".join(str(v) for v in self.values)
        if self.mode in {"uniform", "loguniform", "choice"}:
            return f"{self.mode}({formatted})"
        return f"{self.mode}:{formatted}"


def _convert_token(token: str, dtype: type) -> float:
    token = token.strip()
    if not token:
        raise ValueError("Empty value in parameter specification")
    if dtype is int:
        return float(int(float(token)))
    return float(token)


def _parse_value_list(raw: str, dtype: type) -> Tuple[float, ...]:
    items = [item for item in raw.split(";") if item.strip()]
    if len(items) == 1 and "," in items[0]:
        items = [item for item in items[0].split(",") if item.strip()]
    return tuple(_convert_token(item, dtype) for item in items)


def parse_parameter_spec(expr: str) -> ParameterSpec:
    if "=" not in expr:
        raise ValueError(f"Parameter specification must be NAME=VALUE, got '{expr}'")
    name, raw_expr = expr.split("=", 1)
    name = name.strip()
    raw_expr = raw_expr.strip()

    if name not in ALL_PARAM_TYPES:
        raise KeyError(f"Unknown parameter name '{name}'")

    dtype = ALL_PARAM_TYPES[name]

    def as_tuple(values: Iterable[float]) -> Tuple[float, ...]:
        return tuple(float(v) for v in values)

    if raw_expr.startswith("uniform(") and raw_expr.endswith(")"):
        args = _parse_value_list(raw_expr[len("uniform(") : -1], float)
        if len(args) != 2:
            raise ValueError(f"uniform requires two bounds, got {args}")
        return ParameterSpec(name=name, mode="uniform", values=as_tuple(args), dtype=dtype)

    if raw_expr.startswith("loguniform(") and raw_expr.endswith(")"):
        args = _parse_value_list(raw_expr[len("loguniform(") : -1], float)
        if len(args) != 2:
            raise ValueError(f"loguniform requires two bounds, got {args}")
        return ParameterSpec(name=name, mode="loguniform", values=as_tuple(args), dtype=dtype)

    if raw_expr.startswith("choice(") and raw_expr.endswith(")"):
        args = _parse_value_list(raw_expr[len("choice(") : -1], dtype)
        if len(args) == 0:
            raise ValueError("choice requires at least one candidate value")
        return ParameterSpec(name=name, mode="choice", values=as_tuple(args), dtype=dtype)

    value = _convert_token(raw_expr, dtype)
    return ParameterSpec(name=name, mode="constant", values=(float(value),), dtype=dtype)


def build_default_specs() -> Dict[str, ParameterSpec]:
    defaults = {
        "num_runs": ParameterSpec("num_runs", "constant", (float(DEFAULT_CONFIG.num_runs),), int),
        "shots": ParameterSpec("shots", "constant", (float(DEFAULT_CONFIG.shots),), int),
        "dephase_rate": ParameterSpec("dephase_rate", "constant", (float(DEFAULT_CONFIG.dephase_rate),), float),
        "client_fidelity": ParameterSpec("client_fidelity", "constant", (float(DEFAULT_CONFIG.client_fidelity),), float),
        "distance": ParameterSpec("distance", "loguniform", (30.0, 3000.0), float),
        "T1": ParameterSpec("T1", "constant", (float(DEFAULT_CONFIG.T1),), float),
        "client_T1": ParameterSpec("client_T1", "constant", (float(DEFAULT_CONFIG.client_T1),), float),
        "T2_ratio": ParameterSpec("T2_ratio", "constant", (float(DEFAULT_CONFIG.T2_ratio),), float),
        "sge": ParameterSpec("sge", "constant", (float(DEFAULT_CONFIG.sge),), float),
        "dge": ParameterSpec("dge", "constant", (float(DEFAULT_CONFIG.dge),), float),
        "gate_speed_factor": ParameterSpec(
            "gate_speed_factor",
            "loguniform",
            (1,100),
            float,
        ),
        # クライアント側のスケーリングもサーバと同様に範囲サンプリング（独立）
        # 既定はサーバと同じ loguniform(0.1, 100)
        "client_gate_speed_factor": ParameterSpec(
            "client_gate_speed_factor",
            "loguniform",
            (0.1, 100.0),
            float,
        ),
        "entanglement_fidelity": ParameterSpec("entanglement_fidelity", "constant", (float(DEFAULT_CONFIG.entanglement_fidelity),), float),
        "entanglement_speed_factor": ParameterSpec("entanglement_speed_factor", "loguniform", (2.0, 200.0), float),
        "theta": ParameterSpec("theta", "constant", (float(DEFAULT_THETA),), float),
    }
    return {name: replace(spec) for name, spec in defaults.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomized density-matrix experiments and CSV export"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/density_random.csv",
        help="Path to the output CSV (default: data/density_random.csv)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random experiments to generate per set",
    )
    parser.add_argument(
        "--sets",
        type=int,
        default=1,
        help="Number of independent experiment sets to run",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Seed used for the first simulation run",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=997,
        help="Stride added to the base seed for each subsequent simulation",
    )
    parser.add_argument(
        "--param-seed",
        type=int,
        help="Seed for the parameter random generator (defaults to base seed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--cli-factors",
        type=str,
        help=(
            "Comma/semicolon separated constants for client_gate_speed_factor sweep, "
            "e.g. --cli-factors 0.1,1,10,100. When set, one CSV is written per value "
            "plus an all-in-one concatenated CSV at --output."
        ),
    )
    parser.add_argument(
        "--cli-coupled",
        action="store_true",
        help=(
            "Couple client_gate_speed_factor to gate_speed_factor per candidate (previous behavior). "
            "Ignored when --cli-factors is provided."
        ),
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help=(
            "Parameter specification, e.g. distance=loguniform(1e2,1e3). "
            "Can be provided multiple times."
        ),
    )
    return parser


def _build_config_from_values(values: Dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        num_runs=int(values["num_runs"]),
        shots=int(values["shots"]),
        dephase_rate=float(values["dephase_rate"]),
        client_fidelity=float(values["client_fidelity"]),
        distance=float(values["distance"]),
        T1=float(values["T1"]),
        client_T1=float(values["client_T1"]),
        T2_ratio=float(values["T2_ratio"]),
        sge=float(values["sge"]),
        dge=float(values["dge"]),
        gate_speed_factor=float(values["gate_speed_factor"]),
        client_gate_speed_factor=float(values["client_gate_speed_factor"]),
        entanglement_fidelity=float(values["entanglement_fidelity"]),
        entanglement_speed_factor=float(values["entanglement_speed_factor"]),
    )


def run_random_sampling(args: argparse.Namespace) -> pd.DataFrame:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    specs = build_default_specs()
    custom_specs = set()
    for spec_expr in args.spec:
        spec = parse_parameter_spec(spec_expr)
        specs[spec.name] = spec
        custom_specs.add(spec.name)

    rng_seed = args.param_seed if args.param_seed is not None else args.base_seed
    rng = random.Random(rng_seed)

    total_sets = max(1, args.sets)
    samples_per_set = max(1, args.samples)

    experiment_metadata = {
        "experiment_sets": total_sets,
        "experiment_samples_per_set": samples_per_set,
        "experiment_base_seed": args.base_seed,
        "experiment_seed_stride": args.seed_stride,
        "experiment_param_seed": rng_seed,
        "experiment_custom_specs": ";".join(args.spec) if args.spec else "",
    }

    spec_columns = {
        f"spec_{name}": spec.describe() for name, spec in sorted(specs.items())
    }

    def _run_once(specs_local: Dict[str, ParameterSpec], *, tag: str | None = None) -> pd.DataFrame:
        records: List[Dict[str, float]] = []
        for set_index in range(total_sets):
            for set_sample_index in range(samples_per_set):
                global_index = set_index * samples_per_set + set_sample_index

                sampled_values: Dict[str, float] = {}

                gate_value = specs_local["gate_speed_factor"].sample(rng)
                sampled_values["gate_speed_factor"] = gate_value

                for name in CONFIG_PARAM_TYPES:
                    if name == "gate_speed_factor":
                        continue
                    # Handle client factor policy
                    if name == "client_gate_speed_factor":
                        # Coupled policy takes precedence unless a sweep is requested
                        if args.cli_coupled and not args.cli_factors:
                            sampled_values[name] = gate_value
                            continue
                        # If spec is constant (by --spec or sweep), use that constant
                        if (
                            name not in custom_specs
                            and specs_local[name].mode == "constant"
                        ):
                            sampled_values[name] = specs_local[name].values[0]
                            continue
                    sampled_values[name] = specs_local[name].sample(rng)

                theta_value = specs_local["theta"].sample(rng)

                config = _build_config_from_values(sampled_values)
                seed = args.base_seed + global_index * args.seed_stride

                energy, metrics = single_angle_metrics(theta_value, config, seed)
                filtered_metrics = {
                    k: v for k, v in metrics.items() if k not in EXCLUDED_DENSITY_KEYS
                }
                filtered_metrics.pop("energy", None)

                record = {
                    "set_index": set_index,
                    "set_sample_index": set_sample_index,
                    "sample_index": global_index,
                    **filtered_metrics,
                    "energy": energy,
                    **sampled_values,
                    "theta": theta_value,
                }
                if tag is not None:
                    record["experiment_cli_factor"] = tag
                record.update(experiment_metadata)
                record.update(spec_columns)
                records.append(record)

        return pd.DataFrame(records)

    def _fmt_tag(v: float) -> str:
        s = ("%g" % v).replace(".", "_")
        s = s.replace("-", "m")
        return s

    if args.cli_factors:
        parts = [p for p in args.cli_factors.replace(";", ",").split(",") if p.strip()]
        dfs: List[pd.DataFrame] = []
        for pstr in parts:
            try:
                val = float(pstr)
            except ValueError:
                raise ValueError(f"Invalid cli factor value: {pstr}")
            specs_run = {k: replace(v) for k, v in specs.items()}
            specs_run["client_gate_speed_factor"] = ParameterSpec(
                name="client_gate_speed_factor", mode="constant", values=(float(val),), dtype=float
            )
            tag = _fmt_tag(val)
            df_run = _run_once(specs_run, tag=tag)
            out = output_path.parent / f"{output_path.stem}_cli-{tag}{output_path.suffix}"
            if out.exists() and not args.overwrite:
                raise FileExistsError(f"{out} already exists. Use --overwrite to replace it.")
            df_run.to_csv(out, index=False)
            dfs.append(df_run)
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not df_all.empty:
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace it."
                )
            df_all.to_csv(output_path, index=False)
        return df_all
    else:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Use --overwrite to replace it."
            )
        df = _run_once(specs)
        df.to_csv(output_path, index=False)
        return df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df = run_random_sampling(args)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
