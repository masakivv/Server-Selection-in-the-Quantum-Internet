"""Random sampling exporter for density-based simulations."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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
    "cc_jitter_mean_ns": float,
}

ALL_PARAM_TYPES: Mapping[str, type] = {
    **CONFIG_PARAM_TYPES,
    "theta": float,
}

NAME_ABBREVIATIONS: Mapping[str, str] = {

    "num_runs": "nr",
    "shots": "sh",
    "dephase_rate": "dr",
    "client_fidelity": "cf",
    "distance": "dist",
    "T1": "t1",
    "client_T1": "ct1",
    "T2_ratio": "t2r",
    "sge": "sge",
    "dge": "dge",
    "gate_speed_factor": "gsf",
    "client_gate_speed_factor": "cgsf",
    "entanglement_fidelity": "efid",
    "entanglement_speed_factor": "esf",
    "cc_jitter_mean_ns": "jit",
    "theta": "th",

    "samples": "smp",
    "sets": "set",
    "seed": "sd",
    "stride": "str",
    "paramseed": "psd",
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


def abbreviate_name(name: str, *, min_len: int = 2, max_len: int = 5) -> str:
    """Compact parameter/meta names to keep filenames short."""

    if name in NAME_ABBREVIATIONS:
        return NAME_ABBREVIATIONS[name]

    cleaned = name.replace("-", "_")
    parts = [p for p in cleaned.split("_") if p]
    if parts:
        candidate = "".join(p[0] for p in parts)
        if len(candidate) < min_len:
            candidate = "".join(parts)
    else:
        candidate = cleaned

    candidate = candidate.replace("_", "")
    if len(candidate) < min_len and candidate:
        candidate = candidate.ljust(min_len, candidate[-1])
    if not candidate:
        candidate = "xx"
    return candidate[:max_len]


def build_default_specs() -> Dict[str, ParameterSpec]:
    defaults = {
        "num_runs": ParameterSpec("num_runs", "constant", (float(DEFAULT_CONFIG.num_runs),), int),
        "shots": ParameterSpec("shots", "constant", (10000.0,), int),
        "dephase_rate": ParameterSpec("dephase_rate", "constant", (float(DEFAULT_CONFIG.dephase_rate),), float),
        "client_fidelity": ParameterSpec("client_fidelity", "constant", (float(DEFAULT_CONFIG.client_fidelity),), float),
        "distance": ParameterSpec("distance", "loguniform", (100.0, 1000.0), float),
        "T1": ParameterSpec("T1", "constant", (float(DEFAULT_CONFIG.T1),), float),
        "client_T1": ParameterSpec("client_T1", "constant", (float(DEFAULT_CONFIG.client_T1),), float),
        "T2_ratio": ParameterSpec("T2_ratio", "constant", (float(DEFAULT_CONFIG.T2_ratio),), float),
        "sge": ParameterSpec("sge", "constant", (float(DEFAULT_CONFIG.sge),), float),
        "dge": ParameterSpec("dge", "constant", (float(DEFAULT_CONFIG.dge),), float),
        "gate_speed_factor": ParameterSpec(
        "gate_speed_factor",
        "loguniform",
        (1,10),
        float,
    ),


        "client_gate_speed_factor": ParameterSpec(
            "client_gate_speed_factor",
            "loguniform",
            (0.1, 100.0),
            float,
        ),
        "entanglement_fidelity": ParameterSpec("entanglement_fidelity", "constant", (float(DEFAULT_CONFIG.entanglement_fidelity),), float),
        "entanglement_speed_factor": ParameterSpec("entanglement_speed_factor", "loguniform", (1.0, 100.0), float),
        "cc_jitter_mean_ns": ParameterSpec("cc_jitter_mean_ns", "constant", (float(DEFAULT_CONFIG.cc_jitter_mean_ns),), float),
        "theta": ParameterSpec("theta", "constant", (float(DEFAULT_THETA),), float),
    }
    return {name: replace(spec) for name, spec in defaults.items()}


REGIME_TRIPLET_ORDER = ("Tsrv", "Tcc", "Tent")

REGIME_TRIPLET_OVERRIDES: Mapping[str, Dict[str, ParameterSpec]] = {
    "Tsrv": {
        "gate_speed_factor": ParameterSpec(
            "gate_speed_factor",
            "loguniform",
            (0.474, 1.896),
            float,
        ),
        "distance": ParameterSpec(
            "distance",
            "loguniform",
            (50.0, 200.0),
            float,
        ),
        "entanglement_speed_factor": ParameterSpec(
            "entanglement_speed_factor",
            "loguniform",
            (500.0, 2000.0),
            float,
        ),
    },
    "Tcc": {
        "gate_speed_factor": ParameterSpec(
            "gate_speed_factor",
            "loguniform",
            (4.74, 18.96),
            float,
        ),
        "distance": ParameterSpec(
            "distance",
            "loguniform",
            (500.0, 2000.0),
            float,
        ),
        "entanglement_speed_factor": ParameterSpec(
            "entanglement_speed_factor",
            "loguniform",
            (500.0, 2000.0),
            float,
        ),
    },
    "Tent": {
        "gate_speed_factor": ParameterSpec(
            "gate_speed_factor",
            "loguniform",
            (4.74, 18.96),
            float,
        ),
        "distance": ParameterSpec(
            "distance",
            "loguniform",
            (50.0, 200.0),
            float,
        ),
        "entanglement_speed_factor": ParameterSpec(
            "entanglement_speed_factor",
            "loguniform",
            (50.0, 200.0),
            float,
        ),
    },
}

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
        "--regime-triplet",
        action="store_true",
        help=(
            "If set, ignore --spec/--cli-factors and run three predefined "
            "Tsrv/Tcc/Tent-dominated sampling regimes, writing one CSV per regime "
            "plus an all-in-one concatenated CSV at --output."
        ),
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help=(
            "Parameter specification, e.g. distance=loguniform(1e2,1e3). "
            "gate_speed_factor / client_gate_speed_factor are multiplicative "
            "scalers on gate+measure time (larger => slower). "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--classical-rng-seed-offset",
        type=int,
        default=100000,
        help="Offset applied to classical-channel RNGs relative to base_seed (rng_ab:+offset, rng_ba:+offset+1)",
    )
    parser.add_argument(
        "--ext-stochastic",
        action="store_true",
        help="Trigger the external entanglement source at exponential intervals (default: fixed interval)",
    )
    parser.add_argument(
        "--ext-seed",
        type=int,
        default=None,
        help="RNG seed for external entanglement source (default: base_seed+200002 if omitted)",
    )
    parser.add_argument(
        "--ext-min-delay-ns",
        type=float,
        default=DEFAULT_CONFIG.ext_min_delay_ns,
        help="Minimum delay applied to exponential samples [ns] (prevents zero-wait loops)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples (0 to disable)",
    )
    return parser


def _build_config_from_values(
    values: Dict[str, float],
    *,
    ext_stochastic: bool,
    ext_seed: Optional[int],
    ext_min_delay_ns: float,
) -> SimulationConfig:
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
        cc_jitter_mean_ns=float(values["cc_jitter_mean_ns"]),
        ext_stochastic=ext_stochastic,
        ext_seed=ext_seed,
        ext_min_delay_ns=ext_min_delay_ns,
    )


def run_random_sampling(args: argparse.Namespace) -> pd.DataFrame:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regime_triplet and args.cli_factors:
        raise ValueError("--regime-triplet cannot be combined with --cli-factors")
    if args.regime_triplet and args.spec:
        raise ValueError("--regime-triplet cannot be combined with --spec")

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
        "experiment_ext_stochastic": int(bool(args.ext_stochastic)),
        "experiment_ext_seed": args.ext_seed if args.ext_seed is not None else "",
        "experiment_ext_min_delay_ns": args.ext_min_delay_ns,
    }

    def _run_once(
        specs_local: Dict[str, ParameterSpec],
        *,
        tag: str | None = None,
        regime: str | None = None,
    ) -> pd.DataFrame:
        spec_columns = {
            f"spec_{name}": spec.describe() for name, spec in sorted(specs_local.items())
        }
        records: List[Dict[str, float]] = []


        DEBUG_TARGET_SAMPLE = None
        debug_target_done = False
        total_samples = total_sets * samples_per_set
        for set_index in range(total_sets):
            if debug_target_done:
                break
            for set_sample_index in range(samples_per_set):
                global_index = set_index * samples_per_set + set_sample_index

                sampled_values: Dict[str, float] = {}

                gate_value = specs_local["gate_speed_factor"].sample(rng)
                sampled_values["gate_speed_factor"] = gate_value

                for name in CONFIG_PARAM_TYPES:
                    if name == "gate_speed_factor":
                        continue

                    if name == "client_gate_speed_factor":

                        if args.cli_coupled and not args.cli_factors:
                            sampled_values[name] = gate_value
                            continue

                        if (
                            name not in custom_specs
                            and specs_local[name].mode == "constant"
                        ):
                            sampled_values[name] = specs_local[name].values[0]
                            continue
                    sampled_values[name] = specs_local[name].sample(rng)

                theta_value = specs_local["theta"].sample(rng)

                config = _build_config_from_values(
                    sampled_values,
                    ext_stochastic=args.ext_stochastic,
                    ext_seed=args.ext_seed,
                    ext_min_delay_ns=args.ext_min_delay_ns,
                )
                seed = args.base_seed + global_index * args.seed_stride

                rng_ab = np.random.RandomState(seed + args.classical_rng_seed_offset)
                rng_ba = np.random.RandomState(seed + args.classical_rng_seed_offset + 1)

                if DEBUG_TARGET_SAMPLE is not None:
                    if global_index < DEBUG_TARGET_SAMPLE:
                        continue
                    if global_index > DEBUG_TARGET_SAMPLE:
                        debug_target_done = True
                        break

                try:
                    energy, metrics = single_angle_metrics(
                        theta_value,
                        config,
                        seed,
                        rng_ab=rng_ab,
                        rng_ba=rng_ba,
                    )
                except OverflowError:
                    debug_params = {
                        "distance": sampled_values.get("distance"),
                        "entanglement_speed_factor": sampled_values.get("entanglement_speed_factor"),
                        "gate_speed_factor": sampled_values.get("gate_speed_factor"),
                        "client_gate_speed_factor": sampled_values.get("client_gate_speed_factor"),
                        "cc_jitter_mean_ns": sampled_values.get("cc_jitter_mean_ns"),
                        "theta": theta_value,
                    }
                    print(f"OverflowError at sample {global_index} seed {seed}")
                    print(f"params: {debug_params}")
                    raise
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
                if regime is not None:
                    record["experiment_regime"] = regime
                record.update(experiment_metadata)
                record.update(spec_columns)
                records.append(record)

                if args.progress_every and (global_index + 1) % args.progress_every == 0:
                    tag_txt = f" cli={tag}" if tag is not None else ""
                    regime_txt = f" regime={regime}" if regime is not None else ""
                    print(
                        f"[{global_index + 1}/{total_samples}] samples done{tag_txt}{regime_txt}",
                        flush=True,  
                    )

        return pd.DataFrame(records)

    def _fmt_tag(v: float) -> str:
        s = ("%g" % v).replace(".", "_")
        s = s.replace("-", "m")
        return s

    def _build_experiment_suffix(
        specs_map: Dict[str, ParameterSpec],
        selected_names: Iterable[str],
        *,
        force_names: Iterable[str] = (),
    ) -> str:
        """Return a suffix reflecting parameter specs and run metadata."""

        spec_parts: List[str] = []
        include_names = set(selected_names) | set(force_names)
        for name in sorted(include_names):
            spec = specs_map.get(name)
            if spec is None:
                continue
            abbr = abbreviate_name(name)
            if spec.mode in ("uniform", "loguniform"):
                v0, v1 = spec.values[:2]
                spec_parts.append(f"{abbr}-{_fmt_tag(v0)}-{_fmt_tag(v1)}")
            elif spec.mode == "choice":
                vals = "-".join(_fmt_tag(v) for v in spec.values)
                spec_parts.append(f"{abbr}-{vals}")
            elif spec.mode == "constant":
                formatted = "-".join(_fmt_tag(v) for v in spec.values)
                spec_parts.append(f"{abbr}-{formatted}")

        meta_parts = [
            f"{abbreviate_name('samples')}-{_fmt_tag(samples_per_set)}",
            f"{abbreviate_name('sets')}-{_fmt_tag(total_sets)}",
            f"{abbreviate_name('seed')}-{_fmt_tag(args.base_seed)}",
            f"{abbreviate_name('stride')}-{_fmt_tag(args.seed_stride)}",
            f"{abbreviate_name('paramseed')}-{_fmt_tag(rng_seed)}",
            f"{abbreviate_name('ext_stochastic')}-{_fmt_tag(int(bool(args.ext_stochastic)))}",
        ]

        parts = spec_parts + meta_parts
        return ("_" + "_".join(parts)) if parts else ""

    if args.cli_factors:
        parts = [p for p in args.cli_factors.replace(";", ",").split(",") if p.strip()]
        dfs: List[pd.DataFrame] = []

        base_suffix = _build_experiment_suffix(
            specs,
            custom_specs,
            force_names={"distance", "shots"},
        )
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
            out = output_path.parent / f"{output_path.stem}_cli-{tag}{base_suffix}{output_path.suffix}"
            if out.exists() and not args.overwrite:
                raise FileExistsError(f"{out} already exists. Use --overwrite to replace it.")
            df_run.to_csv(out, index=False)
            dfs.append(df_run)
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not df_all.empty:
            out_all = output_path.parent / f"{output_path.stem}{base_suffix}{output_path.suffix}"
            if out_all.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{out_all} already exists. Use --overwrite to replace it."
                )
            df_all.to_csv(out_all, index=False)

            args.output = str(out_all)
        return df_all
    elif args.regime_triplet:
        regime_dfs: List[pd.DataFrame] = []
        combined_suffix = _build_experiment_suffix(
            specs,
            {"shots"},
        )

        def _build_regime_specs(regime_name: str) -> Dict[str, ParameterSpec]:
            regime_specs = build_default_specs()
            for pname, spec in REGIME_TRIPLET_OVERRIDES[regime_name].items():
                regime_specs[pname] = replace(spec)
            return regime_specs

        for regime_name in REGIME_TRIPLET_ORDER:
            specs_regime = _build_regime_specs(regime_name)
            df_regime = _run_once(specs_regime, regime=regime_name)
            regime_suffix = _build_experiment_suffix(
                specs_regime,
                set(REGIME_TRIPLET_OVERRIDES[regime_name].keys()) | {"shots"},
            )
            out = output_path.parent / f"{output_path.stem}_regime-{regime_name}{regime_suffix}{output_path.suffix}"
            if out.exists() and not args.overwrite:
                raise FileExistsError(f"{out} already exists. Use --overwrite to replace it.")
            df_regime.to_csv(out, index=False)
            regime_dfs.append(df_regime)

        df_all = pd.concat(regime_dfs, ignore_index=True)
        out_all = output_path.parent / f"{output_path.stem}_regimes-all{combined_suffix}{output_path.suffix}"
        if out_all.exists() and not args.overwrite:
            raise FileExistsError(f"{out_all} already exists. Use --overwrite to replace it.")
        df_all.to_csv(out_all, index=False)
        args.output = str(out_all)
        return df_all
    else:

        base_suffix = _build_experiment_suffix(
            specs,
            custom_specs,
            force_names={"distance", "shots"},
        )
        out = output_path.parent / f"{output_path.stem}{base_suffix}{output_path.suffix}"
        if out.exists() and not args.overwrite:
            raise FileExistsError(
                f"{out} already exists. Use --overwrite to replace it."
            )
        df = _run_once(specs)
        df.to_csv(out, index=False)

        args.output = str(out)
        return df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df = run_random_sampling(args)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
