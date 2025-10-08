"""Sobol sensitivity analysis for total execution time using density simulations."""

from dataclasses import replace
import os
import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample.sobol import sample as sobol_sample

from qs.density import DEFAULT_CONFIG, single_angle_metrics

# プロジェクトルートの data ディレクトリを既定出力先に統一
_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

Number = Union[int, float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_float(v: Number) -> str:
    """Compact float formatting suitable for filenames."""
    v = float(v)
    # Use scientific format, then normalize exponents like 1e+03 -> 1e3, 1e-03 -> 1e-3
    s = f"{v:.0e}"
    if "e" not in s:
        return s.replace(".", "_")
    base, exp = s.split("e")
    base = base.rstrip(".0") or "0"
    # normalize exponent
    sign = ""
    if exp.startswith("-"):
        sign = "-"
        exp = exp[1:]
    elif exp.startswith("+"):
        exp = exp[1:]
    exp = exp.lstrip("0") or "0"
    return f"{base}e{sign}{exp}"


def _fmt_range(r: Tuple[Number, Number], segs: int) -> str:
    """Format a numeric range and segment count like 1e1-1e3x3."""
    lo, hi = r
    return f"{_fmt_float(lo)}-{_fmt_float(hi)}x{int(segs)}"


def _build_output_name(
    *,
    timing_mode: str,
    dist_range: Tuple[Number, Number],
    ent_speed_range: Tuple[Number, Number],
    third_range: Tuple[Number, Number],
    n_dist_seg: int,
    n_ent_speed_seg: int,
    n_third_seg: int,
    N_local: int,
    shots: int,
) -> str:
    third_key = "gs" if timing_mode == "factor" else "gt"
    # Keep only ranges+segments, N and shots as requested
    parts = [
        "time_density",
        f"dist-{_fmt_range(dist_range, n_dist_seg)}",
        f"ent-{_fmt_range(ent_speed_range, n_ent_speed_seg)}",
        f"{third_key}-{_fmt_range(third_range, n_third_seg)}",
        f"N{int(N_local)}",
        f"shots{int(shots)}",
    ]
    return "_".join(parts) + ".csv"


def _midpoint_log10(low: float, high: float) -> float:
    """Return the geometric mean for a log-spaced interval."""
    return 10 ** ((np.log10(low) + np.log10(high)) / 2)


def make_log_bins(
    value_range: Tuple[Number, Number],
    n_segments: int,
    *,
    base: float = 10.0,
    include_max: bool = True,
) -> List[float]:
    """Create logarithmic bins mirroring run_sense_time configuration."""
    min_val, max_val = map(float, value_range)
    if min_val <= 0 or max_val <= 0:
        raise ValueError("min_val と max_val は共に 0 より大きい必要があります。")
    if min_val >= max_val:
        raise ValueError("min_val は max_val より小さくなければいけません。")
    if n_segments < 1:
        raise ValueError("n_segments は 1 以上の整数にしてください。")

    log_min = np.log(min_val) / np.log(base)
    log_max = np.log(max_val) / np.log(base)
    log_edges = np.linspace(log_min, log_max, n_segments + 1)

    if not include_max:
        log_edges = log_edges[:-1]

    return list(base ** log_edges)


def run_local_sobol_segment_analysis(
    dist_range: Tuple[Number, Number] = (1e2, 1e4),
    ent_speed_range: Tuple[Number, Number] = (1e2, 1e4),
    gate_speed_range: Tuple[Number, Number] = (1e0, 1e2),  # factor モード用（既定は使用しない）
    *,
    timing_mode: str = "absolute",  # "factor" or "absolute"（abs を既定に）
    gate_time_ns_range: Tuple[Number, Number] = (1e4, 1e6),
    n_dist_seg: int = 3,
    n_ent_speed_seg: int = 3,
    n_gate_speed_seg: int = 3,
    N_local: int = 32,
    output_dir: str = _DEFAULT_DATA_DIR,
    shots: int = 5,
    num_runs: int = 5,
    theta: float = 0.463 * np.pi,
    base_seed: int = 42,
) -> Tuple[str, pd.DataFrame]:
    """Run the time sensitivity analysis using density-matrix simulations."""
    dist_bins = make_log_bins(dist_range, n_dist_seg)
    ent_speed_bins = make_log_bins(ent_speed_range, n_ent_speed_seg)
    if timing_mode not in ("factor", "absolute"):
        raise ValueError("timing_mode は 'factor' か 'absolute' を指定してください。")

    gate_speed_bins = make_log_bins(gate_speed_range, n_gate_speed_seg)
    gate_time_bins = make_log_bins(gate_time_ns_range, n_gate_speed_seg)

    start = time.perf_counter()
    _ensure_dir(output_dir)
    simple_rows = []

    dist_bins = list(dist_bins)
    ent_speed_bins = list(ent_speed_bins)
    gate_speed_bins = list(gate_speed_bins)
    gate_time_bins = list(gate_time_bins)

    if timing_mode == "factor":
        third_bins = gate_speed_bins
    else:
        third_bins = gate_time_bins

    third_label = "gate_speed_factor" if timing_mode == "factor" else "gate_time_ns"
    print(
        "Config: timing_mode=%s, dist_range=%s, ent_speed_range=%s, %s_range=%s, segs=(%d,%d,%d), N_local=%d"
        % (
            timing_mode,
            tuple(dist_range),
            tuple(ent_speed_range),
            third_label,
            tuple(gate_speed_range if timing_mode == "factor" else gate_time_ns_range),
            n_dist_seg,
            n_ent_speed_seg,
            n_gate_speed_seg,
            N_local,
        )
    )

    total_segments = (len(dist_bins) - 1) * (len(ent_speed_bins) - 1) * (len(third_bins) - 1)
    seg_counter = 0
    sample_counter = 0

    base_config = replace(
        DEFAULT_CONFIG,
        num_runs=num_runs,
        shots=shots,
    )

    for i1 in range(len(dist_bins) - 1):
        for i2 in range(len(ent_speed_bins) - 1):
            for i3 in range(len(third_bins) - 1):
                seg_counter += 1
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(
                    f"[{seg_counter}/{total_segments}] {seg_name} running… "
                    f"{third_label}_bin=({third_bins[i3]:.3g},{third_bins[i3+1]:.3g})"
                )

                bounds_log10 = [
                    [np.log10(dist_bins[i1]), np.log10(dist_bins[i1 + 1])],
                    [
                        np.log10(ent_speed_bins[i2]),
                        np.log10(ent_speed_bins[i2 + 1]),
                    ],
                    [
                        np.log10(third_bins[i3]),
                        np.log10(third_bins[i3 + 1]),
                    ],
                ]

                problem = {
                    "num_vars": 3,
                    "names": (
                        [
                            "distance",
                            "entanglement_speed_factor",
                            "gate_speed_factor",
                        ]
                        if timing_mode == "factor"
                        else [
                            "distance",
                            "entanglement_speed_factor",
                            "gate_time_ns",
                        ]
                    ),
                    "bounds": bounds_log10,
                }

                X_log = sobol_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                total_times = []

                for row in X:
                    sample_counter += 1
                    distance, ent_speed, third_var = row

                    if timing_mode == "factor":
                        config = replace(
                            base_config,
                            distance=float(distance),
                            entanglement_speed_factor=float(ent_speed),
                            gate_speed_factor=float(third_var),
                            client_gate_speed_factor=float(third_var),
                            timing_mode="factor",
                            gate_time_ns=None,
                        )
                    else:
                        config = replace(
                            base_config,
                            distance=float(distance),
                            entanglement_speed_factor=float(ent_speed),
                            gate_speed_factor=1.0,
                            client_gate_speed_factor=1.0,
                            timing_mode="absolute",
                            gate_time_ns=float(third_var),
                        )

                    _, metrics = single_angle_metrics(
                        theta=float(theta),
                        config=config,
                        seed=base_seed + sample_counter,
                    )
                    total_times.append(metrics["total_time"])

                Si = sobol.analyze(problem, np.array(total_times), calc_second_order=False)

                mid_dist = _midpoint_log10(dist_bins[i1], dist_bins[i1 + 1])
                mid_ent_speed = _midpoint_log10(
                    ent_speed_bins[i2], ent_speed_bins[i2 + 1]
                )
                mid_third = _midpoint_log10(third_bins[i3], third_bins[i3 + 1])

                print(
                    " -> mean_time=%.3e ns, ST: dist=%.3f, ent=%.3f, third=%.3f"
                    % (
                        float(np.mean(total_times)),
                        float(Si["ST"][0]),
                        float(Si["ST"][1]),
                        float(Si["ST"][2]),
                    )
                )

                row_out = {
                    "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                    "distance": mid_dist,
                    "entanglement_speed_factor": mid_ent_speed,
                    "gate_speed_factor": mid_third if timing_mode == "factor" else None,
                    "gate_time_ns": mid_third if timing_mode == "absolute" else None,
                    "target_metric": float(np.mean(total_times)),
                    "distance_contribution": float(Si["ST"][0]),
                    "entanglement_speed_factor_contribution": float(Si["ST"][1]),
                    "gate_speed_factor_contribution": float(Si["ST"][2]) if timing_mode == "factor" else None,
                    "gate_time_ns_contribution": float(Si["ST"][2]) if timing_mode == "absolute" else None,
                    "use_log_scale": True,
                    "timing_mode": timing_mode,
                }
                simple_rows.append(row_out)

    simple_df = pd.DataFrame(simple_rows)

    # Build descriptive filename embedding experiment settings
    third_range = (
        gate_speed_range if timing_mode == "factor" else gate_time_ns_range
    )
    out_name = _build_output_name(
        timing_mode=timing_mode,
        dist_range=tuple(dist_range),
        ent_speed_range=tuple(ent_speed_range),
        third_range=tuple(third_range),
        n_dist_seg=n_dist_seg,
        n_ent_speed_seg=n_ent_speed_seg,
        n_third_seg=n_gate_speed_seg,
        N_local=N_local,
        shots=shots,
    )
    path_simple = os.path.join(output_dir, out_name)
    simple_df.to_csv(path_simple, index=False)

    elapsed = time.perf_counter() - start
    print(
        f"\n完了: {path_simple} に書き出しました  (elapsed {elapsed:.1f} s, rows={len(simple_df)}, cols={len(simple_df.columns)})"
    )

    return path_simple, simple_df


def main() -> None:
    t0 = time.perf_counter()
    # 既定を absolute モードで実行（factor 設定はコメントアウト）
    # path_simple, _ = run_local_sobol_segment_analysis(timing_mode="factor")
    path_simple, _ = run_local_sobol_segment_analysis(
        timing_mode="absolute",
        gate_time_ns_range=(1e4, 1e6),  # ns 単位
    )
    elapsed = time.perf_counter() - t0
    print(f"\n===== 全体の実行時間: {elapsed:.1f} 秒 =====")


if __name__ == "__main__":
    main()
