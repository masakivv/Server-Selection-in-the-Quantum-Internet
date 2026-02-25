from qs.simulation import ZZ_cost, XX_cost
import numpy as np
import pandas as pd
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
import os
import time
from typing import Tuple, Union, List

Number = Union[int, float]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _midpoint_log10(low: float, high: float) -> float:
    """Return the geometric mean of a log10 range."""
    return 10 ** ((np.log10(low) + np.log10(high)) / 2)


def make_log_bins(
    value_range: Tuple[Number, Number],
    n_segments: int,
    *,
    base: float = 10.0,
    include_max: bool = True,
) -> List[float]:
    """Create log-scale bin edges for a positive numeric range.

    The returned list is evenly spaced in log space (base `base`) and contains:
    - `n_segments + 1` edges when `include_max=True`
    - `n_segments` edges when `include_max=False`
    """
    min_val, max_val = map(float, value_range)
    if min_val <= 0 or max_val <= 0:
        raise ValueError("Both min_val and max_val must be greater than 0.")
    if min_val >= max_val:
        raise ValueError("min_val must be smaller than max_val.")
    if n_segments < 1:
        raise ValueError("n_segments must be an integer greater than or equal to 1.")

    log_min = np.log(min_val) / np.log(base)
    log_max = np.log(max_val) / np.log(base)
    log_edges = np.linspace(log_min, log_max, n_segments + 1)

    if not include_max:
        log_edges = log_edges[:-1]

    return list(base ** log_edges)


def run_local_sobol_segment_analysis(
    dist_range=(1e1, 1e3),
    ent_speed_range=(1e1, 1e3),
    gate_speed_range=(1e0, 1e2),
    n_dist_seg: int = 3,
    n_ent_speed_seg: int = 3,
    n_gate_speed_seg: int = 3,
    N_local: int = 32,
    output_dir: str = "data",
    shots: int = 10,
    num_runs: int = 5,
    theta: float = 0.463 * np.pi,
):
    dist_bins = make_log_bins(dist_range, n_dist_seg)
    ent_speed_bins = make_log_bins(ent_speed_range, n_ent_speed_seg)
    gate_speed_bins = make_log_bins(gate_speed_range, n_gate_speed_seg)

    start = time.perf_counter()
    _ensure_dir(output_dir)
    simple_rows = []

    dist_bins = list(dist_bins)
    ent_speed_bins = list(ent_speed_bins)
    gate_speed_bins = list(gate_speed_bins)

    total_segments = (len(dist_bins) - 1) * (len(ent_speed_bins) - 1) * (
        len(gate_speed_bins) - 1
    )
    seg_counter = 0

    for i1 in range(len(dist_bins) - 1):
        for i2 in range(len(ent_speed_bins) - 1):
            for i3 in range(len(gate_speed_bins) - 1):
                seg_counter += 1
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(f"[{seg_counter}/{total_segments}] {seg_name} running…")

                bounds_log10 = [
                    [np.log10(dist_bins[i1]), np.log10(dist_bins[i1 + 1])],
                    [
                        np.log10(ent_speed_bins[i2]),
                        np.log10(ent_speed_bins[i2 + 1]),
                    ],
                    [
                        np.log10(gate_speed_bins[i3]),
                        np.log10(gate_speed_bins[i3 + 1]),
                    ],
                ]

                problem = {
                    "num_vars": 3,
                    "names": [
                        "distance",
                        "entanglement_speed_factor",
                        "gate_speed_factor",
                    ],
                    "bounds": bounds_log10,
                }

                X_log = sobol_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                total_times = []

                for row in X:
                    distance, ent_speed, gate_speed = row


                    _, time_zz = ZZ_cost(
                        num_runs=num_runs,
                        dephase_rates=[0],
                        client_fidelitys=[0],
                        distances=[distance],
                        T1s=[1e50],
                        T2_ratios=[0.1],
                        client_T1s=[1e50],
                        sges=[0],
                        dges=[0],
                        gate_speed_factors=[gate_speed],
                        client_gate_speed_factors=[gate_speed],
                        entanglement_fidelities=[1],
                        entanglement_speed_factors=[ent_speed],
                        shots=shots,
                        angle=theta,
                        flag=0,
                    )
                    _, time_xx = XX_cost(
                        num_runs=num_runs,
                        dephase_rates=[0],
                        client_fidelitys=[0],
                        distances=[distance],
                        T1s=[1e50],
                        T2_ratios=[0.1],
                        client_T1s=[1e50],
                        sges=[0],
                        dges=[0],
                        gate_speed_factors=[gate_speed],
                        client_gate_speed_factors=[gate_speed],
                        entanglement_fidelities=[1],
                        entanglement_speed_factors=[ent_speed],
                        shots=shots,
                        angle=theta,
                        flag=1,
                    )

                    total_times.append(time_zz + time_xx)

                Si = sobol.analyze(problem, np.array(total_times), calc_second_order=False)

                mid_dist = _midpoint_log10(dist_bins[i1], dist_bins[i1 + 1])
                mid_ent_speed = _midpoint_log10(
                    ent_speed_bins[i2], ent_speed_bins[i2 + 1]
                )
                mid_gate = _midpoint_log10(
                    gate_speed_bins[i3], gate_speed_bins[i3 + 1]
                )

                simple_rows.append(
                    {
                        "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                        "distance": mid_dist,
                        "entanglement_speed_factor": mid_ent_speed,
                        "gate_speed_factor": mid_gate,
                        "target_metric": float(np.mean(total_times)),
                        "distance_contribution": float(Si["ST"][0]),
                        "entanglement_speed_factor_contribution": float(Si["ST"][1]),
                        "gate_speed_factor_contribution": float(Si["ST"][2]),
                        "use_log_scale": True,
                    }
                )

    simple_df = pd.DataFrame(simple_rows)
    path_simple = os.path.join(output_dir, "time_now.csv")
    simple_df.to_csv(path_simple, index=False)

    elapsed = time.perf_counter() - start
    print(f"\nDone: wrote to {path_simple}  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


def main():
    t0 = time.perf_counter()
    path_simple, _ = run_local_sobol_segment_analysis()
    elapsed = time.perf_counter() - t0
    print(f"\n===== Total runtime: {elapsed:.1f} s =====")


if __name__ == "__main__":
    main()
