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
    ct_range=(1e11, 1e13),
    fid_range=(1e-3, 1e-1),
    noise_range=(1e-2, 1e0),
    n_ct_seg: int = 3,
    n_fid_seg: int = 3,
    n_noise_seg: int = 3,
    N_local: int = 64,
    target_energy: float = -1.1457,
    output_dir: str = "data",
    shots: int = 1000,
    num_runs: int = 5,
    flag: int = 0,
    theta: float = 0.1149 * np.pi,
):
    """Replicate the energy_now sensitivity experiment."""
    ct_bins = make_log_bins(ct_range, n_ct_seg)
    fid_bins = make_log_bins(fid_range, n_fid_seg)
    noise_bins = make_log_bins(noise_range, n_noise_seg)

    start = time.perf_counter()
    _ensure_dir(output_dir)
    simple_rows = []
    detail_rows = []

    ct_bins = list(ct_bins)
    fid_bins = list(fid_bins)
    noise_bins = list(noise_bins)

    total_segments = (len(ct_bins) - 1) * (len(fid_bins) - 1) * (len(noise_bins) - 1)
    seg_counter = 0

    for i1 in range(len(ct_bins) - 1):
        for i2 in range(len(fid_bins) - 1):
            for i3 in range(len(noise_bins) - 1):
                seg_counter += 1
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(f"[{seg_counter}/{total_segments}] {seg_name} running…")

                bounds_log10 = [
                    [np.log10(ct_bins[i1]), np.log10(ct_bins[i1 + 1])],
                    [np.log10(fid_bins[i2]), np.log10(fid_bins[i2 + 1])],
                    [np.log10(noise_bins[i3]), np.log10(noise_bins[i3 + 1])],
                ]

                problem = {
                    "num_vars": 3,
                    "names": [
                        "coherence_time",
                        "entanglement_fidelity",
                        "noise_rate",
                    ],
                    "bounds": bounds_log10,
                }

                X_log = sobol_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                energy_errors = []

                for row in X:
                    ct, fid_error, noise = row
                    ent_fid = 1.0 - fid_error
                    sges = [0.0006 * noise]
                    dges = [0.006 * noise]
                    dephase_rates = [0.0039 * noise]

                    cost_zz, _ = ZZ_cost(
                        num_runs=num_runs,
                        dephase_rates=dephase_rates,
                        client_fidelitys=[noise],
                        distances=[500],
                        T1s=[ct],
                        T2_ratios=[0.1],
                        client_T1s=[ct],
                        sges=sges,
                        dges=dges,
                        gate_speed_factors=[1],
                        client_gate_speed_factors=[1],
                        entanglement_fidelities=[ent_fid],
                        entanglement_speed_factors=[100],
                        shots=shots,
                        angle=theta,
                        flag=0,
                    )
                    cost_xx, _ = XX_cost(
                        num_runs=num_runs,
                        dephase_rates=dephase_rates,
                        client_fidelitys=[noise],
                        distances=[500],
                        T1s=[ct],
                        T2_ratios=[0.1],
                        client_T1s=[ct],
                        sges=sges,
                        dges=dges,
                        gate_speed_factors=[1],
                        client_gate_speed_factors=[1],
                        entanglement_fidelities=[ent_fid],
                        entanglement_speed_factors=[100],
                        shots=shots,
                        angle=theta,
                        flag=1,
                    )

                    final_energy = cost_zz + 2 * cost_xx + 1 / 1.4172975


                    energy_error = abs(final_energy - target_energy)

                    detail_rows.append(
                        {
                            "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                            "coherence_time": float(ct),
                            "entanglement_fidelity": float(ent_fid),
                            "noise_rate": float(noise),
                            "cost_zz": float(cost_zz),
                            "cost_xx": float(cost_xx),
                            "final_energy": float(final_energy),
                            "energy_error": float(energy_error),
                        }
                    )
                    energy_errors.append(energy_error)

                Si = sobol.analyze(problem, np.array(energy_errors), calc_second_order=False)

                mid_ct = _midpoint_log10(ct_bins[i1], ct_bins[i1 + 1])
                mid_fid = _midpoint_log10(fid_bins[i2], fid_bins[i2 + 1])
                mid_noise = _midpoint_log10(noise_bins[i3], noise_bins[i3 + 1])

                simple_rows.append(
                    {
                        "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                        "coherence_time": mid_ct,
                        "entanglement_fidelity": mid_fid,
                        "noise_rate": mid_noise,
                        "target_metric": float(np.mean(energy_errors)),
                        "coherence_time_contribution": float(Si["ST"][0]),
                        "entanglement_fidelity_contribution": float(Si["ST"][1]),
                        "noise_rate_contribution": float(Si["ST"][2]),
                        "use_log_scale": True,
                    }
                )

    simple_df = pd.DataFrame(simple_rows)
    path_simple = os.path.join(output_dir, "energy_now.csv")
    simple_df.to_csv(path_simple, index=False)

    detail_df = pd.DataFrame(detail_rows)
    path_detail = os.path.join(output_dir, "energy_details.csv")
    detail_df.to_csv(path_detail, index=False)


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
