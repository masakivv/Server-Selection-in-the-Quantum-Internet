from dataclasses import replace
import os
import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample.saltelli import sample as saltelli_sample

from qs.density import DEFAULT_CONFIG, single_angle_metrics


try:
    import netsquid as ns  
    try:
        from netsquid.qubits.qformalism import QFormalism  
        if hasattr(ns, "get_qstate_formalism") and hasattr(ns, "set_qstate_formalism"):
            if ns.get_qstate_formalism() != QFormalism.DM:
                ns.set_qstate_formalism(QFormalism.DM)
    except Exception:

        if hasattr(ns, "DMForm") and hasattr(ns, "get_qstate_formalism") and hasattr(ns, "set_qstate_formalism"):
            if ns.get_qstate_formalism() != ns.DMForm:
                ns.set_qstate_formalism(ns.DMForm)
except Exception:

    pass


_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

Number = Union[int, float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_float(v: Number) -> str:
    """Compact but informative float formatting for filenames without losing precision."""
    v = float(v)
    if v == 0.0:
        return "0"
    av = abs(v)

    if 1e-3 <= av < 1e3:
        s = f"{v:.6g}"  
        s = s.rstrip("0").rstrip(".")
        return s.replace(".", "_")

    s = f"{v:.3e}"
    base, exp = s.split("e")
    base = base.rstrip("0").rstrip(".").replace(".", "_")
    sign = "-" if exp.startswith("-") else ""
    exp = exp.lstrip("+-0") or "0"
    return f"{base}e{sign}{exp}"


def _fmt_range(r: Tuple[Number, Number], segs: int) -> str:
    lo, hi = r
    return f"{_fmt_float(lo)}-{_fmt_float(hi)}x{int(segs)}"


def _build_output_name(
    *,
    ct_range: Tuple[Number, Number],
    fid_error_range: Tuple[Number, Number],
    third_key: str,
    third_range: Tuple[Number, Number],
    n_ct_seg: int,
    n_fid_seg: int,
    n_third_seg: int,
    N_local: int,
    shots: int,
) -> str:
    parts = [
        "energy_density",
        f"ct-{_fmt_range(ct_range, n_ct_seg)}",
        f"fid-{_fmt_range(fid_error_range, n_fid_seg)}",
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
    """Create logarithmic bins."""
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
    ct_range: Tuple[Number, Number] = (1e10, 1e12),
    fid_error_range: Tuple[Number, Number] = (1e-4, 1e-2),
    noise_range: Tuple[Number, Number] = (1e-3, 1e1),
    *,
    fidelity_mode: str = "absolute",  
    op_fidelity_range: Tuple[Number, Number] = (0.99, 0.9999),
    n_ct_seg: int = 3,
    n_fid_seg: int = 3,
    n_noise_seg: int = 3,
    N_local: int = 32,
    target_energy: float = -1.1456,
    output_dir: str = _DEFAULT_DATA_DIR,
    shots: int = 3,
    num_runs: int = 5,
    theta: float = 0.2297,
    base_seed: int = 42,
) -> Tuple[str, pd.DataFrame]:

    if fidelity_mode not in ("factor", "absolute"):
        raise ValueError("fidelity_mode must be either 'factor' or 'absolute'.")


    ct_bins = list(make_log_bins(ct_range, n_ct_seg))
    fid_bins = list(make_log_bins(fid_error_range, n_fid_seg))
    noise_bins = list(make_log_bins(noise_range, n_noise_seg))


    if fidelity_mode == "absolute":
        op_err_range = (1.0 - float(op_fidelity_range[1]), 1.0 - float(op_fidelity_range[0]))
        if not (0.0 < op_err_range[0] < op_err_range[1] < 1.0):
            raise ValueError("Invalid operation_error range derived from op_fidelity_range.")
        third_bins = list(make_log_bins(op_err_range, n_noise_seg))
        third_label = "operation_error"
    else:
        third_bins = noise_bins
        third_label = "noise_rate"

    start = time.perf_counter()
    _ensure_dir(output_dir)
    simple_rows = []
    detail_rows = []

    total_segments = (len(ct_bins) - 1) * (len(fid_bins) - 1) * (len(third_bins) - 1)
    seg_counter = 0  

    base_config = replace(
        DEFAULT_CONFIG,
        num_runs=num_runs,
        shots=shots,
        distance=500.0,
        entanglement_speed_factor=100.0,
        gate_speed_factor=1.0,
        client_gate_speed_factor=1.0,
    )

    for i1 in range(len(ct_bins) - 1):
        for i2 in range(len(fid_bins) - 1):
            for i3 in range(len(third_bins) - 1):
                seg_counter += 1
                seg_idx = seg_counter - 1  
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(
                    f"[{seg_counter}/{total_segments}] {seg_name} running… "
                    f"{third_label}_bin=({third_bins[i3]:.3g},{third_bins[i3+1]:.3g})"
                )


                bounds_log10 = [
                    [np.log10(ct_bins[i1]), np.log10(ct_bins[i1 + 1])],
                    [np.log10(fid_bins[i2]), np.log10(fid_bins[i2 + 1])],
                    [np.log10(third_bins[i3]), np.log10(third_bins[i3 + 1])],
                ]

                if fidelity_mode == "factor":
                    var_names = ["coherence_time", "entanglement_error", "noise_rate"]
                else:

                    var_names = ["coherence_time", "entanglement_error", "operation_error"]

                problem = {
                    "num_vars": 3,
                    "names": var_names,
                    "bounds": bounds_log10,  
                }


                X_log = saltelli_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log  


                N_base = X.shape[0] // (problem["num_vars"] + 2)

                seg_seed_base = int(base_seed) + seg_idx * 1_000_000

                energy_errors = []

                for idx, row in enumerate(X):
                    k = idx % N_base  
                    seed = seg_seed_base + k

                    ct, fid_error, third_var = row


                    fid_error = float(np.clip(fid_error, 1e-9, 0.999999))
                    third_var = float(np.clip(third_var, 1e-12, 0.999999))

                    ent_fid = 1.0 - fid_error
                    if fidelity_mode == "factor":
                        noise = third_var
                        dephase_rate = 0.0039 * noise
                        sge = 0.0006 * noise
                        dge = 0.006 * noise
                        client_fid_param = noise  
                        op_err = None
                        op_fid = None
                    else:
                        op_err = third_var  
                        op_fid = 1.0 - op_err
                        dephase_rate = op_err
                        sge = op_err
                        dge = op_err
                        client_fid_param = op_err

                    config = replace(
                        base_config,
                        T1=float(ct),
                        client_T1=float(ct),
                        entanglement_fidelity=float(ent_fid),
                        dephase_rate=float(dephase_rate),
                        client_fidelity=float(client_fid_param),
                        sge=float(sge),
                        dge=float(dge),
                    )

                    energy, metrics = single_angle_metrics(
                        theta=float(theta),
                        config=config,
                        seed=seed,
                    )

                    energy_error = abs(energy - target_energy)
                    energy_errors.append(energy_error)

                    detail_rows.append(
                        {
                            "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                            "coherence_time": float(ct),

                            "entanglement_error": float(fid_error),
                            "entanglement_fidelity": float(ent_fid),

                            "noise_rate": float(third_var) if fidelity_mode == "factor" else None,
                            "operation_error": float(op_err) if fidelity_mode == "absolute" else None,
                            "operation_fidelity": float(op_fid) if fidelity_mode == "absolute" else None,
                            "energy": float(energy),
                            "energy_error": float(energy_error),
                            "zz_cost": float(metrics["zz_cost"]),
                            "xx_cost": float(metrics["xx_cost"]),
                            "total_time": float(metrics["total_time"]),
                        }
                    )


                Y = np.array(energy_errors, dtype=float)
                try:
                    varY = float(np.var(Y))
                    if not np.isfinite(varY) or varY < 1e-18:
                        raise FloatingPointError("degenerate output variance")
                    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
                    st0, st1, st2 = float(Si["ST"][0]), float(Si["ST"][1]), float(Si["ST"][2])
                except Exception:
                    st0 = st1 = st2 = 0.0


                mid_ct = _midpoint_log10(ct_bins[i1], ct_bins[i1 + 1])
                mid_err = _midpoint_log10(fid_bins[i2], fid_bins[i2 + 1])
                mid_fid = 1.0 - mid_err
                mid_third = _midpoint_log10(third_bins[i3], third_bins[i3 + 1])


                simple_row = {
                    "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                    "coherence_time": mid_ct,

                    "entanglement_error": mid_err,
                    "entanglement_fidelity": mid_fid,
                    "noise_rate": mid_third if fidelity_mode == "factor" else None,
                    "operation_error": mid_third if fidelity_mode == "absolute" else None,
                    "operation_fidelity": (1.0 - mid_third) if fidelity_mode == "absolute" else None,
                    "target_metric": float(np.mean(energy_errors)),
                    "coherence_time_contribution": st0,
                    "entanglement_error_contribution": st1,
                    "entanglement_fidelity_contribution": st1,  

                    "noise_rate_contribution": st2 if fidelity_mode == "factor" else None,
                    "operation_error_contribution": st2 if fidelity_mode == "absolute" else None,
                    "operation_fidelity_contribution": st2 if fidelity_mode == "absolute" else None,  
                    "use_log_scale": True,
                    "fidelity_mode": fidelity_mode,
                }
                simple_rows.append(simple_row)

    simple_df = pd.DataFrame(simple_rows)
    detail_df = pd.DataFrame(detail_rows)


    if fidelity_mode == "factor":
        third_key = "noise"
        third_range = noise_range
    else:
        third_key = "operr"
        third_range = (1.0 - float(op_fidelity_range[1]), 1.0 - float(op_fidelity_range[0]))

    out_name = _build_output_name(
        ct_range=tuple(ct_range),
        fid_error_range=tuple(fid_error_range),
        third_key=third_key,
        third_range=tuple(third_range),
        n_ct_seg=n_ct_seg,
        n_fid_seg=n_fid_seg,
        n_third_seg=n_noise_seg,
        N_local=N_local,
        shots=shots,
    )
    path_simple = os.path.join(output_dir, out_name)
    base, _ = os.path.splitext(out_name)
    path_detail = os.path.join(output_dir, f"{base}_details.csv")

    simple_df.to_csv(path_simple, index=False)
    detail_df.to_csv(path_detail, index=False)

    elapsed = time.perf_counter() - start
    print(f"\nDone: wrote to {path_simple}  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


def main() -> None:
    t0 = time.perf_counter()
    path_simple, _ = run_local_sobol_segment_analysis()
    elapsed = time.perf_counter() - t0
    print(f"\n===== Total runtime: {elapsed:.1f} s =====")


if __name__ == "__main__":
    main()
