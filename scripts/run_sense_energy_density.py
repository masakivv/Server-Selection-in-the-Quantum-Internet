# """Sobol sensitivity analysis for energy error using density simulations."""

# from dataclasses import replace
# import os
# import time
# from typing import List, Tuple, Union

# import numpy as np
# import pandas as pd
# from SALib.analyze import sobol
# # from SALib.sample.sobol import sample as sobol_sample
# from SALib.sample.saltelli import sample as saltelli_sample

# from qs.density import DEFAULT_CONFIG, single_angle_metrics

# # プロジェクトルートの data ディレクトリを既定出力先に統一
# _DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Number = Union[int, float]


# def _ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)


# def _fmt_float(v: Number) -> str:
#     """Compact float formatting for filenames (e.g., 1e+03 -> 1e3)."""
#     v = float(v)
#     s = f"{v:.0e}"
#     if "e" not in s:
#         return s.replace(".", "_")
#     base, exp = s.split("e")
#     base = base.rstrip(".0") or "0"
#     sign = ""
#     if exp.startswith("-"):
#         sign = "-"
#         exp = exp[1:]
#     elif exp.startswith("+"):
#         exp = exp[1:]
#     exp = exp.lstrip("0") or "0"
#     return f"{base}e{sign}{exp}"


# def _fmt_range(r: Tuple[Number, Number], segs: int) -> str:
#     lo, hi = r
#     return f"{_fmt_float(lo)}-{_fmt_float(hi)}x{int(segs)}"


# def _build_output_name(
#     *,
#     ct_range: Tuple[Number, Number],
#     fid_error_range: Tuple[Number, Number],
#     third_key: str,
#     third_range: Tuple[Number, Number],
#     n_ct_seg: int,
#     n_fid_seg: int,
#     n_third_seg: int,
#     N_local: int,
#     shots: int,
# ) -> str:
#     parts = [
#         "energy_density",
#         f"ct-{_fmt_range(ct_range, n_ct_seg)}",
#         f"fid-{_fmt_range(fid_error_range, n_fid_seg)}",
#         f"{third_key}-{_fmt_range(third_range, n_third_seg)}",
#         f"N{int(N_local)}",
#         f"shots{int(shots)}",
#     ]
#     return "_".join(parts) + ".csv"


# def _midpoint_log10(low: float, high: float) -> float:
#     """Return the geometric mean for a log-spaced interval."""
#     return 10 ** ((np.log10(low) + np.log10(high)) / 2)


# def make_log_bins(
#     value_range: Tuple[Number, Number],
#     n_segments: int,
#     *,
#     base: float = 10.0,
#     include_max: bool = True,
# ) -> List[float]:
#     """Create logarithmic bins mirroring ``run_sense_energy``."""
#     min_val, max_val = map(float, value_range)
#     if min_val <= 0 or max_val <= 0:
#         raise ValueError("min_val と max_val は共に 0 より大きい必要があります。")
#     if min_val >= max_val:
#         raise ValueError("min_val は max_val より小さくなければいけません。")
#     if n_segments < 1:
#         raise ValueError("n_segments は 1 以上の整数にしてください。")

#     log_min = np.log(min_val) / np.log(base)
#     log_max = np.log(max_val) / np.log(base)
#     log_edges = np.linspace(log_min, log_max, n_segments + 1)

#     if not include_max:
#         log_edges = log_edges[:-1]

#     return list(base ** log_edges)


# def run_local_sobol_segment_analysis(
#     ct_range: Tuple[Number, Number] = (1e3, 1e18),
#     fid_error_range: Tuple[Number, Number] = (1e-4, 1e-2),
#     noise_range: Tuple[Number, Number] = (1e-3, 1e1),
#     *,
#     fidelity_mode: str = "absolute",  # "factor" or "absolute"
#     op_fidelity_range: Tuple[Number, Number] = (0.99, 0.9999),
#     n_ct_seg: int = 3,
#     n_fid_seg: int = 3,
#     n_noise_seg: int = 3,
#     N_local: int = 16,
#     target_energy: float = -1.1456,
#     output_dir: str = _DEFAULT_DATA_DIR,
#     shots: int = 5,
#     num_runs: int = 5,
#     theta: float = 0.2297,
#     base_seed: int = 42,
# ) -> Tuple[str, pd.DataFrame]:
#     """Run the energy sensitivity analysis using density-matrix simulations."""

#     if fidelity_mode not in ("factor", "absolute"):
#         raise ValueError("fidelity_mode は 'factor' か 'absolute' を指定してください。")

#     ct_bins = make_log_bins(ct_range, n_ct_seg)
#     fid_bins = make_log_bins(fid_error_range, n_fid_seg)
#     noise_bins = make_log_bins(noise_range, n_noise_seg)
#     # Use the same segment count for op_fidelity as for the third axis
#     opfid_bins = make_log_bins(op_fidelity_range, n_noise_seg)

#     start = time.perf_counter()
#     _ensure_dir(output_dir)
#     simple_rows = []
#     detail_rows = []

#     ct_bins = list(ct_bins)
#     fid_bins = list(fid_bins)
#     noise_bins = list(noise_bins)
#     opfid_bins = list(opfid_bins)

#     third_bins = noise_bins if fidelity_mode == "factor" else opfid_bins
#     total_segments = (len(ct_bins) - 1) * (len(fid_bins) - 1) * (len(third_bins) - 1)
#     seg_counter = 0
#     sample_counter = 0

#     base_config = replace(
#         DEFAULT_CONFIG,
#         num_runs=num_runs,
#         shots=shots,
#         distance=500.0,
#         entanglement_speed_factor=100.0,
#         gate_speed_factor=1.0,
#         client_gate_speed_factor=1.0,
#     )

#     for i1 in range(len(ct_bins) - 1):
#         for i2 in range(len(fid_bins) - 1):
#             # Use the selected third-axis bins for iteration
#             for i3 in range(len(third_bins) - 1):
#                 seg_counter += 1
#                 seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
#                 third_label = "noise_rate" if fidelity_mode == "factor" else "operation_fidelity"
#                 print(
#                     f"[{seg_counter}/{total_segments}] {seg_name} running… "
#                     f"{third_label}_bin=({third_bins[i3]:.3g},{third_bins[i3+1]:.3g})"
#                 )

#                 bounds_log10 = [
#                     [np.log10(ct_bins[i1]), np.log10(ct_bins[i1 + 1])],
#                     [np.log10(fid_bins[i2]), np.log10(fid_bins[i2 + 1])],
#                     [np.log10(third_bins[i3]), np.log10(third_bins[i3 + 1])],
#                 ]

#                 problem = {
#                     "num_vars": 3,
#                     # Note: the second parameter is sampled as an ERROR (1 - fidelity).
#                     # Use an explicit name to avoid downstream confusion.
#                     "names": (
#                         [
#                             "coherence_time",
#                             "entanglement_error",
#                             "noise_rate",
#                         ]
#                         if fidelity_mode == "factor"
#                         else [
#                             "coherence_time",
#                             "entanglement_error",
#                             "operation_fidelity",
#                         ]
#                     ),
#                     "bounds": bounds_log10,
#                 }

#                 # X_log = sobol_sample(problem, N_local, calc_second_order=False)
#                 X_log = saltelli_sample(problem, N_local, calc_second_order=False)
#                 X = 10 ** X_log

#                 energy_errors = []

#                 for row in X:
#                     sample_counter += 1
#                     ct, fid_error, third_var = row

#                     # Parameter sanitisation – SALib may generate values very close to
#                     # the interval boundaries, so we clip to keep physically valid
#                     # probabilities for the noise related parameters.
#                     fid_error = float(np.clip(fid_error, 1e-6, 0.999999))
#                     third_var = float(np.clip(third_var, 1e-9, 0.999999))

#                     ent_fid = 1.0 - fid_error
#                     if fidelity_mode == "factor":
#                         noise = third_var
#                         dephase_rate = 0.0039 * noise
#                         sge = 0.0006 * noise
#                         dge = 0.006 * noise
#                         client_fid_param = noise  # 既存仕様に合わせる
#                     else:
#                         op_fid = third_var
#                         err = 1.0 - op_fid
#                         dephase_rate = err
#                         sge = err
#                         dge = err
#                         client_fid_param = err

#                     config = replace(
#                         base_config,
#                         T1=float(ct),
#                         client_T1=float(ct),
#                         entanglement_fidelity=float(ent_fid),
#                         dephase_rate=float(dephase_rate),
#                         client_fidelity=float(client_fid_param),
#                         sge=float(sge),
#                         dge=float(dge),
#                     )

#                     energy, metrics = single_angle_metrics(
#                         theta=float(theta),
#                         config=config,
#                         seed=base_seed + sample_counter,
#                     )

#                     energy_error = abs(energy - target_energy)
#                     energy_errors.append(energy_error)

#                     detail_rows.append(
#                         {
#                             "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
#                             "coherence_time": float(ct),
#                             # Store both for clarity: we sample ERROR but also record the resulting fidelity
#                             "entanglement_error": float(fid_error),
#                             "entanglement_fidelity": float(ent_fid),
#                             "noise_rate": float(third_var) if fidelity_mode == "factor" else None,
#                             "operation_fidelity": float(third_var) if fidelity_mode == "absolute" else None,
#                             "operation_error": (1.0 - float(third_var)) if fidelity_mode == "absolute" else None,
#                             "energy": float(energy),
#                             "energy_error": float(energy_error),
#                             "zz_cost": float(metrics["zz_cost"]),
#                             "xx_cost": float(metrics["xx_cost"]),
#                             "total_time": float(metrics["total_time"]),
#                         }
#                     )

#                 # Sobol analysis can become numerically unstable when outputs are
#                 # (near-)constant within a segment. Guard against that case.
#                 Y = np.array(energy_errors, dtype=float)
#                 try:
#                     if not np.isfinite(np.var(Y)) or np.var(Y) < 1e-18:
#                         raise FloatingPointError("degenerate output variance")
#                     Si = sobol.analyze(problem, Y, calc_second_order=False)
#                     st0, st1, st2 = float(Si["ST"][0]), float(Si["ST"][1]), float(Si["ST"][2])
#                 except Exception:
#                     st0 = st1 = st2 = 0.0

#                 mid_ct = _midpoint_log10(ct_bins[i1], ct_bins[i1 + 1])
#                 mid_err = _midpoint_log10(fid_bins[i2], fid_bins[i2 + 1])
#                 mid_fid = 1.0 - mid_err
#                 mid_third = _midpoint_log10(third_bins[i3], third_bins[i3 + 1])

#                 simple_rows.append(
#                     {
#                         "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
#                         "coherence_time": mid_ct,
#                         # Report both: error (sampled) and the corresponding fidelity
#                         "entanglement_error": mid_err,
#                         "entanglement_fidelity": mid_fid,
#                         "noise_rate": mid_third if fidelity_mode == "factor" else None,
#                         "operation_fidelity": mid_third if fidelity_mode == "absolute" else None,
#                         "target_metric": float(np.mean(energy_errors)),
#                         "coherence_time_contribution": st0,
#                         "entanglement_error_contribution": st1,
#                         "entanglement_fidelity_contribution": st1,  # alias for compatibility
#                         "noise_rate_contribution": st2 if fidelity_mode == "factor" else None,
#                         "operation_fidelity_contribution": st2 if fidelity_mode == "absolute" else None,
#                         "use_log_scale": True,
#                         "fidelity_mode": fidelity_mode,
#                     }
#                 )

#     simple_df = pd.DataFrame(simple_rows)
#     detail_df = pd.DataFrame(detail_rows)

#     third_key = "noise" if fidelity_mode == "factor" else "opfid"
#     third_range = noise_range if fidelity_mode == "factor" else op_fidelity_range
#     out_name = _build_output_name(
#         ct_range=tuple(ct_range),
#         fid_error_range=tuple(fid_error_range),
#         third_key=third_key,
#         third_range=tuple(third_range),
#         n_ct_seg=n_ct_seg,
#         n_fid_seg=n_fid_seg,
#         n_third_seg=n_noise_seg,
#         N_local=N_local,
#         shots=shots,
#     )
#     path_simple = os.path.join(output_dir, out_name)
#     base, _ = os.path.splitext(out_name)
#     path_detail = os.path.join(output_dir, f"{base}_details.csv")

#     simple_df.to_csv(path_simple, index=False)
#     detail_df.to_csv(path_detail, index=False)

#     elapsed = time.perf_counter() - start
#     print(f"\n完了: {path_simple} に書き出しました  (elapsed {elapsed:.1f} s)")

#     return path_simple, simple_df


# def main() -> None:
#     t0 = time.perf_counter()
#     path_simple, _ = run_local_sobol_segment_analysis()
#     elapsed = time.perf_counter() - t0
#     print(f"\n===== 全体の実行時間: {elapsed:.1f} 秒 =====")


# if __name__ == "__main__":
#     main()

"""Sobol sensitivity analysis for energy error using density simulations.

主な修正点（前版からの差分）:
- Saltelli サンプリング列に対して **共通乱数（common random numbers）** を導入。
  A/B/A_Bi の同じ k 番目サンプルは同一シードを用いることで、確率ノイズの影響を大幅に低減。
- fidelity_mode="absolute" の第三軸を **operation_fidelity ではなく operation_error=1-op_fid**
  を対数一様にサンプリングするよう変更（感度解釈の一貫性向上）。
- ファイル名フォーマット `_fmt_float` を改善し、`0.99` と `0.9999` がともに `1e0` になる
  といった情報消失・衝突を解消。
- simple/detail 出力列を整備（absolute では operation_error を主、operation_fidelity は派生値）。

注: 計算の安定性を高めるには、shots/num_runs/N_local を十分大きく取ることを推奨します。
"""

from dataclasses import replace
import os
import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample.saltelli import sample as saltelli_sample

from qs.density import DEFAULT_CONFIG, single_angle_metrics

# Ensure NetSquid uses density-matrix formalism so noise models take effect
try:
    import netsquid as ns  # type: ignore
    try:
        from netsquid.qubits.qformalism import QFormalism  # type: ignore
        if hasattr(ns, "get_qstate_formalism") and hasattr(ns, "set_qstate_formalism"):
            if ns.get_qstate_formalism() != QFormalism.DM:
                ns.set_qstate_formalism(QFormalism.DM)
    except Exception:
        # Fallback for older NetSquid where DMForm is exported on ns
        if hasattr(ns, "DMForm") and hasattr(ns, "get_qstate_formalism") and hasattr(ns, "set_qstate_formalism"):
            if ns.get_qstate_formalism() != ns.DMForm:
                ns.set_qstate_formalism(ns.DMForm)
except Exception:
    # If NetSquid isn't available at import time, leave as-is
    pass

# プロジェクトルートの data ディレクトリを既定出力先に統一
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
    # 通常表記の範囲
    if 1e-3 <= av < 1e3:
        s = f"{v:.6g}"  # 有効桁6
        s = s.rstrip("0").rstrip(".")
        return s.replace(".", "_")
    # 指数表記（例: 1.23e-4 -> 1_23e-4）
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
    ct_range: Tuple[Number, Number] = (1e10, 1e12),
    fid_error_range: Tuple[Number, Number] = (1e-4, 1e-2),
    noise_range: Tuple[Number, Number] = (1e-3, 1e1),
    *,
    fidelity_mode: str = "absolute",  # "factor" or "absolute"
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
    """Run the energy sensitivity analysis using density-matrix simulations.

    注意:
      - Saltelli/Sobol 法では、確率的応答の場合でも A/B/A_Bi の同一インデックスで乱数系列を揃えること
        （共通乱数）で安定化させています。
      - fidelity_mode="absolute" では第三軸は operation_error = 1 - operation_fidelity を
        対数一様サンプリングします（出力には両方の列を記録）。
    """

    if fidelity_mode not in ("factor", "absolute"):
        raise ValueError("fidelity_mode は 'factor' か 'absolute' を指定してください。")

    # --- bin edges ----
    ct_bins = list(make_log_bins(ct_range, n_ct_seg))
    fid_bins = list(make_log_bins(fid_error_range, n_fid_seg))
    noise_bins = list(make_log_bins(noise_range, n_noise_seg))

    # absolute モードでは「誤差」を第三軸としてサンプリングする
    if fidelity_mode == "absolute":
        op_err_range = (1.0 - float(op_fidelity_range[1]), 1.0 - float(op_fidelity_range[0]))
        if not (0.0 < op_err_range[0] < op_err_range[1] < 1.0):
            raise ValueError("op_fidelity_range に基づく operation_error の範囲が不正です。")
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
    seg_counter = 0  # 1 始まりで進行表示に利用

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
                seg_idx = seg_counter - 1  # 0-based
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(
                    f"[{seg_counter}/{total_segments}] {seg_name} running… "
                    f"{third_label}_bin=({third_bins[i3]:.3g},{third_bins[i3+1]:.3g})"
                )

                # 変数はすべて log10 空間で一様サンプリングし、10**で戻す
                bounds_log10 = [
                    [np.log10(ct_bins[i1]), np.log10(ct_bins[i1 + 1])],
                    [np.log10(fid_bins[i2]), np.log10(fid_bins[i2 + 1])],
                    [np.log10(third_bins[i3]), np.log10(third_bins[i3 + 1])],
                ]

                if fidelity_mode == "factor":
                    var_names = ["coherence_time", "entanglement_error", "noise_rate"]
                else:
                    # absolute: 第三軸は operation_error を直接サンプル
                    var_names = ["coherence_time", "entanglement_error", "operation_error"]

                problem = {
                    "num_vars": 3,
                    "names": var_names,
                    "bounds": bounds_log10,  # sample/analyze の整合のため（analyze は bounds を直接は使わない）
                }

                # Saltelli サンプル（一次／全次のみを推定）
                X_log = saltelli_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log  # log10 -> linear scale

                # A/B/A_Bi の同じ k に共通乱数を使うためのベース N
                N_base = X.shape[0] // (problem["num_vars"] + 2)
                # セグメントごとに大きめのオフセットを付与して衝突回避
                seg_seed_base = int(base_seed) + seg_idx * 1_000_000

                energy_errors = []

                for idx, row in enumerate(X):
                    k = idx % N_base  # A, B, A_Bi の同一 k をそろえる
                    seed = seg_seed_base + k

                    ct, fid_error, third_var = row

                    # SALib は境界付近の値も生成するため、確率量を安全な範囲に制限
                    fid_error = float(np.clip(fid_error, 1e-9, 0.999999))
                    third_var = float(np.clip(third_var, 1e-12, 0.999999))

                    ent_fid = 1.0 - fid_error
                    if fidelity_mode == "factor":
                        noise = third_var
                        dephase_rate = 0.0039 * noise
                        sge = 0.0006 * noise
                        dge = 0.006 * noise
                        client_fid_param = noise  # 既存仕様に合わせる
                        op_err = None
                        op_fid = None
                    else:
                        op_err = third_var  # サンプルしたのは誤差
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
                            # 2軸目は誤差でサンプリング、忠実度も併記
                            "entanglement_error": float(fid_error),
                            "entanglement_fidelity": float(ent_fid),
                            # 第三軸：モードで記録内容を切替
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

                # Sobol 解析（出力分散が極小のときの不安定性を回避）
                Y = np.array(energy_errors, dtype=float)
                try:
                    varY = float(np.var(Y))
                    if not np.isfinite(varY) or varY < 1e-18:
                        raise FloatingPointError("degenerate output variance")
                    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
                    st0, st1, st2 = float(Si["ST"][0]), float(Si["ST"][1]), float(Si["ST"][2])
                except Exception:
                    st0 = st1 = st2 = 0.0

                # 区間の代表点（幾何平均）
                mid_ct = _midpoint_log10(ct_bins[i1], ct_bins[i1 + 1])
                mid_err = _midpoint_log10(fid_bins[i2], fid_bins[i2 + 1])
                mid_fid = 1.0 - mid_err
                mid_third = _midpoint_log10(third_bins[i3], third_bins[i3 + 1])

                # 出力（absolute のときは third は「誤差」）
                simple_row = {
                    "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                    "coherence_time": mid_ct,
                    # Report both: error (sampled) and the corresponding fidelity
                    "entanglement_error": mid_err,
                    "entanglement_fidelity": mid_fid,
                    "noise_rate": mid_third if fidelity_mode == "factor" else None,
                    "operation_error": mid_third if fidelity_mode == "absolute" else None,
                    "operation_fidelity": (1.0 - mid_third) if fidelity_mode == "absolute" else None,
                    "target_metric": float(np.mean(energy_errors)),
                    "coherence_time_contribution": st0,
                    "entanglement_error_contribution": st1,
                    "entanglement_fidelity_contribution": st1,  # alias for compatibility
                    # third-axis contributions
                    "noise_rate_contribution": st2 if fidelity_mode == "factor" else None,
                    "operation_error_contribution": st2 if fidelity_mode == "absolute" else None,
                    "operation_fidelity_contribution": st2 if fidelity_mode == "absolute" else None,  # alias
                    "use_log_scale": True,
                    "fidelity_mode": fidelity_mode,
                }
                simple_rows.append(simple_row)

    simple_df = pd.DataFrame(simple_rows)
    detail_df = pd.DataFrame(detail_rows)

    # ファイル名は第三軸の実際の変数に合わせる（absolute は 'operr'）
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
    print(f"\n完了: {path_simple} に書き出しました  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


def main() -> None:
    t0 = time.perf_counter()
    path_simple, _ = run_local_sobol_segment_analysis()
    elapsed = time.perf_counter() - t0
    print(f"\n===== 全体の実行時間: {elapsed:.1f} 秒 =====")


if __name__ == "__main__":
    main()
