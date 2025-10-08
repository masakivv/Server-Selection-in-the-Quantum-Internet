
import numpy as np
import pandas as pd
import time
import itertools
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
import os
from .simulation import run_vqe_optimization_experiment

# Scale legends a bit larger for readability
_LEGEND_SCALE = 1.3

def _legend_fontsize() -> float:
    fs = plt.rcParams.get('legend.fontsize', 10)
    try:
        return float(fs) * _LEGEND_SCALE
    except Exception:
        mapping = {
            'xx-small': 6,
            'x-small': 8,
            'small': 10,
            'medium': 12,
            'large': 14,
            'x-large': 16,
            'xx-large': 18,
        }
        base = mapping.get(str(fs).lower(), 12)
        return base * _LEGEND_SCALE

def run_sobol_sensitivity_analysis():
    """
    Execute global sensitivity analysis based on Sobol method
    Analyze three parameters: T1/T2 coherence time, entanglement fidelity, and noise rate
    Using log-transformed sampling for all parameters
    """
    # Define the problem
    problem = {
        'num_vars': 3,
        'names': ['coherence_time', 'entanglement_fidelity', 'noise_rate'],
        # 'bounds': [[np.log10(1e11), np.log10(1e13)],      # T1 coherence time (log10 scale)
        #           [np.log10(0.0001), np.log10(0.01)],       # Entanglement fidelity (log10 scale)
        #           [np.log10(0.000001), np.log10(0.0001)]]   # Noise rate (log10 scale)
        'bounds': [[np.log10(1e10), np.log10(1e12)],      # T1 coherence time (log10 scale)
                  [np.log10(0.001), np.log10(0.1)],       # Entanglement fidelity (log10 scale)
                  [np.log10(0.00001), np.log10(0.001)]]   # Noise rate (log10 scale)
    }
    # 'bounds': [[np.log10(1e11), np.log10(1e13)],      # T1 coherence time (log10 scale)
    #               [np.log10(0.99), np.log10(0.9999)],       # Entanglement fidelity (log10 scale)
    #               [np.log10(0.000001), np.log10(0.0001)]]

    # Generate samples using Sobol method
    # N=128: Basic sample size (actual number is N*(2D+2))
    N = 64
    param_values_log = sobol_sample(problem, N, calc_second_order=True)
    print(f"Number of parameter combinations sampled: {param_values_log.shape[0]}")
    
    # Convert log-transformed values back to original scale
    param_values = 10**param_values_log
    
    # Common parameter settings (fixed parameters)
    distances = [1000]               # km
    T2_ratios = [0.1]               # Ratio to T1
    entanglement_speed_factors = [100]
    gate_speed_factors = [1]
    client_gate_speed_factors = [1]
    fidelity_factors = [1]
    num_runs = 5                    # Set low to reduce computational cost
    shots = 500
    flag = 0                        # ZZ circuit
    tol = 0.001
    bounds = (-np.pi, np.pi)
    target_energy = -1.1615         # Target ground state energy for calculating error

    # Lists to store results
    energy_errors = []
    all_results = []

    # Run simulation for each parameter combination
    for i, params in enumerate(param_values):
        coherence_time = params[0]
        # entanglement_fidelity = params[1]
        entanglement_error = params[1]       # エラー率として取得
        entanglement_fidelity = 1 - entanglement_error   # fidelityに変換
        noise_rate = params[2]
        
        # Set parameters with the same values as specified
        T1 = coherence_time
        client_T1 = coherence_time
        dephase_rate = noise_rate
        client_fidelity = noise_rate
        sge = noise_rate
        dge = noise_rate
        
        print(f"\nシミュレーション {i+1}/{len(param_values)}:")
        print(f"coherence_time={coherence_time:.2e} ns, entanglement_fidelity={entanglement_fidelity:.6f}, "
              f"noise_rate={noise_rate:.6f}")

        # Run VQE optimization experiment
        result_df = run_vqe_optimization_experiment(
            fidelity_factors=fidelity_factors,
            num_runs=num_runs,
            dephase_rates=[dephase_rate],
            client_fidelitys=[client_fidelity],
            distances=distances,
            T1s=[T1],
            T2_ratios=T2_ratios,
            client_T1s=[client_T1],
            sges=[sge],
            dges=[dge],
            gate_speed_factors=gate_speed_factors,
            client_gate_speed_factors=client_gate_speed_factors,
            entanglement_fidelities=[entanglement_fidelity],
            entanglement_speed_factors=entanglement_speed_factors,
            shots=shots,
            flag=flag,
            tol=tol,
            bounds=bounds
        )
        
        # Extract final energy and calculate error
        final_energy = result_df['final_energy'].values[0]
        energy_error = abs(final_energy - target_energy)
        energy_errors.append(energy_error)
        
        # Save all results
        result_row = {
            'coherence_time': coherence_time,
            'entanglement_fidelity': 1-entanglement_fidelity,
            'noise_rate': noise_rate,
            'final_energy': final_energy,
            'energy_error': energy_error
        }
        all_results.append(result_row)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    # Create output folder
    output_dir = 'energy_now'
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f'{output_dir}/energy_now.csv', index=False)
    print("シミュレーション結果をCSVに保存しました。")
    
    # For Sobol analysis, we need to pass in param_values_log (log-scale values)
    # to match the problem definition with log-scale bounds
    Si = sobol.analyze(problem, np.array(energy_errors), calc_second_order=True, print_to_console=True)
    
    # Convert sensitivity analysis results to DataFrame
    Si_df = pd.DataFrame({
        'Parameter': problem['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf']
    })
    
    # Sensitivity analysis for different coherence time ranges
    coherence_ranges = [(1e10, 1e11), (1e11, 5e11), (5e11, 1e12)]
    range_results = []
    
    for c_min, c_max in coherence_ranges:
        print(f"\nコヒーレンス時間範囲 {c_min:.1e}-{c_max:.1e} nsの分析:")
        # Filter data for specific coherence time range
        range_df = results_df[(results_df['coherence_time'] >= c_min) & (results_df['coherence_time'] <= c_max)]
        
        if len(range_df) < 10:
            print(f"警告: コヒーレンス時間範囲 {c_min:.1e}-{c_max:.1e} nsのサンプル数が不足しています: {len(range_df)}")
            continue
            
        # Run multivariate regression for this range
        # Using standardized regression coefficients as a simple method
        
        # Standardize data on log scale
        X = pd.DataFrame({
            'coherence_time': np.log10(range_df['coherence_time']),
            'entanglement_fidelity': np.log10(range_df['entanglement_fidelity']),
            'noise_rate': np.log10(range_df['noise_rate'])
        })
        y = np.log10(range_df['energy_error'])
        
        X_mean = X.mean()
        X_std = X.std()
        y_mean = y.mean()
        y_std = y.std()
        
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        
        # Linear regression using least squares
        model = LinearRegression()
        model.fit(X_norm, y_norm)
        
        # Standardized regression coefficients (sensitivity coefficients)
        coefficients = model.coef_
        
        # Contribution rates (decomposition of R² determination coefficient)
        # Note: This is a simplified method, not a complete Sobol index
        contrib = coefficients ** 2
        contrib = contrib / np.sum(contrib)
        
        range_result = {
            'coherence_range': f"{c_min:.1e}-{c_max:.1e}",
            'coherence_sensitivity': abs(coefficients[0]),
            'entanglement_sensitivity': abs(coefficients[1]),
            'noise_sensitivity': abs(coefficients[2]),
            'coherence_contribution': contrib[0],
            'entanglement_contribution': contrib[1],
            'noise_contribution': contrib[2],
            'sample_size': len(range_df)
        }
        range_results.append(range_result)
    
    # Convert results by coherence range to DataFrame
    range_df = pd.DataFrame(range_results)
    print("\nコヒーレンス時間範囲ごとの感度分析結果:")
    print(range_df)
    
    # Visualize results
    plot_sensitivity_results(Si, problem['names'], output_dir)
    plot_range_sensitivity(range_df, output_dir)
    plot_3d_surface(results_df, output_dir)
    
    return Si, Si_df, range_df, results_df

def plot_sensitivity_results(Si, param_names, output_dir):
    """Function to visualize Sobol sensitivity indices"""
    plt.figure(figsize=(10, 6))
    
    # Plot first-order (S1) and total-order (ST) sensitivity indices
    width = 0.35
    indices = np.arange(len(param_names))
    
    plt.bar(indices - width/2, Si['S1'], width, label='一次感度 (S1)')
    plt.bar(indices + width/2, Si['ST'], width, label='全次感度 (ST)')
    
    plt.xticks(indices, param_names)
    plt.xlabel('パラメータ')
    plt.ylabel('Sobol感度指標')
    plt.title('パラメータごとのSobol感度指標')
    plt.legend(fontsize=_legend_fontsize())
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/sobol_sensitivity_indices.png', dpi=300)

def plot_range_sensitivity(range_df, output_dir):
    """Function to visualize sensitivity coefficients by coherence time range"""
    if len(range_df) == 0:
        print("警告: コヒーレンス時間範囲の結果がありません。グラフは作成されません。")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot sensitivity coefficients
    param_names = ['コヒーレンス時間', 'エンタングルメント忠実度', 'ノイズ率']
    colors = ['#8884d8', '#82ca9d', '#ffc658']
    
    x = np.arange(len(range_df))
    width = 0.25
    
    for i, param in enumerate(['coherence_sensitivity', 'entanglement_sensitivity', 'noise_sensitivity']):
        plt.bar(x + (i-1)*width, range_df[param], width, label=f'{param_names[i]}感度', color=colors[i])
    
    plt.xlabel('コヒーレンス時間範囲 (ns)')
    plt.ylabel('感度係数 (正規化)')
    plt.title('コヒーレンス時間範囲ごとのパラメータ感度係数')
    plt.xticks(x, range_df['coherence_range'])
    plt.legend(fontsize=_legend_fontsize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_sensitivity.png', dpi=300)
    
    # Plot contribution rates
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stacked bar chart
    contribution_data = range_df[['coherence_contribution', 'entanglement_contribution', 'noise_contribution']].copy()
    
    # Create stacked bar chart
    contribution_data.plot(kind='bar', stacked=True, figsize=(12, 6), 
                          color=colors,
                          width=0.7)
    
    plt.xlabel('コヒーレンス時間範囲 (ns)')
    plt.ylabel('寄与率')
    plt.title('コヒーレンス時間範囲ごとのパラメータ寄与率')
    plt.xticks(x, range_df['coherence_range'], rotation=0)
    plt.legend(['コヒーレンス時間', 'エンタングルメント忠実度', 'ノイズ率'], fontsize=_legend_fontsize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_contribution.png', dpi=300)

def plot_3d_surface(results_df, output_dir):
    """Function to create 3D surface plots of energy error vs parameters"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # First surface plot: Coherence Time vs Entanglement Fidelity
    grid_size = 20
    coherence_grid = np.logspace(np.log10(1e10), np.log10(1e12), grid_size)
    entanglement_grid = np.logspace(np.log10(0.9), np.log10(0.999), grid_size)
    C, E = np.meshgrid(coherence_grid, entanglement_grid)
    
    # Fix noise rate at median value
    noise_rate = results_df['noise_rate'].median()
    
    # Build linear regression model on log-transformed data
    X_train = pd.DataFrame({
        'coherence_time': np.log10(results_df['coherence_time']),
        'entanglement_fidelity': np.log10(results_df['entanglement_fidelity']),
        'noise_rate': np.log10(results_df['noise_rate'])
    })
    
    y_train = np.log10(results_df['energy_error'])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict energy error on grid
    Z = np.zeros_like(C)
    for i in range(grid_size):
        for j in range(grid_size):
            c = coherence_grid[j]
            e = entanglement_grid[i]
            X_pred = pd.DataFrame({
                'coherence_time': [np.log10(c)],
                'entanglement_fidelity': [np.log10(e)],
                'noise_rate': [np.log10(noise_rate)]
            })
            Z[i, j] = 10 ** model.predict(X_pred)[0]
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(C, E, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('コヒーレンス時間 (ns)')
    ax.set_ylabel('エンタングルメント忠実度')
    ax.set_zlabel('エネルギー誤差')
    
    ax.set_xscale('log')
    ax.set_title('コヒーレンス時間とエンタングルメント忠実度がエネルギー誤差に与える影響')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_entanglement.png', dpi=300)
    
    # Second surface plot: Coherence Time vs Noise Rate
    noise_grid = np.logspace(np.log10(0.00001), np.log10(0.001), grid_size)
    C, N = np.meshgrid(coherence_grid, noise_grid)
    
    # Fix entanglement fidelity at median value
    entanglement_fidelity = results_df['entanglement_fidelity'].median()
    
    # Predict energy error on grid
    Z = np.zeros_like(C)
    for i in range(grid_size):
        for j in range(grid_size):
            c = coherence_grid[j]
            n = noise_grid[i]
            X_pred = pd.DataFrame({
                'coherence_time': [np.log10(c)],
                'entanglement_fidelity': [np.log10(entanglement_fidelity)],
                'noise_rate': [np.log10(n)]
            })
            Z[i, j] = 10 ** model.predict(X_pred)[0]
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(C, N, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('コヒーレンス時間 (ns)')
    ax.set_ylabel('ノイズ率')
    ax.set_zlabel('エネルギー誤差')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('コヒーレンス時間とノイズ率がエネルギー誤差に与える影響')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_noise.png', dpi=300)

# -*- coding: utf-8 -*-
"""
local_sobol_segment_analysis.py
--------------------------------
区間を先に定義してから、その区間ごとに Saltelli サンプリング→VQE 実験→Sobol 感度を計算し、
`visualize_3d_segments()` がそのまま読める形式（simple_segment_results.csv）で保存するスクリプト。

📌 既存の実験関数
    * run_vqe_optimization_experiment() : DataFrame を返す（final_energy 列必須）

🔧 使い方
    $ python local_sobol_segment_analysis.py  # スクリプト単体実行

生成物
    - results/local_simple_segment_results.csv
    - 同じディレクトリにフル結果やログも保存
"""

import os
import time
import numpy as np
import pandas as pd
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression  # fallback 用

# -------------------------------------------------------------
# ★ ユーザが実装済みの関数を import してください ★
# from your_module import run_vqe_optimization_experiment
# -------------------------------------------------------------

# ------------------------ ユーティリティ ---------------------

def _midpoint_log10(low: float, high: float) -> float:
    """対数スケール範囲 [low, high] の幾何平均を返す"""
    return 10 ** ((np.log10(low) + np.log10(high)) / 2)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

import numpy as np
from typing import Sequence, Tuple, Union, List

Number = Union[int, float]

def make_log_bins(
    value_range: Tuple[Number, Number],
    n_segments: int,
    *,
    base: float = 10.0,
    include_max: bool = True,
) -> List[float]:
    """
    指定した範囲を対数スケールで n_segments 等分し、ビン境界を返す。

    Parameters
    ----------
    value_range : (min_val, max_val)
        ビン分割したい数値の（実数）範囲。 min_val > 0 である必要がある。
    n_segments : int
        生成する“区間”の数。境界点は n_segments + 1 個返る。
    base : float, default 10.0
        対数を取るときの底。自然対数でよければ np.e を指定。
    include_max : bool, default True
        True なら最大値を最後の境界点に含める。
        False の場合は最大値を超えない最小の境界点を返す。

    Returns
    -------
    List[float]
        境界点の昇順リスト（長さ n_segments+1）。
    """
    min_val, max_val = map(float, value_range)
    if min_val <= 0 or max_val <= 0:
        raise ValueError("min_val と max_val は共に 0 より大きい必要があります。")
    if min_val >= max_val:
        raise ValueError("min_val は max_val より小さくなければいけません。")
    if n_segments < 1:
        raise ValueError("n_segments は 1 以上の整数にしてください。")

    # log_{base}(value) 空間で等差に区切る
    log_min = np.log(min_val) / np.log(base)
    log_max = np.log(max_val) / np.log(base)

    # np.linspace で n_segments 等分 => n_segments+1 個
    log_edges = np.linspace(log_min, log_max, n_segments + 1)

    if not include_max:
        # 最大値を超えない最後の境界まで取得
        log_edges = log_edges[:-1]

    # base^{log_edges} -> 元のスケールへ
    return list(base ** log_edges)


# ------------------ メイン関数：区間 Sobol -------------------

def run_local_sobol_segment_analysis(
    ct_range=(1e11, 1e13),        # ← ここは範囲だけを渡す
    fid_range=(1e-3, 1e-2),
    noise_range=(1e-3, 1e-1),
    n_ct_seg: int = 3,            # ← “区間数”を引数に
    n_fid_seg: int = 3,
    n_noise_seg: int = 3,
    N_local: int = 16,
    target_energy: float = -1.1615,
    output_dir: str = "59results",
    shots: int = 100,
    num_runs: int = 5,
    flag: int = 0,
):
    """区間ごとに Sobol 感度を計算し simple_segment_results.csv を出力"""
    ct_bins    = make_log_bins(ct_range,    n_ct_seg)
    fid_bins   = make_log_bins(fid_range,   n_fid_seg)
    noise_bins = make_log_bins(noise_range, n_noise_seg)

    start = time.perf_counter()

    # 出力準備
    _ensure_dir(output_dir)
    simple_rows = []

    # ループ用の境界リスト
    ct_bins = list(ct_bins)
    fid_bins = list(fid_bins)
    noise_bins = list(noise_bins)

    total_segments = (len(ct_bins)-1) * (len(fid_bins)-1) * (len(noise_bins)-1)
    seg_counter = 0

    for i1 in range(len(ct_bins) - 1):
        for i2 in range(len(fid_bins) - 1):
            for i3 in range(len(noise_bins) - 1):
                seg_counter += 1
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(f"[{seg_counter}/{total_segments}] {seg_name} running…")

                # --- 区間 bounds を log10 で設定 ---
                bounds_log10 = [
                    [np.log10(ct_bins[i1]),   np.log10(ct_bins[i1+1])],
                    [np.log10(fid_bins[i2]),  np.log10(fid_bins[i2+1])],
                    [np.log10(noise_bins[i3]), np.log10(noise_bins[i3+1])],
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

                # Saltelli サンプリング（第二次まで）
                X_log = sobol_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                energy_errors = []

                # ---------- VQE 実験を実行 ----------
                # ---------- VQE 実験を実行 ----------
                for row in X:
                    # Saltelli で得た 3 パラメータ
                    ct, fid_error, noise = row          # coherence_time, 1-fidelity, noise_rate
                    ent_fid = 1.0 - fid_error           # 誤り率 → 忠実度
                    sges=0.0006*noise                     # 1qubit ゲート誤り率（固定値）
                    dges=0.006*noise  
                    dephase_rates=[0.0039*noise]
                    # --- 本物の VQE 最適化を 1 ケースだけ回す -----------------
                    df = run_vqe_optimization_experiment(
                        fidelity_factors=[1],            # ← ゲート誤り率スケールを変えないなら 1 固定
                        num_runs=num_runs,               # ex. 5
                        dephase_rates=[0.0039*noise],           # メモリ dephase 率
                        client_fidelitys=[1],        # Bob 側ゲート誤り率も同値で渡す例
                        distances=[500],                # [km]：固定値で良ければ 1 要素リスト
                        T1s=[ct],                        # Alice T1
                        T2_ratios=[0.1],                 # T2/T1 比
                        client_T1s=[ct],                 # Bob T1 = Alice と同じにする例
                        sges=[0.0006*noise],                  # 1qubit ゲート誤り率
                        dges=[0.006*noise],                 # 2qubit ゲート誤り率
                        gate_speed_factors=[1],          # ゲート速度スケール
                        client_gate_speed_factors=[1],   # Bob 側も同じ
                        entanglement_fidelities=[ent_fid],       # 忠実度
                        entanglement_speed_factors=[100],        # 生成速度スケール
                        shots=shots,                     # ex. 500
                        flag=flag,                       # 0: ZZ, 1: XX
                        tol=0.015,                      # 最適化の収束閾値
                        bounds=(-np.pi, np.pi),          # θ の探索範囲
                    )


                    # run_vqe_optimization_experiment は 1 行の DataFrame を返す
                    final_energy = df.at[0, "final_energy"]

                    # エラー＝|E_final − E_target|
                    energy_errors.append(abs(final_energy - target_energy))


                # Sobol 感度計算
                Si = sobol.analyze(problem, np.array(energy_errors), calc_second_order=False)

                # 幾何中心点（ログミッド）
                mid_ct = _midpoint_log10(ct_bins[i1], ct_bins[i1 + 1])
                mid_fid = _midpoint_log10(fid_bins[i2], fid_bins[i2 + 1])
                mid_noise = _midpoint_log10(noise_bins[i3], noise_bins[i3 + 1])

                # 可視化用行を構築
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

    # -------------------- CSV 出力 --------------------
    simple_df = pd.DataFrame(simple_rows)
    path_simple = os.path.join(output_dir, "new_energy_future.csv")
    simple_df.to_csv(path_simple, index=False)

    elapsed = time.perf_counter() - start
    print(f"\n完了: {path_simple} に書き出しました  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


# ---------------- スクリプト実行 -------------------
if __name__ == "__main__":
    import time

    t0 = time.perf_counter()            # ← 計測開始
    path_simple, _ = run_local_sobol_segment_analysis()   # 関数実行
    elapsed = time.perf_counter() - t0  # ← 経過時間

    print(f"\n===== 全体の実行時間: {elapsed:.1f} 秒 =====")
