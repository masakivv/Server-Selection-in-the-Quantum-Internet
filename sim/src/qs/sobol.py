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

    problem = {
        'num_vars': 3,
        'names': ['coherence_time', 'entanglement_fidelity', 'noise_rate'],


        'bounds': [[np.log10(1e10), np.log10(1e12)],      
                  [np.log10(0.001), np.log10(0.1)],       
                  [np.log10(0.00001), np.log10(0.001)]]   
    }


    N = 64
    param_values_log = sobol_sample(problem, N, calc_second_order=True)
    print(f"Number of parameter combinations sampled: {param_values_log.shape[0]}")


    param_values = 10**param_values_log


    distances = [1000]               
    T2_ratios = [0.1]               
    entanglement_speed_factors = [100]
    gate_speed_factors = [1]
    client_gate_speed_factors = [1]
    fidelity_factors = [1]
    num_runs = 5                    
    shots = 500
    flag = 0                        
    tol = 0.001
    bounds = (-np.pi, np.pi)
    target_energy = -1.1615         


    energy_errors = []
    all_results = []


    for i, params in enumerate(param_values):
        coherence_time = params[0]

        entanglement_error = params[1]       
        entanglement_fidelity = 1 - entanglement_error   
        noise_rate = params[2]


        T1 = coherence_time
        client_T1 = coherence_time
        dephase_rate = noise_rate
        client_fidelity = noise_rate
        sge = noise_rate
        dge = noise_rate

        print(f"\nSimulation {i+1}/{len(param_values)}:")
        print(f"coherence_time={coherence_time:.2e} ns, entanglement_fidelity={entanglement_fidelity:.6f}, "
              f"noise_rate={noise_rate:.6f}")


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


        final_energy = result_df['final_energy'].values[0]
        energy_error = abs(final_energy - target_energy)
        energy_errors.append(energy_error)


        result_row = {
            'coherence_time': coherence_time,
            'entanglement_fidelity': 1-entanglement_fidelity,
            'noise_rate': noise_rate,
            'final_energy': final_energy,
            'energy_error': energy_error
        }
        all_results.append(result_row)


    results_df = pd.DataFrame(all_results)


    output_dir = 'energy_now'
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(f'{output_dir}/energy_now.csv', index=False)
    print("Saved simulation results to CSV.")


    Si = sobol.analyze(problem, np.array(energy_errors), calc_second_order=True, print_to_console=True)


    Si_df = pd.DataFrame({
        'Parameter': problem['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf']
    })


    coherence_ranges = [(1e10, 1e11), (1e11, 5e11), (5e11, 1e12)]
    range_results = []

    for c_min, c_max in coherence_ranges:
        print(f"\nAnalyzing coherence-time range {c_min:.1e}-{c_max:.1e} ns:")

        range_df = results_df[(results_df['coherence_time'] >= c_min) & (results_df['coherence_time'] <= c_max)]

        if len(range_df) < 10:
            print(f"Warning: insufficient samples for coherence-time range {c_min:.1e}-{c_max:.1e} ns: {len(range_df)}")
            continue


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


        model = LinearRegression()
        model.fit(X_norm, y_norm)


        coefficients = model.coef_


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


    range_df = pd.DataFrame(range_results)
    print("\nSensitivity analysis results by coherence-time range:")
    print(range_df)


    plot_sensitivity_results(Si, problem['names'], output_dir)
    plot_range_sensitivity(range_df, output_dir)
    plot_3d_surface(results_df, output_dir)

    return Si, Si_df, range_df, results_df

def plot_sensitivity_results(Si, param_names, output_dir):
    """Function to visualize Sobol sensitivity indices"""
    plt.figure(figsize=(10, 6))


    width = 0.35
    indices = np.arange(len(param_names))

    plt.bar(indices - width/2, Si['S1'], width, label='First-order sensitivity (S1)')
    plt.bar(indices + width/2, Si['ST'], width, label='Total-order sensitivity (ST)')

    plt.xticks(indices, param_names)
    plt.xlabel('Parameters')
    plt.ylabel('Sobol sensitivity index')
    plt.title('Sobol sensitivity index by parameter')
    plt.legend(fontsize=_legend_fontsize())
    plt.tight_layout()

    plt.savefig(f'{output_dir}/sobol_sensitivity_indices.png', dpi=300)

def plot_range_sensitivity(range_df, output_dir):
    """Function to visualize sensitivity coefficients by coherence time range"""
    if len(range_df) == 0:
        print("Warning: no coherence-time range results. No graph will be generated.")
        return

    plt.figure(figsize=(12, 6))


    param_names = ['Coherence time', 'Entanglement fidelity', 'Noise rate']
    colors = ['#8884d8', '#82ca9d', '#ffc658']

    x = np.arange(len(range_df))
    width = 0.25

    for i, param in enumerate(['coherence_sensitivity', 'entanglement_sensitivity', 'noise_sensitivity']):
        plt.bar(x + (i-1)*width, range_df[param], width, label=f'{param_names[i]} sensitivity', color=colors[i])

    plt.xlabel('Coherence-time range (ns)')
    plt.ylabel('Sensitivity coefficient (normalized)')
    plt.title('Parameter sensitivity coefficients by coherence-time range')
    plt.xticks(x, range_df['coherence_range'])
    plt.legend(fontsize=_legend_fontsize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_sensitivity.png', dpi=300)


    plt.figure(figsize=(12, 6))


    contribution_data = range_df[['coherence_contribution', 'entanglement_contribution', 'noise_contribution']].copy()


    contribution_data.plot(kind='bar', stacked=True, figsize=(12, 6), 
                          color=colors,
                          width=0.7)

    plt.xlabel('Coherence-time range (ns)')
    plt.ylabel('Contribution ratio')
    plt.title('Parameter contribution ratio by coherence-time range')
    plt.xticks(x, range_df['coherence_range'], rotation=0)
    plt.legend(['Coherence time', 'Entanglement fidelity', 'Noise rate'], fontsize=_legend_fontsize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_contribution.png', dpi=300)

def plot_3d_surface(results_df, output_dir):
    """Function to create 3D surface plots of energy error vs parameters"""
    from mpl_toolkits.mplot3d import Axes3D


    grid_size = 20
    coherence_grid = np.logspace(np.log10(1e10), np.log10(1e12), grid_size)
    entanglement_grid = np.logspace(np.log10(0.9), np.log10(0.999), grid_size)
    C, E = np.meshgrid(coherence_grid, entanglement_grid)


    noise_rate = results_df['noise_rate'].median()


    X_train = pd.DataFrame({
        'coherence_time': np.log10(results_df['coherence_time']),
        'entanglement_fidelity': np.log10(results_df['entanglement_fidelity']),
        'noise_rate': np.log10(results_df['noise_rate'])
    })

    y_train = np.log10(results_df['energy_error'])

    model = LinearRegression()
    model.fit(X_train, y_train)


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


    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(C, E, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)

    ax.set_xlabel('Coherence time (ns)')
    ax.set_ylabel('Entanglement fidelity')
    ax.set_zlabel('Energy error')

    ax.set_xscale('log')
    ax.set_title('Impact of coherence time and entanglement fidelity on energy error')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_entanglement.png', dpi=300)


    noise_grid = np.logspace(np.log10(0.00001), np.log10(0.001), grid_size)
    C, N = np.meshgrid(coherence_grid, noise_grid)


    entanglement_fidelity = results_df['entanglement_fidelity'].median()


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


    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(C, N, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)

    ax.set_xlabel('Coherence time (ns)')
    ax.set_ylabel('Noise rate')
    ax.set_zlabel('Energy error')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Impact of coherence time and noise rate on energy error')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_noise.png', dpi=300)



import os
import time
import numpy as np
import pandas as pd
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression  


def _midpoint_log10(low: float, high: float) -> float:
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
    fid_range=(1e-3, 1e-2),
    noise_range=(1e-3, 1e-1),
    n_ct_seg: int = 3,            
    n_fid_seg: int = 3,
    n_noise_seg: int = 3,
    N_local: int = 16,
    target_energy: float = -1.1615,
    output_dir: str = "59results",
    shots: int = 100,
    num_runs: int = 5,
    flag: int = 0,
):
    ct_bins    = make_log_bins(ct_range,    n_ct_seg)
    fid_bins   = make_log_bins(fid_range,   n_fid_seg)
    noise_bins = make_log_bins(noise_range, n_noise_seg)

    start = time.perf_counter()


    _ensure_dir(output_dir)
    simple_rows = []


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


                X_log = sobol_sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                energy_errors = []


                for row in X:

                    ct, fid_error, noise = row          
                    ent_fid = 1.0 - fid_error           
                    sges=0.0006*noise                     
                    dges=0.006*noise  
                    dephase_rates=[0.0039*noise]

                    df = run_vqe_optimization_experiment(
                        fidelity_factors=[1],            
                        num_runs=num_runs,               
                        dephase_rates=[0.0039*noise],           
                        client_fidelitys=[1],        
                        distances=[500],                
                        T1s=[ct],                        
                        T2_ratios=[0.1],                 
                        client_T1s=[ct],                 
                        sges=[0.0006*noise],                  
                        dges=[0.006*noise],                 
                        gate_speed_factors=[1],          
                        client_gate_speed_factors=[1],   
                        entanglement_fidelities=[ent_fid],       
                        entanglement_speed_factors=[100],        
                        shots=shots,                     
                        flag=flag,                       
                        tol=0.015,                      
                        bounds=(-np.pi, np.pi),          
                    )


                    final_energy = df.at[0, "final_energy"]


                    energy_errors.append(abs(final_energy - target_energy))


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
    path_simple = os.path.join(output_dir, "new_energy_future.csv")
    simple_df.to_csv(path_simple, index=False)

    elapsed = time.perf_counter() - start
    print(f"\nDone: wrote to {path_simple}  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()            
    path_simple, _ = run_local_sobol_segment_analysis()   
    elapsed = time.perf_counter() - t0  

    print(f"\n===== Total runtime: {elapsed:.1f} s =====")
