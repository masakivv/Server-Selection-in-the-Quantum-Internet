from qs.simulation import run_vqe_optimization_experiment


import numpy as np
import pandas as pd
import os
from datetime import datetime


results_dir = "req_data"
os.makedirs(results_dir, exist_ok=True)


if __name__ == "__main__":

    num_runs = 10
    shots = 1000
    flag = 0  
    tol = 0.01
    bounds = (-np.pi, np.pi)
    T2_ratios = [0.1]  


    dephase_rates = [0.00]
    client_fidelitys = [0]
    fidelity_factors = [1]
    distances = [500]             
    T1s = [1e50]
    client_T1s = np.logspace(7, 11, 10).tolist()  
    sges = [0]                    
    dges = [0]                    
    gate_speed_factors = [1.0]    
    client_gate_speed_factors = [1.0]
    entanglement_fidelities = [1]
    entanglement_speed_factors = [100]  

    print("Experiment 6: starting client T1 experiment")
    df_result6 = run_vqe_optimization_experiment(
        num_runs=num_runs,
        dephase_rates=dephase_rates,
        client_fidelitys=client_fidelitys,
        fidelity_factors=fidelity_factors,
        distances=distances,
        T1s=T1s,
        client_T1s=client_T1s,
        T2_ratios=T2_ratios,
        sges=sges,
        dges=dges,
        gate_speed_factors=gate_speed_factors,
        client_gate_speed_factors=client_gate_speed_factors,
        entanglement_fidelities=entanglement_fidelities,
        entanglement_speed_factors=entanglement_speed_factors,
        shots=shots,
        flag=flag,
        tol=tol,
        bounds=bounds
    )
    df_result6.to_csv(f"{results_dir}/client_T1s_experiment.csv", index=False)
    print("Experiment 6 complete: client T1")


    print("Experiment 9 complete: gate speed factor")


    print("All experiments are complete!")
