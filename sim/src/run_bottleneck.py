from qs.simulation import run_vqe_optimization_experiment
import numpy as np
import os


BASE_PARAMS = {
    "fidelity_factors": [1.0],
    "num_runs": 10,
    "dephase_rates": [0.0039],
    "client_fidelitys": [0.0],
    "distances": [500],
    "T1s": [1e11],
    "T2_ratios": [0.1],
    "client_T1s": [1e11],
    "sges": [0.0006],
    "dges": [0.006],
    "gate_speed_factors": [1.0],
    "client_gate_speed_factors": [1.0],
    "entanglement_fidelities": [0.99],
    "entanglement_speed_factors": [100],
    "shots": 100,
    "flag": 0,
}


def run_and_save(name, param, values):
    params = BASE_PARAMS.copy()
    params[param] = values
    df = run_vqe_optimization_experiment(**params)
    print(df)
    os.makedirs("bottleneck", exist_ok=True)
    df.to_csv(f"bottleneck/{name}.csv", index=False)


def main():
    experiments = [
        ("server_fidelity_scale", "fidelity_factors", [10, 1, 0.1, 0.01]),
        ("server_time_scale", "gate_speed_factors", [0.1, 1.0, 10.0, 100.0]),
        ("server_T1", "T1s", [1e11, 1e10, 1e9, 1e8]),
        (
            "client_fidelity_scale",
            "client_fidelitys",
            [0.039, 0.0039, 0.00039, 0.000039],
        ),
        (
            "client_time_scale",
            "client_gate_speed_factors",
            [0.1, 1.0, 10.0, 100.0],
        ),
        ("client_T1", "client_T1s", [1e11, 1e10, 1e9, 1e8]),
        (
            "network_bandwidth",
            "entanglement_speed_factors",
            list(np.logspace(np.log10(10), np.log10(10000), num=4)),
        ),
        (
            "network_distance",
            "distances",
            list(np.logspace(np.log10(10), np.log10(10000), num=4)),
        ),
        (
            "network_fidelity",
            "entanglement_fidelities",
            [0.9999, 0.999, 0.99, 0.9],
        ),
    ]

    for name, param, values in experiments:
        run_and_save(name, param, values)


if __name__ == "__main__":
    main()
