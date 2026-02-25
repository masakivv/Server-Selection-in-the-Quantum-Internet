from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np

from qs.density import (
    DEFAULT_CONFIG,
    SimulationConfig,
    compute_expectations,
    run_density as run_density_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQE run using density matrices for expectation values"
    )
    parser.add_argument("--theta", type=float, help="Rotation angle θ to evaluate [rad]")
    parser.add_argument("--optimize", action="store_true", help="Optimize θ")
    parser.add_argument(
        "--random-theta",
        action="store_true",
        help="Sample θ randomly from [-π, π] (when optimize/--theta are not specified)",
    )
    parser.add_argument(
        "--theta-sweep",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="θ sweep settings (degrees). Example: --theta-sweep 0 180 10",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--seed-sweep",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Seed sweep settings (integers). Example: --seed-sweep 0 4 1",
    )
    parser.add_argument(
        "--seed-output",
        type=str,
        help="Path to save density matrices for seed sweeps or single runs",
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_CONFIG.num_runs)
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_CONFIG.shots,
        help="Number of shots to execute in the simulation (1 is enough when using density matrices)",
    )
    parser.add_argument("--dephase-rate", type=float, default=DEFAULT_CONFIG.dephase_rate)
    parser.add_argument(
        "--client-fidelity",
        type=float,
        default=DEFAULT_CONFIG.client_fidelity,
    )
    parser.add_argument(
        "--distance", type=float, default=DEFAULT_CONFIG.distance, help="Distance [km]"
    )
    parser.add_argument("--T1", type=float, default=DEFAULT_CONFIG.T1, help="Server-side T1 [ns]")
    parser.add_argument(
        "--client-T1", type=float, default=DEFAULT_CONFIG.client_T1, help="Client-side T1 [ns]"
    )
    parser.add_argument("--T2-ratio", type=float, default=DEFAULT_CONFIG.T2_ratio, help="T2/T1 ratio")
    parser.add_argument("--sge", type=float, default=DEFAULT_CONFIG.sge, help="Single-qubit gate error rate")
    parser.add_argument("--dge", type=float, default=DEFAULT_CONFIG.dge, help="Two-qubit gate error rate")
    parser.add_argument("--gate-speed-factor", type=float, default=DEFAULT_CONFIG.gate_speed_factor)
    parser.add_argument(
        "--client-gate-speed-factor",
        type=float,
        default=DEFAULT_CONFIG.client_gate_speed_factor,
    )
    parser.add_argument(
        "--entanglement-fidelity",
        type=float,
        default=DEFAULT_CONFIG.entanglement_fidelity,
    )
    parser.add_argument(
        "--entanglement-speed-factor",
        type=float,
        default=DEFAULT_CONFIG.entanglement_speed_factor,
    )
    parser.add_argument(
        "--cc-jitter-mean-ns",
        type=float,
        default=DEFAULT_CONFIG.cc_jitter_mean_ns,
        help="Mean value of exponential jitter added to classical channel delay [ns] (disabled with 0)",
    )
    parser.add_argument(
        "--ext-stochastic",
        action="store_true",
        help="Use exponentially sampled trigger intervals for the external entanglement source (default: fixed interval)",
    )
    parser.add_argument(
        "--ext-seed",
        type=int,
        default=None,
        help="RNG seed for external entanglement source (default: base seed + 200002 if omitted)",
    )
    parser.add_argument(
        "--ext-min-delay-ns",
        type=float,
        default=DEFAULT_CONFIG.ext_min_delay_ns,
        help="Minimum delay applied to exponential samples [ns] (prevents zero-wait loops)",
    )
    parser.add_argument(
        "--classical-rng-seed-offset",
        type=int,
        default=100000,
        help="Offset added to base seed for classical-channel RNGs (shared by ZZ/XX); used to generate rng_ab/rng_ba",
    )
    parser.add_argument("--tol", type=float, default=0.0015)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        default=[-math.pi, math.pi],
        metavar=("LOW", "HIGH"),
        help="θ search bounds (when optimize is enabled)",
    )
    return parser.parse_args()


def _write_density_block(handle, label: str, matrix) -> None:
    handle.write(f"{label}=\n")
    handle.write(np.array2string(matrix, precision=6, suppress_small=True))
    handle.write("\n")


def _build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        num_runs=args.num_runs,
        shots=args.shots,
        dephase_rate=args.dephase_rate,
        client_fidelity=args.client_fidelity,
        distance=args.distance,
        T1=args.T1,
        client_T1=args.client_T1,
        T2_ratio=args.T2_ratio,
        sge=args.sge,
        dge=args.dge,
        gate_speed_factor=args.gate_speed_factor,
        client_gate_speed_factor=args.client_gate_speed_factor,
        entanglement_fidelity=args.entanglement_fidelity,
        entanglement_speed_factor=args.entanglement_speed_factor,
        cc_jitter_mean_ns=args.cc_jitter_mean_ns,
        ext_stochastic=args.ext_stochastic,
        ext_seed=args.ext_seed,
        ext_min_delay_ns=args.ext_min_delay_ns,
    )


def main() -> None:
    args = parse_args()
    config = _build_config(args)

    bounds: Tuple[float, float] = tuple(args.bounds)
    seed_value = args.seed
    rng_ab_seed = seed_value + args.classical_rng_seed_offset
    rng_ba_seed = seed_value + args.classical_rng_seed_offset + 1
    rng_ab = np.random.RandomState(rng_ab_seed)
    rng_ba = np.random.RandomState(rng_ba_seed)
    output_path = args.seed_output
    output_file = open(output_path, "w", encoding="utf-8") if output_path else None

    try:
        result = run_density_workflow(
            theta=args.theta,
            config=config,
            seed=seed_value,
            rng_ab=rng_ab,
            rng_ba=rng_ba,
            optimize=args.optimize,
            random_theta=args.random_theta,
            theta_sweep=tuple(args.theta_sweep) if args.theta_sweep else None,
            seed_sweep=tuple(args.seed_sweep) if args.seed_sweep else None,
            bounds=bounds,
            tol=args.tol,
        )

        mode = result["mode"]

        if mode == "theta_sweep":
            print("=== Theta Sweep Results ===")
            for row in result["results"]:
                exps = compute_expectations(row)
                print(
                    f"theta {row['theta_deg']:.2f} deg ({row['theta']:.6f} rad) | "
                    f"energy {row['energy']:.6f} | ZZ {row['zz_cost']:.6f} | XX {row['xx_cost']:.6f}"
                )
                if output_file:
                    output_file.write(
                        f"theta_deg={row['theta_deg']:.6f} theta_rad={row['theta']:.12f} "
                        f"energy={row['energy']:.6f} zz_cost={row['zz_cost']:.6f} xx_cost={row['xx_cost']:.6f}\n"
                    )
                    _write_density_block(output_file, "ZZ_density_matrix", row["zz_rho"])
                    _write_density_block(output_file, "XX_density_matrix", row["xx_rho"])
                    output_file.write("\n")
            return

        if mode == "seed_sweep":
            theta_value = result["theta"]
            seeds = result["seeds"]
            if args.optimize or args.random_theta:
                raise SystemExit("--seed-sweep cannot be used together with --optimize/--random-theta")
            print("=== Seed Sweep Results ===")
            print(f"θ = {theta_value:.6f} rad fixed (deg = {math.degrees(theta_value):.2f})")
            if output_file:
                output_file.write(f"theta {theta_value:.12f} # radians\n")
            for seed, metrics in zip(seeds, result["results"]):
                exps = compute_expectations(metrics)
                print(
                    f"seed {seed:>4d} | energy {metrics['energy']:.6f} | ZZ {metrics['zz_cost']:.6f} | "
                    f"XX {metrics['xx_cost']:.6f} | <Z0> {exps['exp_z0']:.4f} | <Z1> {exps['exp_z1']:.4f} | "
                    f"<Z0Z1> {exps['exp_z0z1']:.4f} | <XX> {exps['exp_xx']:.4f}"
                )
                if output_file:
                    output_file.write(f"seed {seed}\n")
                    _write_density_block(output_file, "ZZ_density_matrix", metrics["zz_rho"])
                    _write_density_block(output_file, "XX_density_matrix", metrics["xx_rho"])
                    output_file.write("\n")
            return


        metrics = result["metrics"]
        exps = compute_expectations(metrics)
        print("=== Density VQE Result ===")
        print(f"theta [rad]      : {metrics['theta']:.8f}")
        print(f"energy (Hartree) : {metrics['energy']:.8f}")
        print(f"  ZZ cost        : {metrics['zz_cost']:.8f}")
        print(f"  XX cost        : {metrics['xx_cost']:.8f}")
        print(f"  <Z0>           : {exps['exp_z0']:.6f}")
        print(f"  <Z1>           : {exps['exp_z1']:.6f}")
        print(f"  <Z0Z1>         : {exps['exp_z0z1']:.6f}")
        print(f"  <XX>           : {exps['exp_xx']:.6f}")
        print(f"shots            : {config.shots}")
        print(
            "time [ns]        : ZZ {zz:.2f} + XX {xx:.2f} = {total:.2f}".format(
                zz=metrics["zz_time"], xx=metrics["xx_time"], total=metrics["total_time"]
            )
        )

        if output_file:
            output_file.write(f"theta {metrics['theta']:.12f} # radians\n")
            output_file.write(f"seed {seed_value}\n")
            _write_density_block(output_file, "ZZ_density_matrix", metrics["zz_rho"])
            _write_density_block(output_file, "XX_density_matrix", metrics["xx_rho"])
            output_file.write("\n")
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
