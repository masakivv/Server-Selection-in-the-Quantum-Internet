"""Utilities for density-matrix based VQE runs.

This module exposes the core logic that was previously embedded in
``run_density.py`` so that other scripts or notebooks can re-use it directly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple, Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .simulation import XX_cost_density, ZZ_cost_density


ENERGY_OFFSET = 1 / 1.4172975


@dataclass
class SimulationConfig:
    """Container for simulation parameters shared across density runs."""

    num_runs: int
    shots: int
    dephase_rate: float
    client_fidelity: float
    distance: float
    T1: float
    client_T1: float
    T2_ratio: float
    sge: float
    dge: float
    gate_speed_factor: float
    client_gate_speed_factor: float
    entanglement_fidelity: float
    entanglement_speed_factor: float
    cc_jitter_mean_ns: float = 0.0
    ext_stochastic: bool = False
    ext_seed: Optional[int] = None
    ext_min_delay_ns: float = 1.0

    timing_mode: str = "factor"  
    gate_time_ns: Optional[float] = None  

    def as_kwargs(self) -> Dict[str, object]:
        """Convert config into keyword arguments for simulation helpers."""
        return {
            "num_runs": self.num_runs,
            "dephase_rates": [self.dephase_rate],
            "client_fidelitys": [self.client_fidelity],
            "distances": [self.distance],
            "T1s": [self.T1],
            "client_T1s": [self.client_T1],
            "T2_ratios": [self.T2_ratio],
            "sges": [self.sge],
            "dges": [self.dge],
            "gate_speed_factors": [self.gate_speed_factor],
            "client_gate_speed_factors": [self.client_gate_speed_factor],
            "entanglement_fidelities": [self.entanglement_fidelity],
            "entanglement_speed_factors": [self.entanglement_speed_factor],
            "cc_jitter_mean_nss": [self.cc_jitter_mean_ns],
            "ext_stochastic": self.ext_stochastic,
            "ext_seed": self.ext_seed,
            "ext_min_delay_ns": self.ext_min_delay_ns,
            "shots": self.shots,

            "timing_mode": self.timing_mode,
            "gate_time_ns": self.gate_time_ns,
        }


DEFAULT_CONFIG = SimulationConfig(
    num_runs=10,
    shots=1,
    dephase_rate=0.0,
    client_fidelity=0.0,
    distance=500.0,
    T1=1e50,
    client_T1=1e50,
    T2_ratio=0.1,
    sge=0.0,
    dge=0.0,
    gate_speed_factor=1.0,
    client_gate_speed_factor=1.0,
    entanglement_fidelity=1.0,
    entanglement_speed_factor=100.0,
    cc_jitter_mean_ns=0.0,
    ext_stochastic=False,
    ext_seed=None,
    ext_min_delay_ns=1.0,
    timing_mode="factor",
    gate_time_ns=None,
)

DEFAULT_THETA = 0.2297  


_PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z_I = np.kron(_PAULI_Z, _PAULI_I)
_I_Z = np.kron(_PAULI_I, _PAULI_Z)
_Z_Z = np.kron(_PAULI_Z, _PAULI_Z)
_X_X = np.kron(_PAULI_X, _PAULI_X)


def _expectation(dm: np.ndarray, operator: np.ndarray) -> float:
    if dm is None:
        return float("nan")
    return float(np.real(np.trace(dm @ operator)))


def single_angle_metrics(
    theta: float,
    config: SimulationConfig,
    seed: int,
    *,
    rng_ab=None,
    rng_ba=None,
) -> Tuple[float, Dict[str, Any]]:
    """Run ZZ/XX circuits once and collect the combined energy metrics."""

    kwargs = config.as_kwargs()
    if rng_ab is None:
        rng_ab = np.random.RandomState(seed + 100000)
    if rng_ba is None:
        rng_ba = np.random.RandomState(seed + 100001)
    zz_cost, zz_time, zz_rho, zz_exps, zz_ent_delay_mean_ns = ZZ_cost_density(
        angle=theta, flag=0, base_seed=seed, rng_ab=rng_ab, rng_ba=rng_ba, **kwargs
    )
    xx_cost, xx_time, xx_rho, xx_exps, xx_ent_delay_mean_ns = XX_cost_density(
        angle=theta, flag=1, base_seed=seed, rng_ab=rng_ab, rng_ba=rng_ba, **kwargs
    )
    energy = zz_cost + 2 * xx_cost + ENERGY_OFFSET
    metrics = {
        "theta": theta,
        "energy": energy,
        "zz_cost": zz_cost,
        "xx_cost": xx_cost,
        "zz_time": zz_time,
        "xx_time": xx_time,
        "total_time": zz_time + xx_time,
        "seed": seed,
        "zz_entanglement_delay_mean_ns": zz_ent_delay_mean_ns,
        "xx_entanglement_delay_mean_ns": xx_ent_delay_mean_ns,
        "zz_rho": zz_rho,
        "xx_rho": xx_rho,
        "zz_expectations": zz_exps,
        "xx_expectations": xx_exps,
    }
    return energy, metrics


def compute_expectations(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Ensure expectation values exist for logging/reporting."""

    zz_exps = metrics.get("zz_expectations") or {}
    xx_exps = metrics.get("xx_expectations") or {}
    zz_dm = metrics.get("zz_rho")
    xx_dm = metrics.get("xx_rho")
    return {
        "exp_z0": zz_exps.get("exp_z0", _expectation(zz_dm, _Z_I) if zz_dm is not None else float("nan")),
        "exp_z1": zz_exps.get("exp_z1", _expectation(zz_dm, _I_Z) if zz_dm is not None else float("nan")),
        "exp_z0z1": zz_exps.get("exp_z0z1", _expectation(zz_dm, _Z_Z) if zz_dm is not None else float("nan")),
        "exp_xx": xx_exps.get("exp_xx", _expectation(xx_dm, _X_X) if xx_dm is not None else float("nan")),
    }


def sweep_thetas(
    config: SimulationConfig,
    sweep_args: Iterable[float],
    bounds: Tuple[float, float],
    seed: int,
) -> List[Dict[str, Any]]:
    """Evaluate metrics for a theta sweep (degrees input expected)."""

    start_deg, stop_deg, step_deg = sweep_args
    if step_deg == 0:
        raise ValueError("STEP must be non-zero for theta sweep")
    step_rad = math.radians(step_deg)
    start_rad = math.radians(start_deg)
    stop_rad = math.radians(stop_deg)

    results = []
    theta = start_rad
    compare = (lambda a, b: a <= b + 1e-12) if step_rad > 0 else (lambda a, b: a >= b - 1e-12)
    while compare(theta, stop_rad):
        theta_clamped = max(min(theta, bounds[1]), bounds[0])
        _, metrics = single_angle_metrics(theta_clamped, config, seed)
        metrics["theta_deg"] = math.degrees(theta_clamped)
        results.append(metrics)
        theta += step_rad
    return results


def generate_seed_list(start: int, stop: int, step: int) -> List[int]:
    """Generate a list of seed values for a sweep."""

    if step == 0:
        raise ValueError("STEP must be non-zero for seed sweep")
    seeds = []
    current = start
    if step > 0:
        while current <= stop:
            seeds.append(current)
            current += step
    else:
        while current >= stop:
            seeds.append(current)
            current += step
    return seeds


def _choose_theta(
    theta: Optional[float],
    *,
    optimize: bool,
    random_theta: bool,
    bounds: Tuple[float, float],
    tol: float,
    seed: int,
    config: SimulationConfig,
) -> float:
    if optimize:
        def objective(theta_value: float) -> float:
            energy_value, _ = single_angle_metrics(theta_value, config, seed)
            return energy_value

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=bounds,
            tol=tol,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return float(result.x)

    if theta is not None:
        return float(theta)

    if random_theta:
        low, high = bounds
        return random.uniform(low, high)

    return DEFAULT_THETA


def run_density(
    *,
    theta: Optional[float] = None,
    config: Optional[SimulationConfig] = None,
    seed: int = 42,
    rng_ab=None,
    rng_ba=None,
    optimize: bool = False,
    random_theta: bool = False,
    theta_sweep: Optional[Tuple[float, float, float]] = None,
    seed_sweep: Optional[Tuple[int, int, int]] = None,
    bounds: Tuple[float, float] = (-math.pi, math.pi),
    tol: float = 0.0015,
) -> Dict[str, Any]:
    """Run the density-matrix VQE workflow and return raw metrics.

    The returned dictionary contains a ``mode`` key describing which branch
    was executed (``"single"``, ``"theta_sweep"`` or ``"seed_sweep"``) and the
    associated metrics.
    """

    cfg = config or DEFAULT_CONFIG

    cfg = replace(cfg)

    if theta_sweep is not None:
        results = sweep_thetas(cfg, theta_sweep, bounds, seed)
        return {"mode": "theta_sweep", "results": results}

    if seed_sweep is not None:
        if optimize or random_theta:
            raise ValueError("--seed-sweep cannot be combined with optimize/random_theta")
        seeds = generate_seed_list(*seed_sweep)
        theta_value = _choose_theta(theta, optimize=False, random_theta=False, bounds=bounds, tol=tol, seed=seed, config=cfg)
        sweep_results = []
        for sweep_seed in seeds:
            _, metrics = single_angle_metrics(theta_value, cfg, sweep_seed, rng_ab=rng_ab, rng_ba=rng_ba)
            sweep_results.append(metrics)
        return {"mode": "seed_sweep", "theta": theta_value, "seeds": seeds, "results": sweep_results}

    theta_value = _choose_theta(
        theta,
        optimize=optimize,
        random_theta=random_theta,
        bounds=bounds,
        tol=tol,
        seed=seed,
        config=cfg,
    )
    energy, metrics = single_angle_metrics(theta_value, cfg, seed, rng_ab=rng_ab, rng_ba=rng_ba)
    return {"mode": "single", "theta": theta_value, "metrics": metrics, "energy": energy}


__all__ = [
    "ENERGY_OFFSET",
    "SimulationConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_THETA",
    "single_angle_metrics",
    "compute_expectations",
    "sweep_thetas",
    "generate_seed_list",
    "run_density",
]
