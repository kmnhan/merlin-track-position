"""Repeatable closed-loop correction simulations for algorithm tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position import constants
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    PIXEL_AXES,
    compute_lqr_correction_design,
    estimate_command_offset,
    initialize_lqr_kalman_state,
    load_calibration_dataset,
    measure_image_error,
    predict_lqr_kalman_state,
    solve_lqr_state_command_correction,
    update_lqr_kalman_state,
    validate_visual_calibration_dataset,
)

OBSERVATION_AXES = (
    "cam0_du",
    "cam0_dv",
    "cam1_du",
    "cam1_dv",
)

ROBUST_MOTOR_ERROR_MODEL_UM: Mapping[str, Any] = {
    "name": "correction_history_correlated_laplace_slip_v1",
    "tail_probability": 0.1223021582733813,
    "central_location_um": (2.0950474805045225, 0.054132964742873727, -0.4107579641032021),
    "central_basis": (
        (0.83784987, 0.52060870, 0.16423816),
        (-0.18361913, -0.01456508, 0.98288955),
        (-0.51409300, 0.85367115, -0.08339037),
    ),
    "central_laplace_scale_um": (3.00027740, 0.83774588, 0.37174270),
    "tail_location_um": (13.9651915, -3.50546409, -7.87087832),
    "tail_basis": (
        (0.81664222, 0.45630855, 0.35338081),
        (-0.21204776, -0.33224586, 0.91904757),
        (-0.53677857, 0.82546666, 0.17456679),
    ),
    "tail_laplace_scale_um": (4.98566436, 1.59041126, 0.75830819),
}

ZERO_MOTOR_ERROR_MODEL_UM: Mapping[str, Any] = {
    "name": "zero_motor_error",
    "tail_probability": 0.0,
    "central_location_um": (0.0, 0.0, 0.0),
    "central_basis": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "central_laplace_scale_um": (0.0, 0.0, 0.0),
    "tail_location_um": (0.0, 0.0, 0.0),
    "tail_basis": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "tail_laplace_scale_um": (0.0, 0.0, 0.0),
}


def default_initial_offsets_um(
    magnitudes_um: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
) -> np.ndarray:
    """Return deterministic x/y/z stress offsets in microns."""

    rows: list[np.ndarray] = []
    for magnitude_um in magnitudes_um:
        magnitude = float(magnitude_um)
        if not np.isfinite(magnitude) or magnitude < 0.0:
            raise ValueError("magnitudes_um must contain finite non-negative values")
        for axis_index in range(len(COMMAND_AXES)):
            for sign in (-1.0, 1.0):
                row = np.zeros(len(COMMAND_AXES), dtype=np.float64)
                row[axis_index] = sign * magnitude
                rows.append(row)
        for axes in ((0, 1), (0, 2), (1, 2), (0, 1, 2)):
            for sign_bits in range(1 << len(axes)):
                row = np.zeros(len(COMMAND_AXES), dtype=np.float64)
                for bit, axis_index in enumerate(axes):
                    row[axis_index] = (
                        magnitude if (sign_bits >> bit) & 1 else -magnitude
                    )
                rows.append(row)
    return np.stack(rows, axis=0) if rows else np.empty((0, len(COMMAND_AXES)))


def initial_offsets_from_correction_history(
    calibration_path: str | Path,
    correction_history_path: str | Path,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Project correction-history initial image shifts into command-space microns."""

    calibration = load_calibration_dataset(calibration_path)
    visual_jacobian = np.asarray(
        calibration["visual_jacobian_px_per_cmd_mm"].values,
        dtype=np.float64,
    )
    offsets: list[np.ndarray] = []
    with h5py.File(correction_history_path, "r") as file:
        for group_name in sorted(name for name in file if name.startswith("run_")):
            group = file[group_name]
            if "iteration_shift_px" not in group:
                continue
            shifts = group["iteration_shift_px"]
            if shifts.shape[0] == 0:
                continue
            offset_mm = estimate_command_offset(
                visual_jacobian,
                np.asarray(shifts[0], dtype=np.float64),
                weights=_weights_or_default(weights),
            )
            offsets.append(1000.0 * offset_mm)
    if not offsets:
        return np.empty((0, len(COMMAND_AXES)), dtype=np.float64)
    return np.stack(offsets, axis=0)


def sample_robust_motor_error_um(
    rng: np.random.Generator | int,
    count: int,
    model: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample motor execution errors and tail-event flags in microns."""

    generator = _as_rng(rng)
    if model is None:
        model = ROBUST_MOTOR_ERROR_MODEL_UM
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    tail_probability = _model_probability(model)
    tail_event = generator.random(count) < tail_probability
    errors = np.empty((count, len(COMMAND_AXES)), dtype=np.float64)
    central_count = int(np.count_nonzero(~tail_event))
    tail_count = int(np.count_nonzero(tail_event))
    if central_count:
        errors[~tail_event] = _sample_model_component(
            generator,
            central_count,
            model,
            prefix="central",
        )
    if tail_count:
        errors[tail_event] = _sample_model_component(
            generator,
            tail_count,
            model,
            prefix="tail",
        )
    return errors, tail_event


def simulate_shift_correction(
    calibration: xr.Dataset | str | Path,
    algorithm_configs: Sequence[Mapping[str, Any]],
    initial_offsets_um: Sequence[Sequence[float]] | np.ndarray,
    seed: int,
    *,
    damped_wls_pixel_tolerance_px: float = (
        constants.DEFAULT_DAMPED_WLS_CORRECTION_PIXEL_TOLERANCE_PX
    ),
    lqr_projected_tolerance: float = (
        constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
    ),
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    weights: Sequence[float] | np.ndarray | None = None,
    motor_error_model: Mapping[str, Any] | None = None,
    trials_per_offset: int = 1,
    min_command_norm_mm: float = constants.DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM,
    measurement_noise_covariance_px: (
        float | Sequence[Sequence[float]] | np.ndarray | None
    ) = None,
) -> xr.Dataset:
    """Run fast closed-loop simulations directly in image-shift space."""

    calibration_dataset = _as_calibration_dataset(calibration)
    return _simulate_correction(
        calibration_dataset,
        algorithm_configs,
        initial_offsets_um,
        seed,
        mode="shift",
        damped_wls_pixel_tolerance_px=damped_wls_pixel_tolerance_px,
        lqr_projected_tolerance=lqr_projected_tolerance,
        max_moves=max_moves,
        weights=weights,
        motor_error_model=motor_error_model,
        trials_per_offset=trials_per_offset,
        min_command_norm_mm=min_command_norm_mm,
        measurement_noise_covariance_px=measurement_noise_covariance_px,
        shift_kwargs={},
    )


def simulate_image_correction(
    calibration: xr.Dataset | str | Path,
    algorithm_configs: Sequence[Mapping[str, Any]],
    initial_offsets_um: Sequence[Sequence[float]] | np.ndarray,
    seed: int,
    *,
    damped_wls_pixel_tolerance_px: float = (
        constants.DEFAULT_DAMPED_WLS_CORRECTION_PIXEL_TOLERANCE_PX
    ),
    lqr_projected_tolerance: float = (
        constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
    ),
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    weights: Sequence[float] | np.ndarray | None = None,
    motor_error_model: Mapping[str, Any] | None = None,
    trials_per_offset: int = 1,
    min_command_norm_mm: float = constants.DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM,
    shift_kwargs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Run slower closed-loop simulations by rendering and remeasuring images."""

    calibration_dataset = _as_calibration_dataset(calibration)
    return _simulate_correction(
        calibration_dataset,
        algorithm_configs,
        initial_offsets_um,
        seed,
        mode="image",
        damped_wls_pixel_tolerance_px=damped_wls_pixel_tolerance_px,
        lqr_projected_tolerance=lqr_projected_tolerance,
        max_moves=max_moves,
        weights=weights,
        motor_error_model=motor_error_model,
        trials_per_offset=trials_per_offset,
        min_command_norm_mm=min_command_norm_mm,
        measurement_noise_covariance_px=None,
        shift_kwargs={} if shift_kwargs is None else dict(shift_kwargs),
    )


def summarize_simulation(result: xr.Dataset) -> xr.Dataset:
    """Summarize convergence, move count, and final residuals by algorithm."""

    _require_result_vars(
        result,
        (
            "converged",
            "move_count",
            "final_weighted_residual_px",
            "final_correction_criterion_residual",
        ),
    )
    algorithms = result.coords["algorithm"].values
    converged = np.asarray(result["converged"].values, dtype=bool)
    move_count = np.asarray(result["move_count"].values, dtype=np.float64)
    final_residual = np.asarray(
        result["final_weighted_residual_px"].values,
        dtype=np.float64,
    )
    final_criterion = np.asarray(
        result["final_correction_criterion_residual"].values,
        dtype=np.float64,
    )
    rows: dict[str, list[float]] = {
        "trial_count": [],
        "success_rate": [],
        "move_count_mean": [],
        "move_count_median": [],
        "move_count_p90": [],
        "move_count_p95": [],
        "move_count_max": [],
        "final_residual_median_px": [],
        "final_residual_p90_px": [],
        "final_residual_p95_px": [],
        "final_residual_max_px": [],
        "final_criterion_median": [],
        "final_criterion_p90": [],
        "final_criterion_p95": [],
        "final_criterion_max": [],
    }
    for index in range(len(algorithms)):
        moves = move_count[index]
        residuals = final_residual[index]
        criteria = final_criterion[index]
        rows["trial_count"].append(float(moves.size))
        rows["success_rate"].append(float(np.mean(converged[index])))
        rows["move_count_mean"].append(float(np.mean(moves)))
        rows["move_count_median"].append(float(np.median(moves)))
        rows["move_count_p90"].append(float(np.percentile(moves, 90)))
        rows["move_count_p95"].append(float(np.percentile(moves, 95)))
        rows["move_count_max"].append(float(np.max(moves)))
        rows["final_residual_median_px"].append(float(np.median(residuals)))
        rows["final_residual_p90_px"].append(float(np.percentile(residuals, 90)))
        rows["final_residual_p95_px"].append(float(np.percentile(residuals, 95)))
        rows["final_residual_max_px"].append(float(np.max(residuals)))
        rows["final_criterion_median"].append(float(np.median(criteria)))
        rows["final_criterion_p90"].append(float(np.percentile(criteria, 90)))
        rows["final_criterion_p95"].append(float(np.percentile(criteria, 95)))
        rows["final_criterion_max"].append(float(np.max(criteria)))

    return xr.Dataset(
        data_vars={
            name: (("algorithm",), np.asarray(values, dtype=np.float64))
            for name, values in rows.items()
        },
        coords={"algorithm": algorithms},
        attrs={
            "source_simulation_mode": result.attrs.get("simulation_mode", ""),
            "source_seed": result.attrs.get("seed", ""),
            "source_error_model": result.attrs.get("error_model_name", ""),
        },
    )


def _simulate_correction(
    calibration: xr.Dataset,
    algorithm_configs: Sequence[Mapping[str, Any]],
    initial_offsets_um: Sequence[Sequence[float]] | np.ndarray,
    seed: int,
    *,
    mode: str,
    damped_wls_pixel_tolerance_px: float,
    lqr_projected_tolerance: float,
    max_moves: int,
    weights: Sequence[float] | np.ndarray | None,
    motor_error_model: Mapping[str, Any] | None,
    trials_per_offset: int,
    min_command_norm_mm: float,
    measurement_noise_covariance_px: (
        float | Sequence[Sequence[float]] | np.ndarray | None
    ),
    shift_kwargs: Mapping[str, Any],
) -> xr.Dataset:
    weights_values = _observation_weights(weights)
    damped_wls_pixel_tolerance_px = _nonnegative_float(
        damped_wls_pixel_tolerance_px,
        "damped_wls_pixel_tolerance_px",
    )
    lqr_projected_tolerance = _nonnegative_float(
        lqr_projected_tolerance,
        "lqr_projected_tolerance",
    )
    min_command_norm_mm = _nonnegative_float(
        min_command_norm_mm,
        "min_command_norm_mm",
    )
    max_moves = int(max_moves)
    if max_moves < 0:
        raise ValueError("max_moves must be non-negative")
    trials_per_offset = int(trials_per_offset)
    if trials_per_offset <= 0:
        raise ValueError("trials_per_offset must be positive")

    configs = _algorithm_configs(algorithm_configs)
    initial_um = _initial_offsets(initial_offsets_um, trials_per_offset)
    trial_count = initial_um.shape[0]
    algorithm_count = len(configs)
    visual_jacobian = np.asarray(
        calibration["visual_jacobian_px_per_cmd_mm"].values,
        dtype=np.float64,
    )
    jacobian_observation = visual_jacobian.reshape(
        len(CAMERAS) * len(PIXEL_AXES),
        len(COMMAND_AXES),
    )
    axis_scale = np.asarray(calibration["axis_scale_cmd_mm"].values, dtype=np.float64)
    command_models = [
        _command_model(config, jacobian_observation, axis_scale, weights_values)
        for config in configs
    ]

    rng = np.random.default_rng(seed)
    if motor_error_model is None:
        motor_error_model = ROBUST_MOTOR_ERROR_MODEL_UM
    sampled_error_um, sampled_tail_event = sample_robust_motor_error_um(
        rng,
        trial_count * max_moves,
        motor_error_model,
    )
    sampled_error_um = sampled_error_um.reshape(
        trial_count,
        max_moves,
        len(COMMAND_AXES),
    )
    sampled_tail_event = sampled_tail_event.reshape(trial_count, max_moves)
    measurement_noise = _sample_measurement_noise_px(
        rng,
        trial_count,
        max_moves + 1,
        measurement_noise_covariance_px,
    )

    true_state_um = np.full(
        (algorithm_count, trial_count, max_moves + 1, len(COMMAND_AXES)),
        np.nan,
        dtype=np.float64,
    )
    measured_shift_px = np.full(
        (
            algorithm_count,
            trial_count,
            max_moves + 1,
            len(CAMERAS),
            len(PIXEL_AXES),
        ),
        np.nan,
        dtype=np.float64,
    )
    weighted_residual_px = np.full(
        (algorithm_count, trial_count, max_moves + 1),
        np.nan,
        dtype=np.float64,
    )
    correction_criterion_residual = np.full_like(weighted_residual_px, np.nan)
    command_delta_um = np.full(
        (algorithm_count, trial_count, max_moves, len(COMMAND_AXES)),
        np.nan,
        dtype=np.float64,
    )
    applied_delta_um = np.full_like(command_delta_um, np.nan)
    motor_error_um = np.full_like(command_delta_um, np.nan)
    tail_event = np.zeros((algorithm_count, trial_count, max_moves), dtype=bool)
    move_executed = np.zeros((algorithm_count, trial_count, max_moves), dtype=bool)
    converged = np.zeros((algorithm_count, trial_count), dtype=bool)
    move_count = np.zeros((algorithm_count, trial_count), dtype=np.int64)
    final_weighted_residual_px = np.full(
        (algorithm_count, trial_count),
        np.nan,
        dtype=np.float64,
    )
    final_correction_criterion_residual = np.full_like(
        final_weighted_residual_px,
        np.nan,
    )

    for algorithm_index, model in enumerate(command_models):
        state_um = initial_um.copy()
        done = np.zeros(trial_count, dtype=bool)
        moves_done = np.zeros(trial_count, dtype=np.int64)
        kalman_state, kalman_covariance = _initial_simulation_kalman_arrays(
            model,
            trial_count,
        )
        kalman_predicted_state, kalman_predicted_covariance = (
            _initial_simulation_kalman_arrays(model, trial_count)
        )
        kalman_prediction_pending = np.zeros(trial_count, dtype=bool)
        for iteration in range(max_moves + 1):
            true_state_um[algorithm_index, :, iteration, :] = state_um
            shifts = _measure_states(
                calibration,
                jacobian_observation,
                state_um / 1000.0,
                mode=mode,
                shift_kwargs=shift_kwargs,
            )
            if measurement_noise is not None:
                shifts = shifts + measurement_noise[:, iteration, :, :]
            measured_shift_px[algorithm_index, :, iteration, :, :] = shifts
            if bool(model.get("lqr_use_kalman_filter", False)):
                if iteration == 0:
                    for trial_index in range(trial_count):
                        initialized = initialize_lqr_kalman_state(
                            shifts[trial_index],
                            model["lqr_design"],
                            initial_covariance=model["lqr_kalman_initial_covariance"],
                        )
                        kalman_state[trial_index] = initialized["state"]
                        kalman_covariance[trial_index] = initialized["covariance"]
                else:
                    update_indices = np.nonzero(kalman_prediction_pending)[0]
                    for trial_index in update_indices:
                        updated = update_lqr_kalman_state(
                            kalman_predicted_state[trial_index],
                            kalman_predicted_covariance[trial_index],
                            shifts[trial_index],
                            model["lqr_design"],
                            measurement_noise=model[
                                "lqr_kalman_measurement_noise"
                            ],
                            measurement_covariance=model[
                                "lqr_kalman_measurement_covariance"
                            ],
                            innovation_gate=model["lqr_kalman_innovation_gate"],
                        )
                        kalman_state[trial_index] = updated["state"]
                        kalman_covariance[trial_index] = updated["covariance"]
                    kalman_prediction_pending[update_indices] = False
            residuals = _weighted_residuals(shifts, weights_values)
            weighted_residual_px[algorithm_index, :, iteration] = residuals
            criterion_residuals = _criterion_residuals(model, shifts, weights_values)
            correction_criterion_residual[algorithm_index, :, iteration] = (
                criterion_residuals
            )

            newly_converged = (~done) & (
                criterion_residuals <= _model_tolerance(
                    model,
                    damped_wls_pixel_tolerance_px=damped_wls_pixel_tolerance_px,
                    lqr_projected_tolerance=lqr_projected_tolerance,
                )
            )
            converged[algorithm_index, newly_converged] = True
            done[newly_converged] = True
            if iteration == max_moves:
                final_weighted_residual_px[algorithm_index, :] = residuals
                final_correction_criterion_residual[algorithm_index, :] = (
                    criterion_residuals
                )
                move_count[algorithm_index, :] = moves_done
                break

            active = ~done
            if not np.any(active):
                final_weighted_residual_px[algorithm_index, :] = residuals
                final_correction_criterion_residual[algorithm_index, :] = (
                    criterion_residuals
                )
                move_count[algorithm_index, :] = moves_done
                true_state_um[algorithm_index, :, iteration + 1 :, :] = state_um[
                    :,
                    np.newaxis,
                    :,
                ]
                break

            active_indices = np.nonzero(active)[0]
            if bool(model.get("lqr_use_kalman_filter", False)):
                commands_um = _commands_from_kalman_states(
                    model,
                    kalman_state[active],
                )
            else:
                commands_um = _commands_from_shifts(
                    model,
                    shifts[active],
                    axis_scale,
                    jacobian_observation,
                    weights_values,
                )
            command_norm_mm = np.linalg.norm(commands_um / 1000.0, axis=1)
            should_move = command_norm_mm > min_command_norm_mm
            if np.any(should_move):
                moving_indices = active_indices[should_move]
                moving_commands_um = commands_um[should_move]
                command_delta_um[
                    algorithm_index,
                    moving_indices,
                    iteration,
                    :,
                ] = moving_commands_um
                errors_um = sampled_error_um[moving_indices, iteration, :]
                applied_um = moving_commands_um + errors_um
                applied_delta_um[
                    algorithm_index,
                    moving_indices,
                    iteration,
                    :,
                ] = applied_um
                motor_error_um[
                    algorithm_index,
                    moving_indices,
                    iteration,
                    :,
                ] = errors_um
                tail_event[
                    algorithm_index,
                    moving_indices,
                    iteration,
                ] = sampled_tail_event[moving_indices, iteration]
                move_executed[
                    algorithm_index,
                    moving_indices,
                    iteration,
                ] = True
                if bool(model.get("lqr_use_kalman_filter", False)):
                    for local_index, trial_index in enumerate(moving_indices):
                        predicted = predict_lqr_kalman_state(
                            kalman_state[trial_index],
                            kalman_covariance[trial_index],
                            moving_commands_um[local_index] / 1000.0,
                            model["lqr_design"],
                            process_noise=model["lqr_kalman_process_noise"],
                        )
                        kalman_predicted_state[trial_index] = predicted["state"]
                        kalman_predicted_covariance[trial_index] = predicted[
                            "covariance"
                        ]
                    kalman_prediction_pending[moving_indices] = True
                state_um[moving_indices] += applied_um
                moves_done[moving_indices] += 1

    names = [str(config["name"]) for config in configs]
    kinds = [str(config["algorithm"]) for config in configs]
    return xr.Dataset(
        data_vars={
            "algorithm_kind": (("algorithm",), np.asarray(kinds, dtype=object)),
            "initial_offset_um": (
                ("trial", "command_axis"),
                initial_um,
                {"units": "um"},
            ),
            "true_state_um": (
                ("algorithm", "trial", "iteration", "command_axis"),
                true_state_um,
                {"units": "um"},
            ),
            "measured_shift_px": (
                ("algorithm", "trial", "iteration", "camera", "pixel_axis"),
                measured_shift_px,
                {"units": "px"},
            ),
            "weighted_residual_px": (
                ("algorithm", "trial", "iteration"),
                weighted_residual_px,
                {"units": "px"},
            ),
            "correction_criterion_residual": (
                ("algorithm", "trial", "iteration"),
                correction_criterion_residual,
            ),
            "command_delta_um": (
                ("algorithm", "trial", "move", "command_axis"),
                command_delta_um,
                {"units": "um"},
            ),
            "applied_delta_um": (
                ("algorithm", "trial", "move", "command_axis"),
                applied_delta_um,
                {"units": "um"},
            ),
            "motor_error_um": (
                ("algorithm", "trial", "move", "command_axis"),
                motor_error_um,
                {"units": "um"},
            ),
            "tail_event": (("algorithm", "trial", "move"), tail_event),
            "move_executed": (("algorithm", "trial", "move"), move_executed),
            "converged": (("algorithm", "trial"), converged),
            "move_count": (("algorithm", "trial"), move_count),
            "final_weighted_residual_px": (
                ("algorithm", "trial"),
                final_weighted_residual_px,
                {"units": "px"},
            ),
            "final_correction_criterion_residual": (
                ("algorithm", "trial"),
                final_correction_criterion_residual,
            ),
        },
        coords={
            "algorithm": names,
            "trial": np.arange(trial_count, dtype=np.int64),
            "iteration": np.arange(max_moves + 1, dtype=np.int64),
            "move": np.arange(max_moves, dtype=np.int64),
            "command_axis": list(COMMAND_AXES),
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
        },
        attrs={
            "simulation_mode": mode,
            "seed": int(seed),
            "damped_wls_pixel_tolerance_px": float(damped_wls_pixel_tolerance_px),
            "lqr_projected_tolerance": float(lqr_projected_tolerance),
            "max_moves": int(max_moves),
            "trials_per_offset": int(trials_per_offset),
            "weights": tuple(float(value) for value in weights_values),
            "weights_order": OBSERVATION_AXES,
            "error_model_name": str(motor_error_model.get("name", "custom")),
            "measurement_noise_covariance_px": _simulation_attrs_config(
                measurement_noise_covariance_px
            ),
        },
    )


def _as_calibration_dataset(calibration: xr.Dataset | str | Path) -> xr.Dataset:
    if isinstance(calibration, xr.Dataset):
        validate_visual_calibration_dataset(calibration)
        return calibration
    return load_calibration_dataset(calibration)


def _weights_or_default(
    weights: Sequence[float] | np.ndarray | None,
) -> Sequence[float] | np.ndarray | None:
    if weights is not None:
        return weights
    return constants.CORRECTION_OBSERVATION_WEIGHTS


def _observation_weights(weights: Sequence[float] | np.ndarray | None) -> np.ndarray:
    selected = _weights_or_default(weights)
    if selected is None:
        return np.ones(len(OBSERVATION_AXES), dtype=np.float64)
    values = np.asarray(selected, dtype=np.float64)
    if values.shape == (len(CAMERAS), len(PIXEL_AXES)):
        values = values.reshape(-1)
    if values.shape != (len(OBSERVATION_AXES),):
        raise ValueError("weights must have four observation values")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must contain finite non-negative values")
    if not np.any(values > 0.0):
        raise ValueError("weights must include at least one positive value")
    return values


def _initial_offsets(
    initial_offsets_um: Sequence[Sequence[float]] | np.ndarray,
    trials_per_offset: int,
) -> np.ndarray:
    offsets = np.asarray(initial_offsets_um, dtype=np.float64)
    if offsets.ndim == 1 and offsets.shape == (len(COMMAND_AXES),):
        offsets = offsets.reshape(1, len(COMMAND_AXES))
    if offsets.ndim != 2 or offsets.shape[1] != len(COMMAND_AXES):
        raise ValueError("initial_offsets_um must have shape (trial, 3)")
    if not np.isfinite(offsets).all():
        raise ValueError("initial_offsets_um must contain finite values")
    return np.repeat(offsets, trials_per_offset, axis=0)


def _algorithm_configs(
    algorithm_configs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    configs = [dict(config) for config in algorithm_configs]
    if not configs:
        raise ValueError("algorithm_configs must not be empty")
    names: set[str] = set()
    for config in configs:
        name = str(config.get("name", "")).strip()
        algorithm = str(config.get("algorithm", "")).strip().lower()
        if not name:
            raise ValueError("each algorithm config must include a non-empty name")
        if name in names:
            raise ValueError(f"duplicate algorithm config name {name!r}")
        if algorithm not in {"damped_wls", "lqr"}:
            raise ValueError("algorithm must be 'damped_wls' or 'lqr'")
        config["name"] = name
        config["algorithm"] = algorithm
        names.add(name)
    return configs


def _command_model(
    config: Mapping[str, Any],
    jacobian_observation: np.ndarray,
    axis_scale: np.ndarray,
    weights: np.ndarray,
) -> Mapping[str, Any]:
    algorithm = str(config["algorithm"])
    weight_matrix = np.diag(weights)
    if algorithm == "damped_wls":
        gain = _positive_float(
            config.get("gain", constants.DEFAULT_DAMPED_WLS_CORRECTION_GAIN),
            "gain",
        )
        damping_mu = _nonnegative_float(
            config.get(
                "damping_mu",
                constants.DEFAULT_DAMPED_WLS_CORRECTION_DAMPING_MU,
            ),
            "damping_mu",
        )
        scale_matrix = np.diag(axis_scale)
        normalized_jacobian = jacobian_observation @ scale_matrix
        lhs = (
            normalized_jacobian.T @ weight_matrix @ normalized_jacobian
            + damping_mu * np.eye(len(COMMAND_AXES), dtype=np.float64)
        )
        gain_matrix = gain * (
            scale_matrix
            @ np.linalg.solve(lhs, normalized_jacobian.T @ weight_matrix)
        )
        max_normalized_step = config.get(
            "max_normalized_step",
            constants.DEFAULT_DAMPED_WLS_CORRECTION_MAX_NORMALIZED_STEP,
        )
        min_axis_predicted_shift_px = config.get(
            "min_axis_predicted_shift_px",
            0.0,
        )
    else:
        gain = _positive_float(
            config.get("gain", constants.DEFAULT_LQR_CORRECTION_GAIN),
            "gain",
        )
        lqr_design = compute_lqr_correction_design(
            jacobian_observation,
            axis_scale,
            image_scale_px=_positive_float(
                config.get(
                    "lqr_image_scale_px",
                    constants.DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX,
                ),
                "lqr_image_scale_px",
            ),
            motor_penalty=_positive_float(
                config.get(
                    "lqr_motor_penalty",
                    constants.DEFAULT_LQR_CORRECTION_MOTOR_PENALTY,
                ),
                "lqr_motor_penalty",
            ),
            svd_relative_tolerance=_positive_float(
                config.get(
                    "lqr_svd_relative_tolerance",
                    constants.DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE,
                ),
                "lqr_svd_relative_tolerance",
            ),
            weights=weights,
        )
        gain_matrix = gain * np.asarray(lqr_design["real_feedback_gain"])
        max_normalized_step = config.get(
            "max_normalized_step",
            constants.DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP,
        )
        min_axis_predicted_shift_px = config.get(
            "min_axis_predicted_shift_px",
            0.0,
        )
        lqr_use_kalman_filter = bool(
            config.get(
                "lqr_use_kalman_filter",
                constants.DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER,
            )
        )

    max_step = _optional_positive_float(max_normalized_step, "max_normalized_step")
    min_axis_shift = _nonnegative_float(
        min_axis_predicted_shift_px,
        "min_axis_predicted_shift_px",
    )
    axis_sensitivity = np.sqrt(
        np.diag(jacobian_observation.T @ weight_matrix @ jacobian_observation)
    )
    model = {
        "algorithm": algorithm,
        "gain_matrix": gain_matrix,
        "max_normalized_step": max_step,
        "min_axis_predicted_shift_px": min_axis_shift,
        "axis_sensitivity": axis_sensitivity,
    }
    if algorithm == "lqr":
        model["lqr_design"] = lqr_design
    if algorithm == "lqr" and lqr_use_kalman_filter:
        model |= {
            "lqr_use_kalman_filter": True,
            "gain": gain,
            "lqr_kalman_process_noise": _simulation_kalman_covariance_config(
                config.get(
                    "lqr_kalman_process_noise",
                    constants.DEFAULT_LQR_CORRECTION_KALMAN_PROCESS_NOISE,
                ),
                int(lqr_design["rank"]),
                "lqr_kalman_process_noise",
                allow_zero=True,
            ),
            "lqr_kalman_measurement_noise": _positive_float(
                config.get(
                    "lqr_kalman_measurement_noise",
                    constants.DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_NOISE,
                ),
                "lqr_kalman_measurement_noise",
            ),
            "lqr_kalman_measurement_covariance": (
                None
                if config.get("lqr_kalman_measurement_covariance", None) is None
                else _simulation_kalman_covariance_config(
                    config["lqr_kalman_measurement_covariance"],
                    len(OBSERVATION_AXES),
                    "lqr_kalman_measurement_covariance",
                    allow_zero=False,
                )
            ),
            "lqr_kalman_initial_covariance": _positive_float(
                config.get(
                    "lqr_kalman_initial_covariance",
                    constants.DEFAULT_LQR_CORRECTION_KALMAN_INITIAL_COVARIANCE,
                ),
                "lqr_kalman_initial_covariance",
            ),
            "lqr_kalman_innovation_gate": _optional_positive_float(
                config.get(
                    "lqr_kalman_innovation_gate",
                    constants.DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE,
                ),
                "lqr_kalman_innovation_gate",
            ),
        }
    return model


def _commands_from_shifts(
    model: Mapping[str, Any],
    shift_px: np.ndarray,
    axis_scale: np.ndarray,
    jacobian_observation: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    del jacobian_observation, weights
    observations = np.asarray(shift_px, dtype=np.float64).reshape(-1, len(OBSERVATION_AXES))
    correction_mm = -(observations @ np.asarray(model["gain_matrix"]).T)
    max_normalized_step = model["max_normalized_step"]
    if max_normalized_step is not None:
        max_component = np.max(np.abs(correction_mm / axis_scale), axis=1)
        scale = np.minimum(
            1.0,
            float(max_normalized_step) / np.maximum(max_component, 1e-300),
        )
        correction_mm *= scale[:, np.newaxis]
    min_axis_predicted_shift_px = float(model["min_axis_predicted_shift_px"])
    if min_axis_predicted_shift_px > 0.0:
        predicted_axis_shift_px = (
            np.abs(correction_mm) * np.asarray(model["axis_sensitivity"])
        )
        correction_mm = correction_mm.copy()
        correction_mm[predicted_axis_shift_px < min_axis_predicted_shift_px] = 0.0
    return 1000.0 * correction_mm


def _commands_from_kalman_states(
    model: Mapping[str, Any],
    states: np.ndarray,
) -> np.ndarray:
    state_rows = np.asarray(states, dtype=np.float64)
    commands = [
        solve_lqr_state_command_correction(
            model["lqr_design"],
            state,
            gain=float(model["gain"]),
            max_normalized_step=model["max_normalized_step"],
        )
        for state in state_rows
    ]
    if not commands:
        return np.empty((0, len(COMMAND_AXES)), dtype=np.float64)
    return 1000.0 * np.stack(commands, axis=0)


def _initial_simulation_kalman_arrays(
    model: Mapping[str, Any],
    trial_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(model.get("lqr_use_kalman_filter", False)):
        return (
            np.empty((trial_count, 0), dtype=np.float64),
            np.empty((trial_count, 0, 0), dtype=np.float64),
        )
    rank = int(model["lqr_design"]["rank"])
    return (
        np.zeros((trial_count, rank), dtype=np.float64),
        np.zeros((trial_count, rank, rank), dtype=np.float64),
    )


def _measure_states(
    calibration: xr.Dataset,
    jacobian_observation: np.ndarray,
    states_mm: np.ndarray,
    *,
    mode: str,
    shift_kwargs: Mapping[str, Any],
) -> np.ndarray:
    if mode == "shift":
        return (states_mm @ jacobian_observation.T).reshape(
            states_mm.shape[0],
            len(CAMERAS),
            len(PIXEL_AXES),
        )
    if mode != "image":
        raise ValueError(f"unsupported simulation mode {mode!r}")
    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    ideal_shifts = (states_mm @ jacobian_observation.T).reshape(
        states_mm.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    measured_rows: list[np.ndarray] = []
    for shift in ideal_shifts:
        current_cam0 = _shift_image(reference_cam0, shift[0])
        current_cam1 = _shift_image(reference_cam1, shift[1])
        measurement = measure_image_error(
            reference_cam0,
            current_cam0[np.newaxis, :, :],
            reference_cam1,
            current_cam1[np.newaxis, :, :],
            **shift_kwargs,
        )
        measured_rows.append(
            np.asarray(measurement["shift_px"].values, dtype=np.float64)
        )
    return np.stack(measured_rows, axis=0)


def _shift_image(reference: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
    du_px, dv_px = np.asarray(shift_px, dtype=np.float64)
    return np.asarray(
        ndimage.shift(
            np.asarray(reference, dtype=np.float64),
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        ),
        dtype=np.float64,
    )


def _weighted_residuals(shifts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    observations = np.asarray(shifts, dtype=np.float64).reshape(-1, len(OBSERVATION_AXES))
    return np.sqrt(np.sum(weights[np.newaxis, :] * observations * observations, axis=1))


def _criterion_residuals(
    model: Mapping[str, Any],
    shifts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    algorithm = str(model["algorithm"])
    if algorithm == "damped_wls":
        return _weighted_residuals(shifts, weights)
    if algorithm == "lqr":
        observations = np.asarray(shifts, dtype=np.float64).reshape(
            -1,
            len(OBSERVATION_AXES),
        )
        design = model["lqr_design"]
        image_scale = np.asarray(design["image_scale"], dtype=np.float64)
        controllable_basis = np.asarray(design["controllable_basis"], dtype=np.float64)
        projected = (observations / image_scale[np.newaxis, :]) @ controllable_basis
        return np.linalg.norm(projected, axis=1)
    raise ValueError(f"unsupported correction algorithm {algorithm!r}")


def _model_tolerance(
    model: Mapping[str, Any],
    *,
    damped_wls_pixel_tolerance_px: float,
    lqr_projected_tolerance: float,
) -> float:
    algorithm = str(model["algorithm"])
    if algorithm == "damped_wls":
        return float(damped_wls_pixel_tolerance_px)
    if algorithm == "lqr":
        return float(lqr_projected_tolerance)
    raise ValueError(f"unsupported correction algorithm {algorithm!r}")


def _as_rng(rng: np.random.Generator | int) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _model_probability(model: Mapping[str, Any]) -> float:
    probability = float(model.get("tail_probability", 0.0))
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("tail_probability must be between 0 and 1")
    return probability


def _sample_model_component(
    rng: np.random.Generator,
    count: int,
    model: Mapping[str, Any],
    *,
    prefix: str,
) -> np.ndarray:
    location = _model_vector(model, f"{prefix}_location_um")
    basis = _model_matrix(model, f"{prefix}_basis")
    scale = _model_vector(model, f"{prefix}_laplace_scale_um")
    if np.any(scale < 0.0):
        raise ValueError(f"{prefix}_laplace_scale_um must be non-negative")
    coordinates = rng.laplace(
        loc=0.0,
        scale=scale[np.newaxis, :],
        size=(count, len(COMMAND_AXES)),
    )
    return location[np.newaxis, :] + coordinates @ basis.T


def _sample_measurement_noise_px(
    rng: np.random.Generator,
    trial_count: int,
    iteration_count: int,
    covariance_px: float | Sequence[Sequence[float]] | np.ndarray | None,
) -> np.ndarray | None:
    if covariance_px is None:
        return None
    covariance_config = _simulation_kalman_covariance_config(
        covariance_px,
        len(OBSERVATION_AXES),
        "measurement_noise_covariance_px",
        allow_zero=True,
    )
    covariance_values = np.asarray(covariance_config, dtype=np.float64)
    covariance = (
        float(covariance_values) * np.eye(len(OBSERVATION_AXES), dtype=np.float64)
        if covariance_values.ndim == 0
        else covariance_values
    )
    samples = rng.multivariate_normal(
        mean=np.zeros(len(OBSERVATION_AXES), dtype=np.float64),
        cov=covariance,
        size=trial_count * iteration_count,
    )
    return samples.reshape(
        trial_count,
        iteration_count,
        len(CAMERAS),
        len(PIXEL_AXES),
    )


def _model_vector(model: Mapping[str, Any], key: str) -> np.ndarray:
    values = np.asarray(model[key], dtype=np.float64)
    if values.shape != (len(COMMAND_AXES),):
        raise ValueError(f"{key} must have three values")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite values")
    return values


def _model_matrix(model: Mapping[str, Any], key: str) -> np.ndarray:
    values = np.asarray(model[key], dtype=np.float64)
    if values.shape != (len(COMMAND_AXES), len(COMMAND_AXES)):
        raise ValueError(f"{key} must have shape (3, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite values")
    return values


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name)


def _simulation_kalman_covariance_config(
    value: Any,
    size: int,
    name: str,
    *,
    allow_zero: bool,
) -> float | np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        scalar = float(values)
        if (
            not np.isfinite(scalar)
            or scalar < 0.0
            or (scalar == 0.0 and not allow_zero)
        ):
            bound = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
        return scalar
    if values.shape != (size, size):
        raise ValueError(f"{name} must be scalar or have shape ({size}, {size})")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    values = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(values)
    if np.min(eigenvalues) < -1e-12 or (
        not allow_zero and np.min(eigenvalues) <= 0.0
    ):
        bound = "positive semidefinite" if allow_zero else "positive definite"
        raise ValueError(f"{name} must be {bound}")
    return values


def _simulation_attrs_config(value: Any) -> str | float:
    if value is None:
        return ""
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return float(values)
    if not np.isfinite(values).all():
        raise ValueError("simulation attrs config must contain only finite values")
    return " ".join(f"{float(item):.17g}" for item in values.reshape(-1))


def _require_result_vars(result: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in result]
    if missing:
        raise ValueError(f"simulation result is missing variables: {missing}")
