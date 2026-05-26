import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.tracking.calibration_core import (
    compute_lqr_correction_design,
)
from merlin_track_position.tracking.correction_simulation import (
    ROBUST_MOTOR_ERROR_MODEL_UM,
    ZERO_MOTOR_ERROR_MODEL_UM,
    default_initial_offsets_um,
    sample_robust_motor_error_um,
    initial_offsets_from_correction_history,
    simulate_image_correction,
    simulate_shift_correction,
    summarize_simulation,
)
from merlin_track_position.tracking.calibration_core import save_calibration_dataset
from merlin_track_position.tracking.sample_calibration import (
    build_sample_calibration_dataset,
)


def sample_calibration():
    return build_sample_calibration_dataset(
        image_shape_cam0=(128, 144),
        image_shape_cam1=(128, 144),
    )


class RobustMotorErrorSamplerTests(unittest.TestCase):
    def test_sampler_is_deterministic_for_fixed_seed(self):
        first_error, first_tail = sample_robust_motor_error_um(1234, 16)
        second_error, second_tail = sample_robust_motor_error_um(1234, 16)

        np.testing.assert_allclose(first_error, second_error)
        np.testing.assert_array_equal(first_tail, second_tail)
        self.assertEqual(first_error.shape, (16, 3))
        self.assertEqual(first_tail.shape, (16,))
        self.assertTrue(np.isfinite(first_error).all())
        self.assertEqual(first_tail.dtype, bool)

    def test_sampler_supports_central_only_and_tail_only_modes(self):
        central_model = dict(ROBUST_MOTOR_ERROR_MODEL_UM)
        central_model["tail_probability"] = 0.0
        tail_model = dict(ROBUST_MOTOR_ERROR_MODEL_UM)
        tail_model["tail_probability"] = 1.0

        _, central_tail = sample_robust_motor_error_um(1, 8, central_model)
        _, tail_tail = sample_robust_motor_error_um(1, 8, tail_model)

        self.assertFalse(bool(np.any(central_tail)))
        self.assertTrue(bool(np.all(tail_tail)))


class CorrectionSimulationTests(unittest.TestCase):
    def test_lqr_retuned_defaults_are_locked_to_history_fit(self):
        self.assertEqual(
            constants.CORRECTION_OBSERVATION_WEIGHTS,
            (2.12, 0.45, 0.32, 1.11),
        )
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_GAIN, 0.50)
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE, 0.75)
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_MOTOR_PENALTY, 25.0)
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP, 0.25)
        self.assertTrue(constants.DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER)
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_KALMAN_PROCESS_NOISE, 0.05)
        self.assertEqual(
            constants.DEFAULT_LQR_CORRECTION_KALMAN_INITIAL_COVARIANCE,
            1.0,
        )
        self.assertEqual(constants.DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE, 25.0)
        np.testing.assert_allclose(
            np.asarray(
                constants.DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_COVARIANCE
            ),
            np.array(
                [
                    [7.768352e-04, -2.612795e-04, -5.777776e-04, -1.457239e-04],
                    [-2.612795e-04, 4.261280e-03, 4.355556e-03, -1.190572e-03],
                    [-5.777776e-04, 4.355556e-03, 8.296296e-03, -1.940741e-03],
                    [-1.457239e-04, -1.190572e-03, -1.940741e-03, 1.724983e-03],
                ],
                dtype=float,
            ),
        )

    def test_initial_offsets_from_correction_history_projects_first_iteration(self):
        calibration = sample_calibration()
        offset_um = np.array([4.0, -3.0, 2.0], dtype=float)
        jacobian = calibration["px_per_cmd_mm"].values.reshape(4, 3)
        shift_px = (jacobian @ (offset_um / 1000.0)).reshape(2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_path = Path(tmpdir) / "calibration.h5"
            history_path = Path(tmpdir) / "calibration_corrections.h5"
            save_calibration_dataset(calibration, calibration_path)
            with h5py.File(history_path, "w") as history:
                group = history.create_group("run_000000")
                group.create_dataset(
                    "iteration_shift_px",
                    data=shift_px.reshape(1, 2, 2),
                )

            projected = initial_offsets_from_correction_history(
                calibration_path,
                history_path,
                weights=(1.0, 1.0, 1.0, 1.0),
            )

        np.testing.assert_allclose(projected, offset_um.reshape(1, 3), atol=1e-9)

    def test_shift_level_no_error_converges_for_lqr_configs(self):
        result = simulate_shift_correction(
            sample_calibration(),
            [
                {"name": "nominal", "gain": 1.0, "max_normalized_step": None},
                {"name": "limited", "gain": 1.0, "max_normalized_step": 0.5},
            ],
            np.array([[5.0, -3.0, 2.0], [-5.0, 4.0, -3.0]]),
            seed=10,
            lqr_projected_tolerance=0.5,
            max_moves=8,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            weights=(1.0, 1.0, 1.0, 1.0),
        )

        self.assertTrue(bool(result["converged"].values.all()))
        self.assertEqual(result.sizes["lqr_config"], 2)
        self.assertEqual(result.sizes["trial"], 2)
        self.assertEqual(result.sizes["iteration"], 9)
        self.assertEqual(result.sizes["move"], 8)

    def test_shift_level_robust_run_is_repeatable_and_summarizable(self):
        kwargs = dict(
            calibration=sample_calibration(),
            lqr_configs=[
                {"name": "lqr_g06", "gain": 0.6},
                {"name": "lqr_g09", "gain": 0.9},
            ],
            initial_offsets_um=default_initial_offsets_um((2.0, 5.0)),
            seed=20260515,
            lqr_projected_tolerance=2.0,
            max_moves=6,
            trials_per_offset=2,
        )

        first = simulate_shift_correction(**kwargs)
        second = simulate_shift_correction(**kwargs)
        xr.testing.assert_identical(first, second)
        self.assertEqual(
            set(first.dims),
            {
                "lqr_config",
                "trial",
                "iteration",
                "move",
                "command_axis",
                "camera",
                "pixel_axis",
            },
        )
        self.assertTrue(np.isfinite(first["weighted_residual_px"].values).any())

        summary = summarize_simulation(first)
        self.assertIn("success_rate", summary)
        self.assertEqual(summary.sizes["lqr_config"], 2)
        self.assertTrue(np.isfinite(summary["move_count_mean"].values).all())

    def test_shift_level_lqr_kalman_run_is_repeatable_with_measurement_noise(self):
        kwargs = dict(
            calibration=sample_calibration(),
            lqr_configs=[
                {
                    "name": "lqr_kalman",
                    "gain": 0.8,
                    "lqr_use_kalman_filter": True,
                    "lqr_kalman_process_noise": 0.05,
                    "lqr_kalman_measurement_noise": 1.0,
                    "lqr_kalman_innovation_gate": None,
                }
            ],
            initial_offsets_um=default_initial_offsets_um((2.0,)),
            seed=12345,
            lqr_projected_tolerance=2.0,
            max_moves=5,
            trials_per_offset=2,
            measurement_noise_covariance_px=0.01,
        )

        first = simulate_shift_correction(**kwargs)
        second = simulate_shift_correction(**kwargs)

        xr.testing.assert_identical(first, second)
        self.assertEqual(first.attrs["measurement_noise_covariance_px"], 0.01)
        self.assertEqual(first.sizes["lqr_config"], 1)
        self.assertTrue(np.isfinite(first["weighted_residual_px"].values).any())

    def test_shift_level_default_kalman_uses_measurement_covariance_matrix(self):
        measurement_covariance = np.asarray(
            constants.DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_COVARIANCE,
            dtype=float,
        )
        result = simulate_shift_correction(
            sample_calibration(),
            [{"name": "default_lqr"}],
            default_initial_offsets_um((2.0,)),
            seed=20260521,
            lqr_projected_tolerance=(
                constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
            ),
            max_moves=constants.DEFAULT_CORRECTION_MAX_MOVES,
            weights=constants.CORRECTION_OBSERVATION_WEIGHTS,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            trials_per_offset=2,
            measurement_noise_covariance_px=measurement_covariance,
        )
        explicit = simulate_shift_correction(
            sample_calibration(),
            [
                {
                    "name": "default_lqr",
                    "lqr_kalman_measurement_covariance": measurement_covariance,
                }
            ],
            default_initial_offsets_um((2.0,)),
            seed=20260521,
            lqr_projected_tolerance=(
                constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
            ),
            max_moves=constants.DEFAULT_CORRECTION_MAX_MOVES,
            weights=constants.CORRECTION_OBSERVATION_WEIGHTS,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            trials_per_offset=2,
            measurement_noise_covariance_px=measurement_covariance,
        )

        xr.testing.assert_identical(result, explicit)
        self.assertEqual(result.sizes["lqr_config"], 1)
        self.assertTrue(bool(result["converged"].values.all()))
        self.assertTrue(np.isfinite(result["final_weighted_residual_px"].values).all())

    def test_retuned_lqr_defaults_improve_over_responsive_response_replay(self):
        calibration = sample_calibration()
        initial_offsets_um = default_initial_offsets_um((2.0, 5.0, 10.0))
        old = _simulate_over_responsive_lqr(
            calibration,
            initial_offsets_um,
            gain=0.95,
            lqr_projected_tolerance=2.0,
            lqr_motor_penalty=100.0,
            max_normalized_step=0.5,
            response_scale=1.7,
        )
        retuned = _simulate_over_responsive_lqr(
            calibration,
            initial_offsets_um,
            gain=constants.DEFAULT_LQR_CORRECTION_GAIN,
            lqr_projected_tolerance=(
                constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
            ),
            lqr_motor_penalty=constants.DEFAULT_LQR_CORRECTION_MOTOR_PENALTY,
            max_normalized_step=constants.DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP,
            response_scale=1.7,
        )

        self.assertLess(retuned["final_residual_p90_px"], old["final_residual_p90_px"])
        self.assertLess(retuned["move_count_mean"], old["move_count_mean"])
        self.assertLess(retuned["final_residual_p90_px"], 0.1)
        self.assertGreaterEqual(retuned["move_count_mean"], 1.5)
        self.assertLessEqual(retuned["move_count_mean"], 2.5)

    def test_image_level_no_error_reduces_measured_residual(self):
        result = simulate_image_correction(
            sample_calibration(),
            [
                {
                    "name": "lqr",
                    "gain": 1.0,
                    "max_normalized_step": None,
                }
            ],
            np.array([[10.0, -4.0, 6.0]]),
            seed=100,
            max_moves=1,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            weights=(1.0, 1.0, 1.0, 1.0),
            shift_kwargs={"check_tiles": False},
        )

        residual = result["weighted_residual_px"].values[0, 0]
        self.assertLess(residual[-1], residual[0])
        self.assertTrue(np.isfinite(result["measured_shift_px"].values).all())

    def test_lqr_higher_gain_uses_fewer_moves_with_projected_criterion(self):
        result = simulate_shift_correction(
            sample_calibration(),
            [
                {"name": "lqr_g06", "gain": 0.6},
                {"name": "lqr_g10", "gain": 1.0},
            ],
            default_initial_offsets_um((2.0, 5.0, 10.0)),
            seed=20260515,
            lqr_projected_tolerance=2.0,
            max_moves=12,
            trials_per_offset=2,
        )

        lqr_g06_mean = float(result["move_count"].sel(lqr_config="lqr_g06").mean())
        lqr_g10_mean = float(result["move_count"].sel(lqr_config="lqr_g10").mean())
        self.assertLess(lqr_g10_mean, lqr_g06_mean)


def _simulate_over_responsive_lqr(
    calibration: xr.Dataset,
    initial_offsets_um: np.ndarray,
    *,
    gain: float,
    lqr_projected_tolerance: float,
    lqr_motor_penalty: float,
    max_normalized_step: float | None,
    response_scale: float,
) -> dict[str, float]:
    jacobian = np.asarray(
        calibration["px_per_cmd_mm"].values,
        dtype=float,
    ).reshape(4, 3)
    axis_scale = np.asarray(calibration["axis_scale_cmd_mm"].values, dtype=float)
    weights = np.asarray(constants.CORRECTION_OBSERVATION_WEIGHTS, dtype=float)
    weight_matrix = np.diag(weights)
    design = compute_lqr_correction_design(
        jacobian,
        axis_scale,
        image_scale_px=constants.DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX,
        motor_penalty=lqr_motor_penalty,
        svd_relative_tolerance=constants.DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE,
        weights=weights,
    )
    basis = np.asarray(design["controllable_basis"], dtype=float)
    image_scale = np.asarray(design["image_scale"], dtype=float)
    feedback_gain = np.asarray(design["real_feedback_gain"], dtype=float)

    final_residuals: list[float] = []
    move_counts: list[int] = []
    for offset_um in np.asarray(initial_offsets_um, dtype=float):
        shift = jacobian @ (offset_um / 1000.0)
        move_count = 0
        for _ in range(constants.DEFAULT_CORRECTION_MAX_MOVES):
            normalized_state = basis.T @ (shift / image_scale)
            if float(np.linalg.norm(normalized_state)) <= lqr_projected_tolerance:
                break
            correction_mm = -gain * (feedback_gain @ shift)
            if max_normalized_step is not None:
                max_component = float(np.max(np.abs(correction_mm / axis_scale)))
                if max_component > max_normalized_step:
                    correction_mm *= max_normalized_step / max_component
            shift = shift + response_scale * (jacobian @ correction_mm)
            move_count += 1
        final_residuals.append(float(np.sqrt(shift @ weight_matrix @ shift)))
        move_counts.append(move_count)

    return {
        "final_residual_p90_px": float(np.percentile(final_residuals, 90)),
        "move_count_mean": float(np.mean(move_counts)),
    }


if __name__ == "__main__":
    unittest.main()
