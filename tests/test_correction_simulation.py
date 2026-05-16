import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

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
    def test_initial_offsets_from_correction_history_projects_first_iteration(self):
        calibration = sample_calibration()
        offset_um = np.array([4.0, -3.0, 2.0], dtype=float)
        jacobian = calibration["visual_jacobian_px_per_cmd_mm"].values.reshape(4, 3)
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

    def test_shift_level_no_error_converges_for_wls_and_lqr(self):
        result = simulate_shift_correction(
            sample_calibration(),
            [
                {
                    "name": "wls",
                    "algorithm": "damped_wls",
                    "gain": 1.0,
                    "damping_mu": 0.0,
                    "max_normalized_step": None,
                },
                {
                    "name": "lqr",
                    "algorithm": "lqr",
                    "gain": 1.0,
                    "max_normalized_step": None,
                },
            ],
            np.array([[5.0, -3.0, 2.0], [-5.0, 4.0, -3.0]]),
            seed=10,
            pixel_tolerance_px=0.05,
            max_moves=8,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            weights=(1.0, 1.0, 1.0, 1.0),
        )

        self.assertTrue(bool(result["converged"].values.all()))
        self.assertEqual(result.sizes["algorithm"], 2)
        self.assertEqual(result.sizes["trial"], 2)
        self.assertEqual(result.sizes["iteration"], 9)
        self.assertEqual(result.sizes["move"], 8)

    def test_shift_level_robust_run_is_repeatable_and_summarizable(self):
        kwargs = dict(
            calibration=sample_calibration(),
            algorithm_configs=[
                {
                    "name": "wls",
                    "algorithm": "damped_wls",
                    "gain": 0.6,
                    "damping_mu": 1.0,
                },
                {
                    "name": "lqr",
                    "algorithm": "lqr",
                    "gain": 0.6,
                },
            ],
            initial_offsets_um=default_initial_offsets_um((2.0, 5.0)),
            seed=20260515,
            pixel_tolerance_px=0.2,
            max_moves=6,
            trials_per_offset=2,
        )

        first = simulate_shift_correction(**kwargs)
        second = simulate_shift_correction(**kwargs)
        xr.testing.assert_identical(first, second)
        self.assertEqual(
            set(first.dims),
            {
                "algorithm",
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
        self.assertEqual(summary.sizes["algorithm"], 2)
        self.assertTrue(np.isfinite(summary["move_count_mean"].values).all())

    def test_image_level_no_error_reduces_measured_residual(self):
        result = simulate_image_correction(
            sample_calibration(),
            [
                {
                    "name": "wls",
                    "algorithm": "damped_wls",
                    "gain": 1.0,
                    "damping_mu": 0.0,
                    "max_normalized_step": None,
                }
            ],
            np.array([[10.0, -4.0, 6.0]]),
            seed=100,
            pixel_tolerance_px=0.05,
            max_moves=1,
            motor_error_model=ZERO_MOTOR_ERROR_MODEL_UM,
            weights=(1.0, 1.0, 1.0, 1.0),
            shift_kwargs={"check_tiles": False, "upsample_factor": 20},
        )

        residual = result["weighted_residual_px"].values[0, 0]
        self.assertLess(residual[-1], residual[0])
        self.assertTrue(np.isfinite(result["measured_shift_px"].values).all())

    def test_equal_gain_wls_and_lqr_match_and_higher_lqr_gain_uses_fewer_moves(self):
        result = simulate_shift_correction(
            sample_calibration(),
            [
                {
                    "name": "wls_g06",
                    "algorithm": "damped_wls",
                    "gain": 0.6,
                    "damping_mu": 1.0,
                },
                {"name": "lqr_g06", "algorithm": "lqr", "gain": 0.6},
                {"name": "lqr_g10", "algorithm": "lqr", "gain": 1.0},
            ],
            default_initial_offsets_um((2.0, 5.0, 10.0)),
            seed=20260515,
            pixel_tolerance_px=0.2,
            max_moves=12,
            trials_per_offset=2,
        )

        move_count = result["move_count"].sel(algorithm=["wls_g06", "lqr_g06"])
        np.testing.assert_array_equal(
            move_count.sel(algorithm="wls_g06").values,
            move_count.sel(algorithm="lqr_g06").values,
        )
        lqr_g06_mean = float(result["move_count"].sel(algorithm="lqr_g06").mean())
        lqr_g10_mean = float(result["move_count"].sel(algorithm="lqr_g10").mean())
        self.assertLess(lqr_g10_mean, lqr_g06_mean)


if __name__ == "__main__":
    unittest.main()
