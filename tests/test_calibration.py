import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    PIXEL_AXES,
    derive_axis_scale_from_jacobian,
    flush_pending_calibration_datasets,
    fit_visual_jacobian_calibration,
    load_calibration_dataset,
    refine_visual_jacobian_from_observations,
    save_calibration_dataset,
    save_calibration_dataset_deferred,
    solve_damped_command_correction,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)
import merlin_track_position.tracking.calibration_core as calibration_core
import merlin_track_position.tracking.correct as correct_module
from merlin_track_position.tracking.calibrate import visual_calibration_probe_count
from merlin_track_position.tracking.correct import (
    correction_history_path,
    do_correction,
    flush_pending_correction_history_datasets,
    load_latest_correction_history_dataset,
)
from merlin_track_position.tracking.persistence import pending_entry_count


def visual_jacobian():
    return np.array(
        [
            [[270.0, -140.0, 70.0], [90.0, 310.0, -120.0]],
            [[-210.0, 180.0, 330.0], [240.0, 50.0, 160.0]],
        ],
        dtype=float,
    )


def probe_deltas():
    rows = []
    for axis in COMMAND_AXES:
        step = constants.DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS[axis]
        for sign in (1.0, -1.0, 1.0):
            row = np.zeros(len(COMMAND_AXES), dtype=float)
            row[COMMAND_AXES.index(axis)] = sign * step
            rows.append(row)
    return np.asarray(rows, dtype=float)


def measured_from_jacobian(command_delta, jacobian):
    observation = jacobian.reshape(4, 3)
    return (command_delta @ observation.T).reshape(
        command_delta.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )


def empty_capture_stacks(probe_count, shape=(8, 9)):
    return [np.zeros((1, *shape), dtype=np.float32) for _ in range(probe_count)]


def shift_dataset(values):
    return xr.Dataset(
        data_vars={
            "shift_px": (("camera", "pixel_axis"), np.asarray(values, dtype=float)),
            "capture_shift_mad_px": (
                ("camera", "pixel_axis"),
                np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float),
            ),
            "current_cam0": (("y_cam0", "x_cam0"), np.zeros((4, 5), dtype=float)),
            "current_cam1": (("y_cam1", "x_cam1"), np.zeros((6, 7), dtype=float)),
        },
        coords={
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
            "y_cam0": np.arange(4),
            "x_cam0": np.arange(5),
            "y_cam1": np.arange(6),
            "x_cam1": np.arange(7),
        },
        attrs={"warnings": ""},
    )


def x_shift(value: float) -> np.ndarray:
    return np.array([[float(value), 0.0], [0.0, 0.0]], dtype=float)


def assert_hdf5_image_dataset_compressed(
    test_case: unittest.TestCase,
    dataset: h5py.Dataset,
    expected_dtype,
) -> None:
    test_case.assertEqual(dataset.dtype, np.dtype(expected_dtype))
    test_case.assertEqual(dataset.compression, "gzip")
    test_case.assertEqual(dataset.compression_opts, 4)
    test_case.assertTrue(dataset.shuffle)


def calibration_dataset(jacobian=None):
    if jacobian is None:
        jacobian = visual_jacobian()
    command_delta = probe_deltas()
    measured = measured_from_jacobian(command_delta, jacobian)
    pre_commanded = np.cumsum(
        np.vstack([np.zeros((1, 3)), command_delta[:-1]]),
        axis=0,
    )
    post_commanded = pre_commanded + command_delta
    dataset = xr.Dataset(
        data_vars={
            "visual_jacobian_px_per_cmd_mm": (
                ("camera", "pixel_axis", "command_axis"),
                jacobian,
            ),
            "axis_scale_cmd_mm": (("command_axis",), np.array([0.5, 0.5, 0.5])),
            "reference_cam0": (("y_cam0", "x_cam0"), np.zeros((4, 5))),
            "reference_cam1": (("y_cam1", "x_cam1"), np.zeros((6, 7))),
            "probe_command_delta_mm": (("probe", "command_axis"), command_delta),
            "probe_measured_delta_px": (
                ("probe", "camera", "pixel_axis"),
                measured,
            ),
            "pre_commanded_position_mm": (("probe", "command_axis"), pre_commanded),
            "post_commanded_position_mm": (("probe", "command_axis"), post_commanded),
            "pre_readback_position_mm": (
                ("probe", "command_axis"),
                pre_commanded + 10.0,
            ),
            "post_readback_position_mm": (
                ("probe", "command_axis"),
                post_commanded - 7.0,
            ),
        },
        coords={
            "probe": np.arange(command_delta.shape[0]),
            "command_axis": list(COMMAND_AXES),
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
            "y_cam0": np.arange(4),
            "x_cam0": np.arange(5),
            "y_cam1": np.arange(6),
            "x_cam1": np.arange(7),
        },
        attrs={
            "warnings": "",
            "jacobian_refinement_count": 0,
        },
    )
    validate_visual_calibration_dataset(dataset)
    return dataset


class VisualCalibrationTests(unittest.TestCase):
    def test_visual_calibration_probe_count_uses_default_repeats(self):
        self.assertEqual(
            visual_calibration_probe_count(),
            constants.DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION * 6,
        )

    def test_fitted_visual_jacobian_matches_synthetic_response(self):
        command_delta = probe_deltas()
        expected = visual_jacobian()
        measured = measured_from_jacobian(command_delta, expected)
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        pre = np.zeros_like(command_delta)
        post = pre + command_delta
        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            calibration = fit_visual_jacobian_calibration(
                reference_cam0=np.zeros((8, 9)),
                reference_cam1=np.zeros((8, 9)),
                before_images_cam0=empty_capture_stacks(len(command_delta)),
                after_images_cam0=empty_capture_stacks(len(command_delta)),
                before_images_cam1=empty_capture_stacks(len(command_delta)),
                after_images_cam1=empty_capture_stacks(len(command_delta)),
                command_delta_mm=command_delta,
                pre_commanded_position_mm=pre,
                post_commanded_position_mm=post,
                pre_readback_position_mm=pre + 100.0,
                post_readback_position_mm=post - 50.0,
                min_shift_px=0.0,
                n_jobs=1,
            )

        np.testing.assert_allclose(
            calibration["visual_jacobian_px_per_cmd_mm"].values,
            expected,
            atol=1e-10,
        )

    def test_fitted_visual_calibration_preserves_reference_image_dtype(self):
        command_delta = probe_deltas()
        expected = visual_jacobian()
        measured = measured_from_jacobian(command_delta, expected)
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        pre = np.zeros_like(command_delta)
        post = pre + command_delta
        reference_cam0 = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9)
        reference_cam1 = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9) + np.uint16(
            100
        )
        capture_stacks = [
            np.zeros((1, 8, 9), dtype=np.uint16) for _ in range(len(command_delta))
        ]
        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            calibration = fit_visual_jacobian_calibration(
                reference_cam0=reference_cam0,
                reference_cam1=reference_cam1,
                before_images_cam0=capture_stacks,
                after_images_cam0=capture_stacks,
                before_images_cam1=capture_stacks,
                after_images_cam1=capture_stacks,
                command_delta_mm=command_delta,
                pre_commanded_position_mm=pre,
                post_commanded_position_mm=post,
                pre_readback_position_mm=pre,
                post_readback_position_mm=post,
                min_shift_px=0.0,
                n_jobs=1,
            )

        self.assertEqual(calibration["reference_cam0"].dtype, np.dtype(np.uint16))
        self.assertEqual(calibration["reference_cam1"].dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(
            calibration["reference_cam0"].values,
            reference_cam0,
        )
        np.testing.assert_array_equal(
            calibration["reference_cam1"].values,
            reference_cam1,
        )

    def test_axis_scale_is_derived_from_jacobian_and_clamped(self):
        command_delta = probe_deltas()
        jacobian = np.array(
            [
                [[1000.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                [[0.0, 0.0, 100.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )
        measured = measured_from_jacobian(command_delta, jacobian)
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        pre = np.zeros_like(command_delta)
        post = pre + command_delta
        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            calibration = fit_visual_jacobian_calibration(
                reference_cam0=np.zeros((8, 9)),
                reference_cam1=np.zeros((8, 9)),
                before_images_cam0=empty_capture_stacks(len(command_delta)),
                after_images_cam0=empty_capture_stacks(len(command_delta)),
                before_images_cam1=empty_capture_stacks(len(command_delta)),
                after_images_cam1=empty_capture_stacks(len(command_delta)),
                command_delta_mm=command_delta,
                pre_commanded_position_mm=pre,
                post_commanded_position_mm=post,
                pre_readback_position_mm=pre,
                post_readback_position_mm=post,
                min_shift_px=0.0,
                condition_warning_threshold=1.0e6,
                n_jobs=1,
            )

        (
            derived_axis_scale,
            axis_sensitivity,
            axis_scale_unclamped,
            _axis_scale_bounds,
            _target_response,
        ) = derive_axis_scale_from_jacobian(
            calibration["visual_jacobian_px_per_cmd_mm"].values,
            calibration["probe_command_delta_mm"].values,
        )
        np.testing.assert_allclose(axis_sensitivity, [1000.0, 10.0, 100.0], atol=1e-9)
        np.testing.assert_allclose(axis_scale_unclamped, [0.03, 3.0, 0.3], atol=1e-9)
        np.testing.assert_allclose(derived_axis_scale, [0.1, 1.0, 0.3], atol=1e-9)
        np.testing.assert_allclose(
            calibration["axis_scale_cmd_mm"].values,
            [0.1, 1.0, 0.3],
            atol=1e-9,
        )

    def test_noisy_readback_does_not_change_fitted_jacobian(self):
        dataset = calibration_dataset()
        np.testing.assert_allclose(
            dataset["probe_measured_delta_px"].values,
            measured_from_jacobian(
                dataset["probe_command_delta_mm"].values,
                dataset["visual_jacobian_px_per_cmd_mm"].values,
            ),
        )

    def test_saved_calibration_drops_redundant_derived_fields(self):
        redundant_vars = (
            "axis_scale_unclamped_cmd_mm",
            "axis_sensitivity_px_per_cmd_mm",
            "axis_scale_bounds_cmd_mm",
            "axis_scale_target_response_px",
            "probe_predicted_delta_px",
            "probe_residual_delta_px",
            "probe_before_cam0",
            "probe_after_cam0",
            "probe_before_cam1",
            "probe_after_cam1",
        )
        redundant_attrs = (
            "condition_number",
            "residual_rms_px",
            "residual_max_px",
        )
        dataset = (
            calibration_dataset()
            .assign(
                {
                    "axis_sensitivity_px_per_cmd_mm": (
                        ("command_axis",),
                        np.ones(len(COMMAND_AXES)),
                    ),
                    "probe_predicted_delta_px": (
                        ("probe", "camera", "pixel_axis"),
                        np.zeros(
                            (
                                calibration_dataset().sizes["probe"],
                                len(CAMERAS),
                                len(PIXEL_AXES),
                            )
                        ),
                    ),
                }
            )
            .assign_attrs(
                {
                    "condition_number": 1.0,
                    "residual_rms_px": 0.0,
                    "residual_max_px": 0.0,
                }
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            save_calibration_dataset(dataset, path)
            with xr.open_dataset(path, engine="h5netcdf") as raw:
                raw_dataset = raw.load()
            loaded = load_calibration_dataset(path)

        for name in redundant_vars:
            self.assertNotIn(name, raw_dataset)
            self.assertNotIn(name, loaded)
        for name in redundant_attrs:
            self.assertNotIn(name, raw_dataset.attrs)
        self.assertEqual(loaded.attrs["calibration_path"], str(path))

    def test_saved_calibration_compresses_reference_images_and_preserves_dtype(self):
        dataset = calibration_dataset()
        reference_cam0 = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
        reference_cam1 = np.arange(6 * 7, dtype=np.uint16).reshape(6, 7)
        dataset["reference_cam0"] = (("y_cam0", "x_cam0"), reference_cam0)
        dataset["reference_cam1"] = (("y_cam1", "x_cam1"), reference_cam1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            save_calibration_dataset(dataset, path)
            with h5py.File(path, "r") as saved_file:
                assert_hdf5_image_dataset_compressed(
                    self,
                    saved_file["reference_cam0"],
                    np.uint16,
                )
                assert_hdf5_image_dataset_compressed(
                    self,
                    saved_file["reference_cam1"],
                    np.uint16,
                )
                self.assertIsNone(saved_file["probe_command_delta_mm"].compression)
            loaded = load_calibration_dataset(path)

        self.assertEqual(loaded["reference_cam0"].dtype, np.dtype(np.uint16))
        self.assertEqual(loaded["reference_cam1"].dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(loaded["reference_cam0"].values, reference_cam0)
        np.testing.assert_array_equal(loaded["reference_cam1"].values, reference_cam1)

    def test_spooled_calibration_write_compresses_reference_images(self):
        dataset = calibration_dataset()
        dataset["reference_cam0"] = (
            ("y_cam0", "x_cam0"),
            np.arange(4 * 5, dtype=np.uint16).reshape(4, 5),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            spool_path = tmp_path / "spool"
            path = tmp_path / "calibration.h5"
            with (
                patch.dict(
                    os.environ,
                    {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
                ),
                patch(
                    "merlin_track_position.tracking.calibration_core."
                    "save_calibration_dataset",
                    side_effect=OSError("locked"),
                ),
            ):
                result = save_calibration_dataset_deferred(dataset, path)

            self.assertTrue(result.pending)
            self.assertIsNotNone(result.spool_path)
            assert result.spool_path is not None
            with h5py.File(result.spool_path / "data.h5", "r") as saved_file:
                assert_hdf5_image_dataset_compressed(
                    self,
                    saved_file["reference_cam0"],
                    np.uint16,
                )

    def test_stale_queued_calibration_write_is_not_flushed_over_changed_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            spool_path = tmp_path / "spool"
            path = tmp_path / "calibration.h5"
            original_save = calibration_core.save_calibration_dataset
            original_save(
                calibration_dataset().assign_attrs(jacobian_refinement_count=0),
                path,
            )

            queued = calibration_dataset().assign_attrs(jacobian_refinement_count=1)
            replacement = calibration_dataset().assign_attrs(
                jacobian_refinement_count=99
            )

            with patch.dict(
                os.environ,
                {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
            ):
                with patch(
                    "merlin_track_position.tracking.calibration_core."
                    "save_calibration_dataset",
                    side_effect=OSError("locked"),
                ):
                    result = save_calibration_dataset_deferred(queued, path)

                self.assertTrue(result.pending)
                original_save(replacement, path)
                flush_results = flush_pending_calibration_datasets(path)
                loaded = load_calibration_dataset(path)

            self.assertEqual(int(loaded.attrs["jacobian_refinement_count"]), 99)
            self.assertEqual(len(flush_results), 1)
            self.assertFalse(flush_results[0].flushed)
            self.assertFalse(flush_results[0].pending)

    def test_small_image_shifts_below_threshold_are_rejected(self):
        command_delta = np.eye(3)
        measured = np.zeros((3, 2, 2), dtype=float)
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        with (
            patch(
                "merlin_track_position.tracking.calibration_core.estimate_shift",
                side_effect=fake_estimate_shift,
            ),
            self.assertRaisesRegex(ValueError, "below threshold"),
        ):
            fit_visual_jacobian_calibration(
                reference_cam0=np.zeros((3, 3)),
                reference_cam1=np.zeros((3, 3)),
                before_images_cam0=empty_capture_stacks(3, shape=(3, 3)),
                after_images_cam0=empty_capture_stacks(3, shape=(3, 3)),
                before_images_cam1=empty_capture_stacks(3, shape=(3, 3)),
                after_images_cam1=empty_capture_stacks(3, shape=(3, 3)),
                command_delta_mm=command_delta,
                pre_commanded_position_mm=np.zeros((3, 3)),
                post_commanded_position_mm=command_delta,
                pre_readback_position_mm=np.zeros((3, 3)),
                post_readback_position_mm=command_delta,
                min_shift_px=2.0,
                n_jobs=1,
            )

    def test_rank_deficient_visual_jacobian_is_rejected(self):
        bad = calibration_dataset()
        bad["visual_jacobian_px_per_cmd_mm"] = (
            ("camera", "pixel_axis", "command_axis"),
            np.ones((2, 2, 3), dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "rank"):
            validate_visual_calibration_dataset(bad)

    def test_command_delta_mismatch_is_rejected(self):
        bad = calibration_dataset()
        bad["post_commanded_position_mm"] = (
            ("probe", "command_axis"),
            bad["post_commanded_position_mm"].values + 0.01,
        )

        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_visual_calibration_dataset(bad)

    def test_axis_scale_outside_configured_bounds_is_rejected(self):
        bad = calibration_dataset()
        bad["axis_scale_cmd_mm"] = (
            ("command_axis",),
            np.array([10.0, 0.5, 0.5], dtype=float),
        )

        with self.assertRaisesRegex(ValueError, "configured bounds"):
            validate_visual_calibration_dataset(bad)

    def test_nonfinite_probe_shift_measurements_are_rejected_with_probe_index(self):
        command_delta = np.eye(3)
        measured = np.ones((3, 2, 2), dtype=float)
        measured[1, 0, 0] = np.nan
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": "registration failed"},
            )

        with (
            patch(
                "merlin_track_position.tracking.calibration_core.estimate_shift",
                side_effect=fake_estimate_shift,
            ),
            self.assertRaisesRegex(ValueError, r"probe\(s\): 1"),
        ):
            fit_visual_jacobian_calibration(
                reference_cam0=np.zeros((3, 3)),
                reference_cam1=np.zeros((3, 3)),
                before_images_cam0=empty_capture_stacks(3, shape=(3, 3)),
                after_images_cam0=empty_capture_stacks(3, shape=(3, 3)),
                before_images_cam1=empty_capture_stacks(3, shape=(3, 3)),
                after_images_cam1=empty_capture_stacks(3, shape=(3, 3)),
                command_delta_mm=command_delta,
                pre_commanded_position_mm=np.zeros((3, 3)),
                post_commanded_position_mm=command_delta,
                pre_readback_position_mm=np.zeros((3, 3)),
                post_readback_position_mm=command_delta,
                min_shift_px=0.0,
                n_jobs=1,
            )


class CorrectionTests(unittest.TestCase):
    def save_calibration(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / "calibration.h5"
        save_calibration_dataset(calibration_dataset(), path)
        return path

    def patch_hardware(self, measurements, *, positions=(10.0, 20.0, 30.0)):
        measurement_iter = iter(measurements)
        return (
            patch(
                "merlin_track_position.tracking.correct.get_positions",
                return_value=positions,
            ),
            patch(
                "merlin_track_position.tracking.correct.capture_image_stack",
                return_value=(
                    np.zeros((1, 4, 5), dtype=float),
                    np.zeros((1, 6, 7), dtype=float),
                ),
            ),
            patch(
                "merlin_track_position.tracking.correct.measure_image_error",
                side_effect=lambda *args, **kwargs: next(measurement_iter),
            ),
        )

    def test_no_move_when_initial_residual_is_under_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            measurements = [shift_dataset(np.full((2, 2), 0.1))]
            hardware_patches = self.patch_hardware(measurements)
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    Mock(),
                ) as move,
            ):
                result = do_correction(path, capture_count=1)

        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(result.sizes["move"], 0)
        move.assert_not_called()

    def test_damped_correction_uses_command_state_not_readback_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.zeros((2, 2), dtype=float)
            expected_delta = solve_damped_command_correction(
                calibration_dataset()["visual_jacobian_px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration_dataset()["axis_scale_cmd_mm"].values,
                gain=0.3,
                damping_mu=constants.DEFAULT_CORRECTION_DAMPING_MU,
            )
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)],
                positions=(10.0, 20.0, 30.0),
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(100.0, 200.0, 300.0),
                ) as move,
            ):
                do_correction(path, capture_count=1, max_moves=1)

        requested = move.call_args.args[1]
        np.testing.assert_allclose(
            np.asarray(requested, dtype=float),
            np.array([10.0, 20.0, 30.0]) + expected_delta,
        )

    def test_normalized_step_limit_caps_largest_axis_component(self):
        jacobian = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )
        correction = solve_damped_command_correction(
            jacobian,
            shift_dataset(np.array([[10.0, 0.0], [0.0, 0.0]])),
            axis_scale_cmd_mm=[1.0, 1.0, 1.0],
            gain=1.0,
            damping_mu=0.0,
            max_normalized_step=0.5,
            min_axis_predicted_shift_px=0.0,
        )

        np.testing.assert_allclose(correction, [-0.5, 0.0, 0.0])

    def test_low_visual_impact_axis_components_are_zeroed(self):
        correction = solve_damped_command_correction(
            visual_jacobian(),
            shift_dataset(np.array([[2.0, 0.0], [0.0, 0.0]])),
            calibration_dataset()["axis_scale_cmd_mm"].values,
            gain=0.3,
            damping_mu=1e-2,
            min_axis_predicted_shift_px=0.25,
        )

        self.assertNotEqual(correction[0], 0.0)
        self.assertEqual(correction[1], 0.0)
        self.assertEqual(correction[2], 0.0)

    def test_tiny_correction_stops_without_motor_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = np.array([[1.2, 0.0], [0.0, 0.0]])
            hardware_patches = self.patch_hardware([shift_dataset(p0)])
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    Mock(),
                ) as move,
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

        move.assert_not_called()
        self.assertFalse(result.attrs["correction_converged"])
        self.assertIn(
            "estimated command offset is within correction deadbands",
            result.attrs["warnings"],
        )

    def test_offset_above_deadband_is_corrected_when_gained_step_is_small(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            offset_mm = np.array([0.0, 0.020, 0.0], dtype=float)
            p0 = (
                calibration["visual_jacobian_px_per_cmd_mm"].values.reshape(
                    len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)
                )
                @ offset_mm
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            expected_delta = solve_damped_command_correction(
                calibration["visual_jacobian_px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=0.3,
                damping_mu=constants.DEFAULT_CORRECTION_DAMPING_MU,
            )
            self.assertLess(
                abs(expected_delta[1]),
                constants.CORRECTION_COMMAND_DEADBAND_MM_BY_AXIS["y"],
            )
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    Mock(),
                ) as move,
            ):
                do_correction(path, capture_count=1, max_moves=1)

        self.assertEqual(move.call_args.args[0], ("y",))
        np.testing.assert_allclose(
            np.asarray(move.call_args.args[1], dtype=float),
            np.asarray([20.0 + expected_delta[1]], dtype=float),
        )

    def test_correction_command_deadband_uses_correction_specific_constants(self):
        correction = correct_module._zero_deadband_axis_corrections(
            np.array([0.005, 0.011, -0.0051], dtype=float)
        )

        np.testing.assert_allclose(correction, [0.0, 0.011, -0.0051])

    def test_zeroed_axes_are_not_sent_to_motor_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            offset_mm = np.array([0.020, 0.0, 0.0], dtype=float)
            p0 = (
                calibration["visual_jacobian_px_per_cmd_mm"].values.reshape(
                    len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)
                )
                @ offset_mm
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0,),
                ) as move,
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

        self.assertEqual(move.call_args.args[0], ("x",))
        self.assertEqual(
            result["move_active_axis_mask"].values.tolist(),
            [[True, False, False]],
        )

    def test_residual_improvement_refines_jacobian_and_updates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(300.0)
            p1 = x_shift(100.0)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ),
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

            reloaded = load_calibration_dataset(path)

        self.assertTrue(bool(result["move_jacobian_refined"].values[0]))
        self.assertEqual(int(reloaded.attrs["jacobian_refinement_count"]), 1)
        self.assertFalse(
            np.allclose(
                reloaded["visual_jacobian_px_per_cmd_mm"].values,
                visual_jacobian(),
            )
        )

    def test_jacobian_refinement_keeps_tiny_move_low_leverage(self):
        calibration = calibration_dataset()
        tiny_delta = np.array([[0.001, 0.0, 0.0]])
        bad_tiny_measurement = np.full((1, len(CAMERAS), len(PIXEL_AXES)), 5.0)

        refined = refine_visual_jacobian_from_observations(
            calibration,
            tiny_delta,
            bad_tiny_measurement,
        )

        np.testing.assert_allclose(
            refined["visual_jacobian_px_per_cmd_mm"].values,
            calibration["visual_jacobian_px_per_cmd_mm"].values,
            atol=0.1,
        )

    def test_correction_history_file_records_move_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(30.0)
            p1 = x_shift(10.0)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ),
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

            history_path = correction_history_path(path)
            with xr.open_dataset(
                history_path,
                engine="h5netcdf",
                group="run_000000",
            ) as saved_on_disk:
                saved = saved_on_disk.load()

        self.assertEqual(Path(result.attrs["correction_history_path"]), history_path)
        self.assertEqual(str(saved.attrs["calibration_path"]), str(path))
        self.assertEqual(int(saved.attrs["correction_history_completed"]), 1)
        self.assertEqual(saved.sizes["move"], 1)
        self.assertIn("move_measured_delta_px", saved)
        self.assertIn("move_predicted_delta_px", saved)
        self.assertIn("move_visual_jacobian_before_px_per_cmd_mm", saved)
        self.assertIn("move_visual_jacobian_after_px_per_cmd_mm", saved)

    def test_correction_history_compresses_current_images_and_preserves_dtype(self):
        result = shift_dataset(x_shift(1.0)).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
            }
        )
        current_cam0 = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
        current_cam1 = np.arange(6 * 7, dtype=np.uint16).reshape(6, 7)
        result["current_cam0"] = (("y_cam0", "x_cam0"), current_cam0)
        result["current_cam1"] = (("y_cam1", "x_cam1"), current_cam1)

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "correction-history.h5"
            correct_module.save_correction_history_dataset(
                result,
                history_path,
                run_id=0,
            )
            with h5py.File(history_path, "r") as history_file:
                group = history_file["run_000000"]
                assert_hdf5_image_dataset_compressed(
                    self,
                    group["current_cam0"],
                    np.uint16,
                )
                assert_hdf5_image_dataset_compressed(
                    self,
                    group["current_cam1"],
                    np.uint16,
                )
                self.assertIsNone(group["shift_px"].compression)
            with xr.open_dataset(
                history_path,
                engine="h5netcdf",
                group="run_000000",
            ) as loaded_on_disk:
                loaded = loaded_on_disk.load()

        self.assertEqual(loaded["current_cam0"].dtype, np.dtype(np.uint16))
        self.assertEqual(loaded["current_cam1"].dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(loaded["current_cam0"].values, current_cam0)
        np.testing.assert_array_equal(loaded["current_cam1"].values, current_cam1)

    def test_locked_correction_history_is_queued_and_flushes_later(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            path = self.save_calibration(tmpdir)
            history_path = correction_history_path(path)
            spool_path = tmp_path / "spool"
            with h5py.File(history_path, "w") as history_file:
                history_file.attrs["format"] = (
                    "merlin_track_position_correction_history"
                )
            history_reader = h5py.File(history_path, "r")
            try:
                p0 = x_shift(30.0)
                p1 = x_shift(10.0)
                hardware_patches = self.patch_hardware(
                    [shift_dataset(p0), shift_dataset(p1)]
                )
                with (
                    patch.dict(
                        os.environ,
                        {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
                    ),
                    hardware_patches[0],
                    hardware_patches[1],
                    hardware_patches[2],
                    patch(
                        "merlin_track_position.tracking.correct.move_motors_and_wait",
                        return_value=(10.0, 20.0, 30.0),
                    ),
                ):
                    result = do_correction(path, capture_count=1, max_moves=1)
                    pending = load_latest_correction_history_dataset(path)

                    self.assertEqual(
                        result.attrs["correction_history_persistence_status"],
                        "pending",
                    )
                    self.assertIsNotNone(pending)
                    assert pending is not None
                    self.assertEqual(
                        pending.attrs["correction_history_persistence_status"],
                        "pending",
                    )
                    self.assertTrue(pending.attrs["correction_history_completed"])
            finally:
                history_reader.close()

            with patch.dict(
                os.environ,
                {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
            ):
                flush_results = flush_pending_correction_history_datasets(history_path)
                loaded = load_latest_correction_history_dataset(path)
                remaining_pending = pending_entry_count()

            self.assertTrue(any(result.flushed for result in flush_results))
            self.assertEqual(remaining_pending, 0)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.attrs["correction_history_completed"])
            self.assertEqual(loaded.sizes["move"], 1)

    def test_pending_history_is_used_for_refinement_before_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            path = self.save_calibration(tmpdir)
            history_path = correction_history_path(path)
            spool_path = tmp_path / "spool"
            with h5py.File(history_path, "w") as history_file:
                history_file.attrs["format"] = (
                    "merlin_track_position_correction_history"
                )
            history_reader = h5py.File(history_path, "r")
            try:
                p0 = x_shift(30.0)
                p1 = x_shift(10.0)
                first_patches = self.patch_hardware(
                    [shift_dataset(p0), shift_dataset(p1)]
                )
                with (
                    patch.dict(
                        os.environ,
                        {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
                    ),
                    first_patches[0],
                    first_patches[1],
                    first_patches[2],
                    patch(
                        "merlin_track_position.tracking.correct.move_motors_and_wait",
                        return_value=(10.0, 20.0, 30.0),
                    ),
                ):
                    do_correction(path, capture_count=1, max_moves=1)

                observed_refinement_rows: list[int] = []
                original_refine = (
                    correct_module.refine_visual_jacobian_from_observations
                )

                def spy_refine(calibration, correction_delta, measured_delta, **kwargs):
                    observed_refinement_rows.append(
                        np.asarray(correction_delta).shape[0]
                    )
                    return original_refine(
                        calibration,
                        correction_delta,
                        measured_delta,
                        **kwargs,
                    )

                second_patches = self.patch_hardware(
                    [shift_dataset(p0), shift_dataset(p1)]
                )
                with (
                    patch.dict(
                        os.environ,
                        {"MERLIN_TRACK_POSITION_SPOOL_DIR": str(spool_path)},
                    ),
                    second_patches[0],
                    second_patches[1],
                    second_patches[2],
                    patch(
                        "merlin_track_position.tracking.correct.move_motors_and_wait",
                        return_value=(10.0, 20.0, 30.0),
                    ),
                    patch(
                        "merlin_track_position.tracking.correct."
                        "refine_visual_jacobian_from_observations",
                        side_effect=spy_refine,
                    ),
                ):
                    do_correction(path, capture_count=1, max_moves=1)
            finally:
                history_reader.close()

        self.assertEqual(observed_refinement_rows[0], 2)

    def test_progress_callback_receives_intermediate_correction_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(30.0)
            p1 = x_shift(10.0)
            progress_results = []
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )

            def fake_move_motors_and_wait(*args, **kwargs):
                self.assertEqual(len(progress_results), 1)
                first_progress = progress_results[0]
                self.assertFalse(first_progress.attrs["correction_history_completed"])
                self.assertEqual(first_progress.sizes["move"], 0)
                self.assertIn("correction_cmd_mm", first_progress)
                return (10.0, 20.0, 30.0)

            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    side_effect=fake_move_motors_and_wait,
                ),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    max_moves=1,
                    progress_callback=progress_results.append,
                )

        self.assertEqual(len(progress_results), 2)
        initial_progress, progress = progress_results
        self.assertFalse(initial_progress.attrs["correction_history_completed"])
        self.assertEqual(initial_progress.sizes["move"], 0)
        self.assertFalse(progress.attrs["correction_history_completed"])
        self.assertEqual(progress.sizes["move"], 1)
        self.assertEqual(result.attrs["correction_history_completed"], True)

    def test_latest_correction_history_dataset_can_be_reloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(30.0)
            p1 = x_shift(10.0)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ),
            ):
                do_correction(path, capture_count=1, max_moves=1)

            loaded = load_latest_correction_history_dataset(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.attrs["correction_history_completed"])
        self.assertTrue(loaded.attrs["correction_applied"])
        self.assertEqual(loaded.sizes["move"], 1)
        self.assertEqual(loaded["move_jacobian_refined"].dtype, bool)

    def test_residual_increase_reduces_gain_increases_damping_and_skips_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(30.0)
            p1 = x_shift(40.0)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ),
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

            reloaded = load_calibration_dataset(path)

        self.assertFalse(bool(result["move_jacobian_refined"].values[0]))
        self.assertAlmostEqual(float(result.attrs["correction_final_gain"]), 0.15)
        self.assertAlmostEqual(float(result.attrs["correction_final_damping_mu"]), 2.0)
        self.assertEqual(int(reloaded.attrs["jacobian_refinement_count"]), 0)

    def test_non_convergence_returns_false_without_raising_after_motion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = x_shift(30.0)
            p1 = x_shift(20.0)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ),
            ):
                result = do_correction(path, capture_count=1, max_moves=1)

        self.assertFalse(result.attrs["correction_converged"])
        self.assertIn("did not converge", result.attrs["warnings"])

    def test_correction_requires_saved_calibration_path(self):
        with self.assertRaisesRegex(ValueError, "requires a calibration file path"):
            do_correction(calibration_dataset(), capture_count=1)

    def test_zero_weight_objective_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            weighted_pixel_residual(
                shift_dataset(np.ones((2, 2))),
                weights=[0, 0, 0, 0],
            )


if __name__ == "__main__":
    unittest.main()
