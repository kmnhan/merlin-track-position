import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CallableCameraPlugin,
    CameraPairPlugin,
    crop_image_to_roi,
)
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    PIXEL_AXES,
    derive_axis_scale_from_jacobian,
    flush_pending_calibration_datasets,
    fit_jacobian_calibration,
    load_calibration_dataset,
    save_calibration_dataset,
    save_calibration_dataset_deferred,
    solve_lqr_command_correction,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)
import merlin_track_position.tracking.calibration_core as calibration_core
import merlin_track_position.tracking.correct as correct_module
from merlin_track_position.tracking.calibrate import (
    _make_visual_probe_deltas,
    _make_visual_probe_offsets_um,
    run_calibration,
    visual_calibration_probe_count,
)
from merlin_track_position.tracking.correct import (
    correction_timestamps_from_history,
    correction_total_move_by_axis_from_history,
    correction_history_path,
    do_correction,
    flush_pending_correction_history_datasets,
    load_latest_correction_history_dataset,
)
from merlin_track_position.tracking.detect import detect_shift
from merlin_track_position.tracking.persistence import pending_entry_count

TEST_CORRECTION_GAIN = 0.6
TEST_CORRECTION_MAX_NORMALIZED_STEP = 0.5
TEST_CORRECTION_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
TEST_LQR_IMAGE_SCALE_PX = 0.1
TEST_LQR_MOTOR_PENALTY = 100.0
TEST_LQR_SVD_RELATIVE_TOLERANCE = 1e-6


def px_per_cmd_mm():
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
        step = 0.03
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


def absolute_command_positions(command_delta, center=(1.0, 2.0, 3.0)):
    center_array = np.asarray(center, dtype=float)
    current = center_array.copy()
    pre_rows = []
    post_rows = []
    for offset in np.asarray(command_delta, dtype=float):
        pre_rows.append(current.copy())
        current = center_array + offset
        post_rows.append(current.copy())
    return np.asarray(pre_rows, dtype=float), np.asarray(post_rows, dtype=float)


def empty_capture_stacks(probe_count, shape=(8, 9), capture_count=1):
    return [
        np.zeros((capture_count, *shape), dtype=np.float32)
        for _ in range(probe_count)
    ]


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
        jacobian = px_per_cmd_mm()
    command_delta = probe_deltas()
    measured = measured_from_jacobian(command_delta, jacobian)
    pre_commanded = np.cumsum(
        np.vstack([np.zeros((1, 3)), command_delta[:-1]]),
        axis=0,
    )
    post_commanded = pre_commanded + command_delta
    dataset = xr.Dataset(
        data_vars={
            "px_per_cmd_mm": (
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
        attrs={"warnings": ""},
    )
    validate_visual_calibration_dataset(dataset)
    return dataset


class ShiftDetectionTests(unittest.TestCase):
    def test_detect_shift_reports_signed_offset_without_history_or_mutation(self):
        offset_mm = np.asarray([0.004, -0.005, 0.006], dtype=float)
        calibration = calibration_dataset()
        shift = measured_from_jacobian(
            offset_mm.reshape(1, len(COMMAND_AXES)),
            calibration["px_per_cmd_mm"].values,
        )[0]
        measurement = shift_dataset(shift).assign_attrs(
            {"warnings": "registration warning"}
        )
        attrs_before = dict(calibration.attrs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = calibration.assign_attrs({"calibration_path": str(path)})
            attrs_before |= {"calibration_path": str(path)}

            with patch(
                "merlin_track_position.tracking.detect._capture_measurement",
                return_value=measurement,
            ) as capture:
                result = detect_shift(calibration, object(), capture_count=1)

            capture.assert_called_once()
            self.assertEqual(capture.call_args.args[4], 1)
            np.testing.assert_allclose(
                result["estimated_command_offset_mm"].values,
                offset_mm,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result["detected_shift_um"].values,
                1000.0 * offset_mm,
                atol=1e-9,
            )
            self.assertAlmostEqual(
                float(result["weighted_residual_px"].values),
                weighted_pixel_residual(measurement),
            )
            self.assertEqual(result.attrs["warnings"], "registration warning")
            self.assertEqual(dict(calibration.attrs), attrs_before)
            self.assertFalse(correction_history_path(path).exists())

    def test_detect_shift_uses_calibration_beam_targets_as_roi_local_ecc_points(self):
        calibration = calibration_dataset().assign_attrs(
            {
                "roi_cam0_x": 1.0,
                "roi_cam0_y": 1.0,
                "roi_cam0_width": 2.0,
                "roi_cam0_height": 2.0,
                "roi_cam1_x": 2.0,
                "roi_cam1_y": 1.0,
                "roi_cam1_width": 3.0,
                "roi_cam1_height": 2.0,
                "beam_target_cam0_u": 2.0,
                "beam_target_cam0_v": 2.0,
                "beam_target_cam1_u": 4.0,
                "beam_target_cam1_v": 2.0,
            }
        )

        with (
            patch(
                "merlin_track_position.tracking.correct.capture_image_stack",
                return_value=(
                    np.zeros((1, 4, 5), dtype=float),
                    np.zeros((1, 6, 7), dtype=float),
                ),
            ),
            patch(
                "merlin_track_position.tracking.correct.measure_image_error",
                return_value=shift_dataset(np.zeros((2, 2), dtype=float)),
            ) as measure,
        ):
            detect_shift(
                calibration,
                object(),
                capture_count=1,
                use_ecc_refinement=True,
            )

        kwargs = measure.call_args.kwargs
        self.assertTrue(kwargs["use_ecc_refinement"])
        np.testing.assert_allclose(
            kwargs["ecc_reference_point_px"]["cam0"],
            [1.0, 1.0],
        )
        np.testing.assert_allclose(
            kwargs["ecc_reference_point_px"]["cam1"],
            [2.0, 1.0],
        )


class VisualCalibrationTests(unittest.TestCase):
    def test_capture_count_defaults_are_split_by_workflow(self):
        self.assertEqual(
            run_calibration.__kwdefaults__["capture_count"],
            constants.DEFAULT_CALIBRATION_CAPTURE_COUNT,
        )
        self.assertEqual(
            do_correction.__kwdefaults__["capture_count"],
            constants.DEFAULT_CORRECTION_CAPTURE_COUNT,
        )
        self.assertEqual(
            detect_shift.__kwdefaults__["capture_count"],
            constants.DEFAULT_CORRECTION_CAPTURE_COUNT,
        )

    def test_measure_image_error_median_registers_each_capture(self):
        shifts = iter(
            (
                np.asarray([1.0, 2.0]),
                np.asarray([3.0, 4.0]),
                np.asarray([5.0, 6.0]),
                np.asarray([7.0, 8.0]),
            )
        )
        calls = []

        def fake_estimate_shift(reference, current, **kwargs):
            calls.append((np.asarray(reference).copy(), np.asarray(current).copy()))
            del kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shifts))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            result = calibration_core.measure_image_error(
                np.zeros((2, 2)),
                np.stack(
                    (
                        np.full((2, 2), 10.0),
                        np.full((2, 2), 20.0),
                    ),
                    axis=0,
                ),
                np.zeros((3, 3)),
                np.stack(
                    (
                        np.full((3, 3), 30.0),
                        np.full((3, 3), 40.0),
                    ),
                    axis=0,
                ),
            )

        self.assertEqual(len(calls), 4)
        np.testing.assert_allclose(
            result["shift_px"].values,
            [[2.0, 3.0], [6.0, 7.0]],
        )
        np.testing.assert_allclose(result["capture_shift_mad_px"].values, 1.0)
        self.assertEqual(result.attrs["capture_count"], 2)
        self.assertEqual(result.attrs["capture_aggregation"], "median_shifts")

    def test_measure_image_error_mean_image_registers_averaged_image_once(self):
        shifts = iter((np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])))
        calls = []

        def fake_estimate_shift(reference, current, **kwargs):
            calls.append((np.asarray(reference).copy(), np.asarray(current).copy()))
            del kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shifts))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": "warning"},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            result = calibration_core.measure_image_error(
                np.zeros((2, 2)),
                np.stack(
                    (
                        np.full((2, 2), 10.0),
                        np.full((2, 2), 20.0),
                    ),
                    axis=0,
                ),
                np.zeros((3, 3)),
                np.stack(
                    (
                        np.full((3, 3), 30.0),
                        np.full((3, 3), 40.0),
                    ),
                    axis=0,
                ),
                capture_aggregation="mean_image",
            )

        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][1], np.full((2, 2), 15.0))
        np.testing.assert_array_equal(calls[1][1], np.full((3, 3), 35.0))
        np.testing.assert_allclose(
            result["shift_px"].values,
            [[1.0, 2.0], [3.0, 4.0]],
        )
        np.testing.assert_allclose(result["capture_shift_mad_px"].values, 0.0)
        self.assertTrue(np.isfinite(result["capture_shift_mad_px"].values).all())
        self.assertEqual(result.attrs["capture_count"], 2)
        self.assertEqual(result.attrs["capture_aggregation"], "mean_image")
        self.assertIn("cam0 warning", result.attrs["warnings"])
        self.assertIn("cam1 warning", result.attrs["warnings"])

    def test_fit_calibration_mean_image_records_aggregation(self):
        command_delta = probe_deltas()
        expected = px_per_cmd_mm()
        measured = measured_from_jacobian(command_delta, expected)
        shift_values = iter(
            measured[probe_index, camera_index]
            for probe_index in range(command_delta.shape[0])
            for camera_index in range(len(CAMERAS))
        )
        calls = []
        progress = []

        def fake_estimate_shift(reference, current, **kwargs):
            calls.append((np.asarray(reference).copy(), np.asarray(current).copy()))
            del kwargs
            return xr.Dataset(
                {"shift_px": (("pixel_axis",), next(shift_values))},
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            result = fit_jacobian_calibration(
                reference_cam0=np.zeros((3, 3)),
                reference_cam1=np.zeros((3, 3)),
                before_images_cam0=empty_capture_stacks(
                    command_delta.shape[0],
                    shape=(3, 3),
                    capture_count=2,
                ),
                after_images_cam0=empty_capture_stacks(
                    command_delta.shape[0],
                    shape=(3, 3),
                    capture_count=2,
                ),
                before_images_cam1=empty_capture_stacks(
                    command_delta.shape[0],
                    shape=(3, 3),
                    capture_count=2,
                ),
                after_images_cam1=empty_capture_stacks(
                    command_delta.shape[0],
                    shape=(3, 3),
                    capture_count=2,
                ),
                command_delta_mm=command_delta,
                pre_commanded_position_mm=np.zeros((command_delta.shape[0], 3)),
                post_commanded_position_mm=command_delta,
                pre_readback_position_mm=np.zeros((command_delta.shape[0], 3)),
                post_readback_position_mm=command_delta,
                min_shift_px=0.0,
                capture_aggregation="mean_image",
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                ),
                n_jobs=1,
            )

        self.assertEqual(len(calls), command_delta.shape[0] * len(CAMERAS))
        self.assertEqual(progress[0], (0, command_delta.shape[0] * len(CAMERAS)))
        self.assertEqual(progress[-1], (command_delta.shape[0] * len(CAMERAS),) * 2)
        self.assertEqual(result.attrs["capture_count"], 2)
        self.assertEqual(result.attrs["capture_aggregation"], "mean_image")

    def test_visual_calibration_probe_count_uses_default_n(self):
        self.assertEqual(visual_calibration_probe_count(), 21)

    def test_visual_calibration_probe_offsets_match_old_backlash_path(self):
        offsets = _make_visual_probe_offsets_um(3, 10.0)
        expected_um = np.asarray(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, -10.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 10.0],
                [-10.0, 10.0, 10.0],
                [-10.0, 10.0, -10.0],
                [10.0, 10.0, -10.0],
                [10.0, -10.0, -10.0],
                [10.0, -10.0, 10.0],
                [-10.0, -10.0, 10.0],
                [-10.0, -10.0, -10.0],
                [0.0, -10.0, 0.0],
                [10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(offsets, expected_um)
        np.testing.assert_allclose(offsets[-1], 0.0, atol=1e-15)

    def test_visual_calibration_movement_deltas_minimize_negative_y_moves(self):
        deltas = np.asarray(_make_visual_probe_deltas(3, 10.0), dtype=float)
        np.testing.assert_allclose(np.sum(deltas, axis=0), 0.0, atol=1e-15)
        self.assertEqual(np.count_nonzero(deltas[:, 1] < 0.0), 1)

    def test_visual_calibration_probe_offsets_cover_old_backlash_path(self):
        offsets = _make_visual_probe_offsets_um(5, 10.0)
        expected_offsets = (-20.0, -10.0, 10.0, 20.0)
        expected_rows = {
            (0.0, 0.0, z_offset) for z_offset in expected_offsets
        } | {
            (0.0, y_offset, 0.0) for y_offset in expected_offsets
        } | {
            (x_offset, 0.0, 0.0) for x_offset in expected_offsets
        } | {
            (x_offset, y_offset, z_offset)
            for x_offset in (-10.0, 10.0)
            for y_offset in (-10.0, 10.0)
            for z_offset in (-10.0, 10.0)
        } | {(0.0, 0.0, 0.0)}

        self.assertEqual({tuple(row) for row in offsets}, expected_rows)
        self.assertEqual(offsets.shape[0], len(expected_rows))

    def test_run_calibration_stores_full_references_and_fits_cropped_images(self):
        full_cam0 = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
        full_cam1 = np.arange(7 * 8, dtype=np.uint16).reshape(7, 8) + 100
        roi_metadata = {
            "roi_cam0_x": 1.0,
            "roi_cam0_y": 2.0,
            "roi_cam0_width": 3.0,
            "roi_cam0_height": 2.0,
            "roi_cam1_x": 2.0,
            "roi_cam1_y": 1.0,
            "roi_cam1_width": 2.0,
            "roi_cam1_height": 3.0,
        }
        expected_crop_cam0 = crop_image_to_roi(
            full_cam0,
            (
                roi_metadata["roi_cam0_x"],
                roi_metadata["roi_cam0_y"],
                roi_metadata["roi_cam0_width"],
                roi_metadata["roi_cam0_height"],
            ),
        )
        expected_crop_cam1 = crop_image_to_roi(
            full_cam1,
            (
                roi_metadata["roi_cam1_x"],
                roi_metadata["roi_cam1_y"],
                roi_metadata["roi_cam1_width"],
                roi_metadata["roi_cam1_height"],
            ),
        )
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: full_cam0),
            CallableCameraPlugin("cam1", lambda: full_cam1),
        )
        fit_calls = []
        expected_offsets = _make_visual_probe_offsets_um(3, 10.0) / 1000.0

        def fake_fit_jacobian_calibration(**kwargs):
            fit_calls.append(kwargs)
            np.testing.assert_array_equal(kwargs["reference_cam0"], expected_crop_cam0)
            np.testing.assert_array_equal(kwargs["reference_cam1"], expected_crop_cam1)
            np.testing.assert_allclose(kwargs["command_delta_mm"], expected_offsets)
            np.testing.assert_allclose(
                kwargs["post_commanded_position_mm"],
                expected_offsets,
            )
            self.assertIs(kwargs["check_tiles"], True)
            self.assertEqual(
                kwargs["additional_context"]["probe_command_delta_mode"],
                "absolute_center_offset",
            )
            for name, expected_shape in (
                ("before_images_cam0", (1, *expected_crop_cam0.shape)),
                ("after_images_cam0", (1, *expected_crop_cam0.shape)),
                ("before_images_cam1", (1, *expected_crop_cam1.shape)),
                ("after_images_cam1", (1, *expected_crop_cam1.shape)),
            ):
                for stack in kwargs[name]:
                    self.assertEqual(stack.shape, expected_shape)
            for stack in kwargs["before_images_cam0"]:
                np.testing.assert_array_equal(stack[0], expected_crop_cam0)
            for stack in kwargs["before_images_cam1"]:
                np.testing.assert_array_equal(stack[0], expected_crop_cam1)
            return calibration_dataset().assign_attrs(kwargs["additional_context"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.h5"
            with (
                patch(
                    "merlin_track_position.tracking.calibrate.get_positions",
                    return_value=(0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
                ),
                patch(
                    "merlin_track_position.tracking.calibrate.move_motors_and_wait",
                    side_effect=lambda _axes, goals: tuple(goals),
                ),
                patch("merlin_track_position.tracking.calibrate.time.sleep") as sleep,
                patch(
                    "merlin_track_position.tracking.calibrate.fit_jacobian_calibration",
                    side_effect=fake_fit_jacobian_calibration,
                ),
            ):
                calibration = run_calibration(
                    camera_pair,
                    output_path=output_path,
                    n=3,
                    step_um=10.0,
                    capture_count=1,
                    additional_context=roi_metadata,
                )

        self.assertEqual(len(fit_calls), 1)
        sleep.assert_not_called()
        np.testing.assert_array_equal(calibration["reference_cam0"].values, full_cam0)
        np.testing.assert_array_equal(calibration["reference_cam1"].values, full_cam1)
        self.assertEqual(calibration["reference_cam0"].shape, full_cam0.shape)
        self.assertEqual(calibration["reference_cam1"].shape, full_cam1.shape)
        self.assertEqual(calibration.attrs["roi_cam0_width"], 3.0)
        self.assertEqual(calibration.attrs["roi_cam1_height"], 3.0)

    def test_run_calibration_keeps_camera_switch_sleep(self):
        image_cam0 = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
        image_cam1 = np.arange(6 * 7, dtype=np.uint16).reshape(6, 7)
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: image_cam0),
            CallableCameraPlugin("cam1", lambda: image_cam1),
        )

        def fake_get_positions(aliases):
            aliases = tuple(aliases)
            if aliases == ("x", "y", "z", "p", "t", "cam"):
                return (0.0, 0.0, 0.0, 0.0, 0.0, 4.0)
            return tuple(0.0 for _alias in aliases)

        def fake_fit_jacobian_calibration(**kwargs):
            return calibration_dataset().assign_attrs(kwargs["additional_context"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.h5"
            with (
                patch(
                    "merlin_track_position.tracking.calibrate.get_positions",
                    side_effect=fake_get_positions,
                ),
                patch(
                    "merlin_track_position.tracking.calibrate.move_motors_and_wait",
                    side_effect=lambda _axes, goals: tuple(goals),
                ),
                patch("merlin_track_position.tracking.calibrate.time.sleep") as sleep,
                patch(
                    "merlin_track_position.tracking.calibrate.fit_jacobian_calibration",
                    side_effect=fake_fit_jacobian_calibration,
                ),
            ):
                run_calibration(
                    camera_pair,
                    output_path=output_path,
                    n=2,
                    step_um=10.0,
                    capture_count=1,
                )

        sleep.assert_called_once_with(4.0)

    def test_fitted_px_per_cmd_mm_matches_synthetic_response(self):
        command_delta = probe_deltas()
        expected = px_per_cmd_mm()
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
            calibration = fit_jacobian_calibration(
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
            calibration["px_per_cmd_mm"].values,
            expected,
            atol=1e-10,
        )

    def test_fitted_absolute_center_offsets_accept_zero_closure_row(self):
        command_delta = _make_visual_probe_offsets_um(3, 10.0) / 1000.0
        expected = px_per_cmd_mm()
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

        pre, post = absolute_command_positions(command_delta)
        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            calibration = fit_jacobian_calibration(
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
                min_shift_px=0.2,
                additional_context={
                    "initial_x_mm": 1.0,
                    "initial_y_mm": 2.0,
                    "initial_z_mm": 3.0,
                    "probe_command_delta_mode": "absolute_center_offset",
                },
                n_jobs=1,
            )

        self.assertEqual(
            calibration.attrs["probe_command_delta_mode"],
            "absolute_center_offset",
        )
        np.testing.assert_allclose(command_delta[-1], 0.0, atol=1e-15)
        np.testing.assert_allclose(
            calibration["px_per_cmd_mm"].values,
            expected,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            calibration["probe_measured_delta_px"].values[-1],
            0.0,
            atol=1e-15,
        )

    def test_fitted_px_per_cmd_mm_uses_raw_repeats_and_warns_on_transient_spread(
        self,
    ):
        command_delta = np.asarray(_make_visual_probe_deltas(5, 15.0), dtype=float)
        expected = px_per_cmd_mm()
        measured = measured_from_jacobian(command_delta, expected)
        biased_observation = measured.reshape(command_delta.shape[0], 4).copy()
        command_groups: dict[tuple[float, ...], list[int]] = {}
        for probe_index, command_row in enumerate(command_delta):
            command_groups.setdefault(tuple(command_row), []).append(probe_index)
        repeated_probe_indices = next(
            indices for indices in command_groups.values() if len(indices) >= 2
        )
        transient = np.array([3.2, 1.6, 2.8, -8.0], dtype=float)
        biased_observation[repeated_probe_indices[0]] += transient
        biased_observation[repeated_probe_indices[1]] -= transient
        biased_measured = biased_observation.reshape(command_delta.shape[0], 2, 2)
        shift_values = iter(
            biased_measured[probe_index, camera_index]
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
            calibration = fit_jacobian_calibration(
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
                n_jobs=1,
            )

        raw_fit = calibration_core._fit_robust_calibration_response(
            command_delta,
            biased_observation,
        )
        raw_px_per_cmd_mm = raw_fit.T.reshape(len(CAMERAS), len(PIXEL_AXES), 3)
        np.testing.assert_allclose(
            calibration["px_per_cmd_mm"].values,
            raw_px_per_cmd_mm,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            calibration["probe_measured_delta_px"].values,
            biased_measured,
            atol=1e-12,
        )
        self.assertIn(
            "repeated probe response spread",
            calibration.attrs["warnings"],
        )

    def test_fitted_visual_calibration_preserves_reference_image_dtype(self):
        command_delta = probe_deltas()
        expected = px_per_cmd_mm()
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
            calibration = fit_jacobian_calibration(
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
        with (
            patch(
                "merlin_track_position.tracking.calibration_core.estimate_shift",
                side_effect=fake_estimate_shift,
            ),
            patch.object(constants, "DEFAULT_JACOBIAN_CONDITION_WARNING", 1.0e6),
        ):
            calibration = fit_jacobian_calibration(
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
            calibration["px_per_cmd_mm"].values,
            calibration["probe_command_delta_mm"].values,
        )
        np.testing.assert_allclose(axis_sensitivity, [1000.0, 10.0, 100.0], atol=1e-9)
        np.testing.assert_allclose(axis_scale_unclamped, [0.003, 0.3, 0.03], atol=1e-9)
        np.testing.assert_allclose(derived_axis_scale, [0.1, 0.3, 0.1], atol=1e-9)
        np.testing.assert_allclose(
            calibration["axis_scale_cmd_mm"].values,
            [0.1, 0.3, 0.1],
            atol=1e-9,
        )

    def test_noisy_readback_does_not_change_fitted_jacobian(self):
        dataset = calibration_dataset()
        np.testing.assert_allclose(
            dataset["probe_measured_delta_px"].values,
            measured_from_jacobian(
                dataset["probe_command_delta_mm"].values,
                dataset["px_per_cmd_mm"].values,
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
                calibration_dataset().assign_attrs(calibration_note="original"),
                path,
            )

            queued = calibration_dataset().assign_attrs(calibration_note="queued")
            replacement = calibration_dataset().assign_attrs(
                calibration_note="replacement"
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

            self.assertEqual(loaded.attrs["calibration_note"], "replacement")
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
            fit_jacobian_calibration(
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

    def test_rank_deficient_px_per_cmd_mm_is_rejected(self):
        bad = calibration_dataset()
        bad["px_per_cmd_mm"] = (
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

    def test_absolute_center_command_offsets_validate(self):
        dataset = calibration_dataset().assign_attrs(
            {
                "initial_x_mm": 1.0,
                "initial_y_mm": 2.0,
                "initial_z_mm": 3.0,
                "probe_command_delta_mode": "absolute_center_offset",
            }
        )
        command_delta = dataset["probe_command_delta_mm"].values
        pre, post = absolute_command_positions(command_delta)
        dataset["pre_commanded_position_mm"] = (
            ("probe", "command_axis"),
            pre,
        )
        dataset["post_commanded_position_mm"] = (
            ("probe", "command_axis"),
            post,
        )
        dataset["pre_readback_position_mm"] = (
            ("probe", "command_axis"),
            pre,
        )
        dataset["post_readback_position_mm"] = (
            ("probe", "command_axis"),
            post,
        )

        validate_visual_calibration_dataset(dataset)

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
            fit_jacobian_calibration(
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

    def test_capture_measurement_crops_full_references_and_current_images(self):
        calibration = calibration_dataset().assign_attrs(
            {
                "roi_cam0_x": 1.0,
                "roi_cam0_y": 1.0,
                "roi_cam0_width": 2.0,
                "roi_cam0_height": 2.0,
                "roi_cam1_x": 2.0,
                "roi_cam1_y": 1.0,
                "roi_cam1_width": 3.0,
                "roi_cam1_height": 2.0,
            }
        )
        reference_cam0 = np.asarray(calibration["reference_cam0"].values)
        reference_cam1 = np.asarray(calibration["reference_cam1"].values)
        current_cam0 = np.arange(1 * 4 * 5, dtype=float).reshape(1, 4, 5)
        current_cam1 = np.arange(1 * 6 * 7, dtype=float).reshape(1, 6, 7)
        expected_reference_cam0 = crop_image_to_roi(reference_cam0, (1, 1, 2, 2))
        expected_reference_cam1 = crop_image_to_roi(reference_cam1, (2, 1, 3, 2))
        expected_current_cam0 = crop_image_to_roi(current_cam0[0], (1, 1, 2, 2))
        expected_current_cam1 = crop_image_to_roi(current_cam1[0], (2, 1, 3, 2))

        with (
            patch(
                "merlin_track_position.tracking.correct.capture_image_stack",
                return_value=(current_cam0, current_cam1),
            ),
            patch(
                "merlin_track_position.tracking.correct.measure_image_error",
                return_value=shift_dataset(np.zeros((2, 2), dtype=float)),
            ) as measure,
        ):
            correct_module._capture_measurement(
                calibration,
                object(),
                reference_cam0,
                reference_cam1,
                1,
            )

        args = measure.call_args.args
        np.testing.assert_array_equal(args[0], expected_reference_cam0)
        np.testing.assert_array_equal(args[1], expected_current_cam0[np.newaxis, :, :])
        np.testing.assert_array_equal(args[2], expected_reference_cam1)
        np.testing.assert_array_equal(args[3], expected_current_cam1[np.newaxis, :, :])
        self.assertNotIn("check_tiles", measure.call_args.kwargs)

    def test_no_move_when_initial_residual_is_under_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            measurements = [shift_dataset(np.full((2, 2), 0.01))]
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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                )

            history_path = correction_history_path(path)
            with xr.open_dataset(
                history_path,
                engine="h5netcdf",
                group="run_000000",
            ) as saved_on_disk:
                saved = saved_on_disk.load()

        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(result.attrs["correction_mode"], "camera")
        self.assertNotIn("analyzer_offset_mm", result)
        self.assertNotIn("iteration_analyzer_offset_mm", result)
        self.assertEqual(result.sizes["move"], 0)
        self.assertNotIn("correction_move_started_at", result.attrs)
        self.assertNotIn("correction_move_finished_at", result.attrs)
        self.assertNotIn("correction_move_started_at", saved.attrs)
        self.assertNotIn("correction_move_finished_at", saved.attrs)
        move.assert_not_called()

    def test_beam_geometry_uses_polar_angle(self):
        zero_polar = calibration_dataset().assign_attrs(polar=0.0)
        geometry = correct_module._beam_geometry_from_calibration(
            zero_polar,
            beam_xz_angle_from_analyzer_deg=65.0,
        )
        angle = np.deg2rad(65.0)
        np.testing.assert_allclose(
            geometry["beam_unit"],
            [np.sin(angle), 0.0, np.cos(angle)],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            geometry["beam_transverse_unit"],
            [np.cos(angle), 0.0, -np.sin(angle)],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            geometry["analyzer_unit"],
            [0.0, 0.0, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            geometry["analyzer_transverse_unit"],
            [1.0, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(geometry["projection_matrix"], dtype=float),
            [
                [np.cos(angle), 0.0, -np.sin(angle)],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            atol=1e-12,
        )

        positive_polar = calibration_dataset().assign_attrs(polar=10.0)
        geometry = correct_module._beam_geometry_from_calibration(
            positive_polar,
            beam_xz_angle_from_analyzer_deg=65.0,
        )
        self.assertAlmostEqual(geometry["beam_xz_angle_deg"], 55.0)
        self.assertAlmostEqual(geometry["analyzer_xz_angle_deg"], -10.0)
        self.assertAlmostEqual(
            geometry["beam_xz_angle_deg"] - geometry["analyzer_xz_angle_deg"],
            65.0,
        )

    def test_beam_mode_requires_polar_but_camera_mode_accepts_old_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            hardware_patches = self.patch_hardware([shift_dataset(np.zeros((2, 2)))])
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                )

            with self.assertRaisesRegex(ValueError, "polar"):
                do_correction(path, object(), capture_count=1, correction_mode="beam")

        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(result.attrs["correction_mode"], "camera")

    def test_beam_mode_corrects_longitudinal_offset_from_analyzer_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = calibration_dataset().assign_attrs(polar=0.0)
            save_calibration_dataset(calibration, path)
            geometry = correct_module._beam_geometry_from_calibration(
                calibration,
                beam_xz_angle_from_analyzer_deg=65.0,
            )
            offset_mm = 0.020 * np.asarray(geometry["beam_unit"], dtype=float)
            shift = measured_from_jacobian(
                offset_mm.reshape(1, len(COMMAND_AXES)),
                calibration["px_per_cmd_mm"].values,
            )[0]
            hardware_patches = self.patch_hardware(
                [shift_dataset(shift), shift_dataset(np.zeros((2, 2)))],
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="beam",
                    max_moves=1,
                )

        move.assert_called_once()
        active_axes = tuple(move.call_args.args[0])
        requested = np.asarray(move.call_args.args[1], dtype=float)
        delta = np.zeros(len(COMMAND_AXES), dtype=float)
        initial = np.array([10.0, 20.0, 30.0], dtype=float)
        for axis, goal in zip(active_axes, requested, strict=True):
            index = COMMAND_AXES.index(axis)
            delta[index] = goal - initial[index]
        beam_unit = np.asarray(geometry["beam_unit"], dtype=float)
        analyzer_transverse = np.asarray(
            geometry["analyzer_transverse_unit"],
            dtype=float,
        )
        self.assertLess(float(beam_unit @ delta), 0.0)
        self.assertLess(float(analyzer_transverse @ delta), 0.0)
        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(result.attrs["correction_mode"], "beam")
        self.assertEqual(
            result.attrs["correction_criterion"],
            "beam_analyzer_projected_normalized_error",
        )
        np.testing.assert_allclose(
            result["beam_offset_mm"].values[:2],
            [0.0, 0.0],
            atol=1e-12,
        )
        self.assertAlmostEqual(result["beam_offset_mm"].values[2], 0.0)
        self.assertAlmostEqual(
            result["iteration_beam_offset_mm"]
            .isel(iteration=0)
            .sel(beam_axis="beam_longitudinal")
            .item(),
            0.020,
        )
        self.assertIn("analyzer_offset_mm", result)
        self.assertIn("iteration_analyzer_offset_mm", result)
        self.assertAlmostEqual(
            result["iteration_analyzer_offset_mm"]
            .isel(iteration=0)
            .sel(analyzer_axis="analyzer_transverse")
            .item(),
            np.sin(np.deg2rad(65.0)) * 0.020,
        )
        self.assertAlmostEqual(
            result["analyzer_offset_mm"]
            .sel(analyzer_axis="analyzer_transverse")
            .item(),
            0.0,
        )
        self.assertEqual(
            result.attrs["correction_beam_observation_axes"],
            "beam_transverse analyzer_transverse vertical",
        )
        self.assertEqual(result.attrs["correction_analyzer_runtime_xz_angle_deg"], 0.0)

    def test_beam_mode_does_not_move_when_dual_view_residual_is_within_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = calibration_dataset().assign_attrs(polar=0.0)
            save_calibration_dataset(calibration, path)
            geometry = correct_module._beam_geometry_from_calibration(
                calibration,
                beam_xz_angle_from_analyzer_deg=65.0,
            )
            offset_mm = 0.0002 * np.asarray(geometry["beam_unit"], dtype=float)
            shift = measured_from_jacobian(
                offset_mm.reshape(1, len(COMMAND_AXES)),
                calibration["px_per_cmd_mm"].values,
            )[0]
            hardware_patches = self.patch_hardware([shift_dataset(shift)])
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    Mock(),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="beam",
                )

        move.assert_not_called()
        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(result.sizes["move"], 0)
        self.assertLessEqual(
            float(result["iteration_correction_criterion_residual"].values[-1]),
            result.attrs["correction_tolerance"],
        )

    def test_beam_mode_corrects_transverse_and_vertical_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = calibration_dataset().assign_attrs(polar=0.0)
            save_calibration_dataset(calibration, path)
            geometry = correct_module._beam_geometry_from_calibration(
                calibration,
                beam_xz_angle_from_analyzer_deg=65.0,
            )
            transverse = np.asarray(geometry["beam_transverse_unit"], dtype=float)
            offset_mm = 0.010 * transverse + np.array([0.0, 0.005, 0.0])
            shift = measured_from_jacobian(
                offset_mm.reshape(1, len(COMMAND_AXES)),
                calibration["px_per_cmd_mm"].values,
            )[0]
            hardware_patches = self.patch_hardware(
                [shift_dataset(shift), shift_dataset(np.zeros((2, 2)))],
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0, 20.0, 30.0),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="beam",
                    max_moves=1,
                )

        requested = np.asarray(move.call_args.args[1], dtype=float)
        delta = requested - np.array([10.0, 20.0, 30.0])
        self.assertLess(float(transverse @ delta), 0.0)
        self.assertLess(float(delta[1]), 0.0)
        self.assertEqual(result.attrs["correction_mode"], "beam")
        self.assertIn("iteration_beam_offset_mm", result)
        self.assertIn("iteration_analyzer_offset_mm", result)
        self.assertFalse(result.attrs["correction_lqr_kalman_filter_enabled"])

    def test_correction_uses_command_state_not_readback_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.zeros((2, 2), dtype=float)
            expected_delta = solve_lqr_command_correction(
                calibration_dataset()["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration_dataset()["axis_scale_cmd_mm"].values,
                gain=TEST_CORRECTION_GAIN,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                weights=TEST_CORRECTION_WEIGHTS,
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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        requested = move.call_args.args[1]
        np.testing.assert_allclose(
            np.asarray(requested, dtype=float),
            np.array([10.0, 20.0, 30.0]) + expected_delta,
        )
        self.assertEqual(
            result.attrs["correction_criterion"],
            "lqr_projected_normalized_error",
        )

    def test_correction_can_delegate_iterative_moves_to_motor_backend(self):
        class FakeMotorBackend:
            def __init__(self):
                self.positions = np.array([10.0, 20.0, 30.0], dtype=float)
                self.moves = []
                self.backlash_corrections = []

            def get_positions(self, motor_aliases):
                events.append(("get", tuple(motor_aliases)))
                return tuple(
                    self.positions[COMMAND_AXES.index(axis)] for axis in motor_aliases
                )

            def move_motors_and_wait(
                self,
                motor_aliases,
                goals,
                *,
                max_retries=4,
                backlash_correction=None,
                move_timeout_s=60.0,
            ):
                events.append(("move", tuple(motor_aliases)))
                self.moves.append((tuple(motor_aliases), tuple(goals)))
                self.backlash_corrections.append(backlash_correction)
                for axis, goal in zip(motor_aliases, goals, strict=True):
                    self.positions[COMMAND_AXES.index(axis)] = float(goal)
                return tuple(goals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            measurements = iter(
                [
                    shift_dataset(x_shift(300.0)),
                    shift_dataset(x_shift(100.0)),
                    shift_dataset(x_shift(0.0)),
                ]
            )
            events = []
            backend = FakeMotorBackend()

            def fake_measure_image_error(*args, **kwargs):
                events.append(("measure", None))
                return next(measurements)

            with (
                patch(
                    "merlin_track_position.tracking.correct._local_timestamp_iso",
                    side_effect=(
                        "2026-01-01T00:00:01-08:00",
                        "2026-01-01T00:00:02-08:00",
                        "2026-01-01T00:00:03-08:00",
                    ),
                ),
                patch(
                    "merlin_track_position.tracking.correct.get_positions",
                    side_effect=AssertionError("direct BCS get should not be used"),
                ),
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    side_effect=AssertionError("direct BCS move should not be used"),
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
                    side_effect=fake_measure_image_error,
                ),
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER",
                    False,
                ),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=2,
                    motor_backend=backend,
                )

        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(len(backend.moves), 2)
        self.assertEqual(backend.backlash_corrections, [{}, {}])
        self.assertFalse(result.attrs["correction_backlash_enabled"])
        self.assertEqual(
            result.attrs["correction_move_started_at"],
            "2026-01-01T00:00:01-08:00",
        )
        self.assertEqual(
            result.attrs["correction_move_finished_at"],
            "2026-01-01T00:00:03-08:00",
        )
        self.assertEqual(events[0], ("get", tuple(COMMAND_AXES)))
        self.assertEqual(
            [event[0] for event in events],
            ["get", "measure", "move", "get", "measure", "move", "get", "measure"],
        )

    def test_lqr_correction_moves_commanded_mm_through_existing_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            expected_delta = solve_lqr_command_correction(
                calibration["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=TEST_CORRECTION_GAIN,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                weights=TEST_CORRECTION_WEIGHTS,
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
                    return_value=(100.0, 200.0, 300.0),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        requested = np.asarray(move.call_args.args[1], dtype=float)
        np.testing.assert_allclose(
            requested,
            np.array([10.0, 20.0, 30.0]) + expected_delta,
        )
        expected_model_residual = p1 - p0 - result["move_predicted_delta_px"].values[0]
        np.testing.assert_allclose(
            result["move_model_residual_delta_px"].values[0],
            expected_model_residual,
        )
        self.assertIn("correction_lqr_motor_penalty", result.attrs)

    def test_lqr_convergence_uses_projected_normalized_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            jacobian = calibration["px_per_cmd_mm"].values.reshape(
                len(CAMERAS) * len(PIXEL_AXES),
                len(COMMAND_AXES),
            )
            axis_scale = calibration["axis_scale_cmd_mm"].values
            design = calibration_core.compute_lqr_correction_design(
                jacobian,
                axis_scale,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                weights=TEST_CORRECTION_WEIGHTS,
            )
            controllable_basis = np.asarray(design["controllable_basis"])
            _, _, vh = np.linalg.svd(controllable_basis.T, full_matrices=True)
            normalized_null_error = vh[-1]
            shift = (
                10.0
                * np.asarray(design["image_scale"])
                * normalized_null_error
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            self.assertGreater(
                weighted_pixel_residual(
                    shift_dataset(shift),
                    weights=TEST_CORRECTION_WEIGHTS,
                ),
                0.0,
            )
            self.assertLess(
                calibration_core.lqr_projected_residual_from_design(
                    design,
                    shift_dataset(shift),
                ),
                constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE,
            )
            hardware_patches = self.patch_hardware([shift_dataset(shift)])
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    Mock(),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        move.assert_not_called()
        self.assertTrue(result.attrs["correction_converged"])
        self.assertEqual(
            result.attrs["correction_criterion"],
            "lqr_projected_normalized_error",
        )
        self.assertIn("iteration_correction_criterion_residual", result)

    def test_lqr_kalman_correction_uses_filtered_state_for_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            expected_delta = solve_lqr_command_correction(
                calibration["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=TEST_CORRECTION_GAIN,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                weights=TEST_CORRECTION_WEIGHTS,
            )
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER",
                    True,
                ),
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE",
                    None,
                ),
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(100.0, 200.0, 300.0),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        requested = np.asarray(move.call_args.args[1], dtype=float)
        np.testing.assert_allclose(
            requested,
            np.array([10.0, 20.0, 30.0]) + expected_delta,
        )
        self.assertTrue(result.attrs["correction_lqr_kalman_filter_enabled"])
        self.assertIn("iteration_lqr_kalman_state", result)
        self.assertEqual(result.sizes["lqr_state"], len(COMMAND_AXES))
        np.testing.assert_array_equal(
            result["iteration_lqr_kalman_measurement_accepted"].values,
            np.array([True, True]),
        )

    def test_lqr_kalman_gate_rejects_inconsistent_post_move_measurement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.full((len(CAMERAS), len(PIXEL_AXES)), 1_000.0, dtype=float)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER",
                    True,
                ),
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE",
                    1e-9,
                ),
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(100.0, 200.0, 300.0),
                ),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        np.testing.assert_array_equal(
            result["iteration_lqr_kalman_measurement_accepted"].values,
            np.array([True, False]),
        )
        np.testing.assert_allclose(
            result["iteration_lqr_kalman_state"].values[-1],
            result["iteration_lqr_kalman_predicted_state"].values[-1],
        )
        self.assertIn(
            "LQR Kalman measurement rejected",
            result.attrs["warnings"],
        )

    def test_lqr_kalman_gate_stops_before_prediction_only_moves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.full((len(CAMERAS), len(PIXEL_AXES)), 1_000.0, dtype=float)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER",
                    True,
                ),
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE",
                    1e-9,
                ),
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(100.0, 200.0, 300.0),
                ) as move,
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=3,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertEqual(move.call_count, 1)
        self.assertEqual(result.sizes["move"], 1)
        self.assertFalse(result.attrs["correction_converged"])
        self.assertIn("LQR Kalman measurement rejected", result.attrs["warnings"])
        self.assertIn("stopped before convergence", result.attrs["warnings"])

    def test_lqr_uses_lqr_specific_default_gain_and_step_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            p0 = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            lqr_gain = 0.25
            lqr_max_step = 0.125
            expected_delta = solve_lqr_command_correction(
                calibration["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=lqr_gain,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=lqr_max_step,
                weights=TEST_CORRECTION_WEIGHTS,
            )
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)]
            )
            with (
                patch.object(constants, "DEFAULT_LQR_CORRECTION_GAIN", lqr_gain),
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP",
                    lqr_max_step,
                ),
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(100.0, 200.0, 300.0),
                ) as move,
            ):
                do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        requested = np.asarray(move.call_args.args[1], dtype=float)
        np.testing.assert_allclose(
            requested,
            np.array([10.0, 20.0, 30.0]) + expected_delta,
        )

    def test_lqr_keeps_gain_fixed_after_valid_nondecreasing_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            p0 = x_shift(30.0)
            correction_delta = solve_lqr_command_correction(
                calibration["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=TEST_CORRECTION_GAIN,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                weights=TEST_CORRECTION_WEIGHTS,
            )
            predicted_delta = (
                calibration["px_per_cmd_mm"].values.reshape(
                    len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)
                )
                @ correction_delta
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            predicted_vector = predicted_delta.reshape(-1)
            orthogonal_disturbance = np.array(
                [
                    predicted_vector[1],
                    -predicted_vector[0],
                    predicted_vector[3],
                    -predicted_vector[2],
                ],
                dtype=float,
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            orthogonal_disturbance *= (
                2.0 * np.linalg.norm(p0) / np.linalg.norm(orthogonal_disturbance)
            )
            p1 = p0 + predicted_delta + orthogonal_disturbance
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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertTrue(bool(result["move_feedback_valid"].values[0]))
        self.assertGreater(
            float(result["move_post_weighted_residual_px"].values[0]),
            float(result["move_pre_weighted_residual_px"].values[0]),
        )
        self.assertAlmostEqual(
            float(result.attrs["correction_final_gain"]),
            TEST_CORRECTION_GAIN,
        )

    def test_lqr_gain_is_stable_and_reduces_predicted_residual(self):
        calibration = calibration_dataset()
        jacobian = calibration["px_per_cmd_mm"].values.reshape(
            len(CAMERAS) * len(PIXEL_AXES),
            len(COMMAND_AXES),
        )
        axis_scale = calibration["axis_scale_cmd_mm"].values
        feedback_gain, eigenvalues, rank, singular_values = (
            calibration_core._compute_lqr_feedback_gain(
                jacobian,
                axis_scale,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                weights=TEST_CORRECTION_WEIGHTS,
            )
        )

        self.assertEqual(
            feedback_gain.shape,
            (len(COMMAND_AXES), len(CAMERAS) * len(PIXEL_AXES)),
        )
        self.assertEqual(rank, len(COMMAND_AXES))
        self.assertEqual(singular_values.size, len(COMMAND_AXES))
        self.assertTrue(np.all(np.abs(eigenvalues) < 1.0))

        shift = np.array([[60.0, -40.0], [20.0, 10.0]], dtype=float)
        correction = solve_lqr_command_correction(
            calibration["px_per_cmd_mm"].values,
            shift_dataset(shift),
            axis_scale,
            gain=TEST_CORRECTION_GAIN,
            image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
            motor_penalty=TEST_LQR_MOTOR_PENALTY,
            svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
            weights=TEST_CORRECTION_WEIGHTS,
        )
        before = shift.reshape(-1)
        after = before + jacobian @ correction
        self.assertLess(float(np.linalg.norm(after)), float(np.linalg.norm(before)))

    def test_lqr_kalman_helpers_update_and_gate_measurements(self):
        calibration = calibration_dataset()
        jacobian = calibration["px_per_cmd_mm"].values.reshape(
            len(CAMERAS) * len(PIXEL_AXES),
            len(COMMAND_AXES),
        )
        axis_scale = calibration["axis_scale_cmd_mm"].values
        design = calibration_core.compute_lqr_correction_design(
            jacobian,
            axis_scale,
            image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
            motor_penalty=TEST_LQR_MOTOR_PENALTY,
            svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
            weights=TEST_CORRECTION_WEIGHTS,
        )
        initial = calibration_core.initialize_lqr_kalman_state(
            shift_dataset(np.array([[4.0, -3.0], [2.0, 1.0]], dtype=float)),
            design,
            initial_covariance=10.0,
        )
        command = calibration_core.solve_lqr_state_command_correction(
            design,
            initial["state"],
            gain=0.5,
            max_normalized_step=None,
        )
        predicted = calibration_core.predict_lqr_kalman_state(
            initial["state"],
            initial["covariance"],
            command,
            design,
            process_noise=0.1,
        )
        accepted = calibration_core.update_lqr_kalman_state(
            predicted["state"],
            predicted["covariance"],
            shift_dataset(np.array([[1.0, -1.0], [0.5, 0.25]], dtype=float)),
            design,
            measurement_noise=1.0,
            innovation_gate=1e12,
        )

        self.assertTrue(bool(accepted["measurement_accepted"]))
        self.assertEqual(accepted["state"].shape, (len(COMMAND_AXES),))
        self.assertEqual(
            accepted["covariance"].shape,
            (len(COMMAND_AXES), len(COMMAND_AXES)),
        )
        self.assertTrue(np.isfinite(accepted["state"]).all())

        measured = shift_dataset(np.array([[1.0, -1.0], [0.5, 0.25]], dtype=float))
        image_scale = np.asarray(design["image_scale"], dtype=float)
        raw_covariance = np.diag(image_scale * image_scale)
        raw_covariance_update = calibration_core.update_lqr_kalman_state(
            predicted["state"],
            predicted["covariance"],
            measured,
            design,
            measurement_noise=999.0,
            measurement_covariance=raw_covariance,
            innovation_gate=None,
        )
        controllable_basis = np.asarray(design["controllable_basis"], dtype=float)
        predicted_state = np.asarray(predicted["state"], dtype=float)
        predicted_covariance = np.asarray(predicted["covariance"], dtype=float)
        observation = np.asarray(measured["shift_px"].values, dtype=float).reshape(-1)
        innovation = observation / image_scale - controllable_basis @ predicted_state
        innovation_covariance = (
            controllable_basis @ predicted_covariance @ controllable_basis.T
            + np.eye(len(CAMERAS) * len(PIXEL_AXES), dtype=float)
        )
        kalman_gain = (
            predicted_covariance
            @ controllable_basis.T
            @ np.linalg.solve(
                innovation_covariance,
                np.eye(len(CAMERAS) * len(PIXEL_AXES), dtype=float),
            )
        )
        expected_state = predicted_state + kalman_gain @ innovation
        np.testing.assert_allclose(raw_covariance_update["state"], expected_state)

        rejected = calibration_core.update_lqr_kalman_state(
            predicted["state"],
            predicted["covariance"],
            shift_dataset(np.full((len(CAMERAS), len(PIXEL_AXES)), 1_000.0)),
            design,
            measurement_noise=1.0,
            innovation_gate=1e-9,
        )

        self.assertFalse(bool(rejected["measurement_accepted"]))
        np.testing.assert_allclose(rejected["state"], predicted["state"])
        np.testing.assert_allclose(rejected["covariance"], predicted["covariance"])

    def test_lqr_normalized_step_limit_caps_largest_axis_component(self):
        jacobian = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )
        correction = solve_lqr_command_correction(
            jacobian,
            shift_dataset(np.array([[10.0, 0.0], [0.0, 0.0]])),
            axis_scale_cmd_mm=[1.0, 1.0, 1.0],
            gain=1.0,
            image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
            motor_penalty=TEST_LQR_MOTOR_PENALTY,
            svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
            max_normalized_step=0.5,
            weights=TEST_CORRECTION_WEIGHTS,
        )

        np.testing.assert_allclose(correction, [-0.5, 0.0, 0.0])

    def test_zeroed_axes_are_not_sent_to_motor_move(self):
        jacobian = np.array(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]],
                [[0.0, 0.0, 100.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = calibration_dataset(jacobian)
            save_calibration_dataset(calibration, path)
            offset_mm = np.array([0.020, 0.0, 0.0], dtype=float)
            p0 = (
                calibration["px_per_cmd_mm"].values.reshape(
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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertEqual(move.call_args.args[0], ("x",))
        self.assertEqual(
            result["move_active_axis_mask"].values.tolist(),
            [[True, False, False]],
        )

    def test_tiny_correction_components_are_zeroed_before_motor_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            hardware_patches = self.patch_hardware(
                [shift_dataset(x_shift(10.0)), shift_dataset(np.zeros((2, 2)))],
                positions=(10.0, 20.0, 30.0),
            )
            deadband_um = constants.DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM
            active_delta_mm = (deadband_um + 0.1) / 1000.0
            raw_delta_mm = np.array(
                [
                    active_delta_mm,
                    (deadband_um / 2.0) / 1000.0,
                    -(deadband_um / 3.0) / 1000.0,
                ],
                dtype=float,
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.solve_lqr_command_correction",
                    return_value=raw_delta_mm,
                ),
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(10.0 + active_delta_mm,),
                ) as move,
                patch.object(
                    constants,
                    "DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER",
                    False,
                ),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertEqual(move.call_args.args[0], ("x",))
        np.testing.assert_allclose(
            np.asarray(move.call_args.args[1], dtype=float),
            np.array([10.0 + active_delta_mm], dtype=float),
        )
        np.testing.assert_allclose(
            result["move_command_delta_mm"].values,
            [[active_delta_mm, 0.0, 0.0]],
        )
        self.assertEqual(
            result["move_active_axis_mask"].values.tolist(),
            [[True, False, False]],
        )
        self.assertEqual(result.attrs["correction_move_delta_deadband_um"], deadband_um)

    def test_bcs_api_correction_move_uses_motor_backlash_constant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            offset_mm = np.array([0.020, 0.0, 0.0], dtype=float)
            p0 = (
                calibration["px_per_cmd_mm"].values.reshape(
                    len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)
                )
                @ offset_mm
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            expected_delta = solve_lqr_command_correction(
                calibration["px_per_cmd_mm"].values,
                shift_dataset(p0),
                calibration["axis_scale_cmd_mm"].values,
                gain=TEST_CORRECTION_GAIN,
                image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                motor_penalty=TEST_LQR_MOTOR_PENALTY,
                svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                weights=TEST_CORRECTION_WEIGHTS,
            )
            sanitized_delta = correct_module._validate_command_correction(expected_delta)
            active_indices = tuple(
                index for index, value in enumerate(sanitized_delta) if value != 0.0
            )
            active_axes = tuple(COMMAND_AXES[index] for index in active_indices)
            active_goals = tuple(
                np.array([0.5, 0.0, 0.0], dtype=float)[index]
                + sanitized_delta[index]
                for index in active_indices
            )
            self.assertLess(abs(expected_delta[0]), 0.1)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)],
                positions=(0.5, 0.0, 0.0),
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=active_goals,
                ) as move,
                patch.object(constants, "CORRECTION_USE_BCS_API_BACKLASH", True),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertEqual(move.call_args.args[0], active_axes)
        np.testing.assert_allclose(
            np.asarray(move.call_args.args[1], dtype=float),
            np.asarray(active_goals, dtype=float),
        )
        self.assertEqual(
            move.call_args.kwargs["backlash_correction"],
            constants.MOTOR_BACKLASH_CORRECTION,
        )
        self.assertTrue(result.attrs["correction_backlash_enabled"])

    def test_bcs_api_correction_backlash_can_be_disabled_by_constant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.save_calibration(tmpdir)
            calibration = calibration_dataset()
            offset_mm = np.array([0.020, 0.0, 0.0], dtype=float)
            p0 = (
                calibration["px_per_cmd_mm"].values.reshape(
                    len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)
                )
                @ offset_mm
            ).reshape(len(CAMERAS), len(PIXEL_AXES))
            p1 = np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=float)
            hardware_patches = self.patch_hardware(
                [shift_dataset(p0), shift_dataset(p1)],
                positions=(0.5, 0.0, 0.0),
            )
            with (
                hardware_patches[0],
                hardware_patches[1],
                hardware_patches[2],
                patch(
                    "merlin_track_position.tracking.correct.move_motors_and_wait",
                    return_value=(0.5, 0.0, 0.0),
                ) as move,
                patch.object(constants, "CORRECTION_USE_BCS_API_BACKLASH", False),
            ):
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                    gain=TEST_CORRECTION_GAIN,
                    max_normalized_step=TEST_CORRECTION_MAX_NORMALIZED_STEP,
                    lqr_image_scale_px=TEST_LQR_IMAGE_SCALE_PX,
                    lqr_motor_penalty=TEST_LQR_MOTOR_PENALTY,
                    lqr_svd_relative_tolerance=TEST_LQR_SVD_RELATIVE_TOLERANCE,
                    weights=TEST_CORRECTION_WEIGHTS,
                )

        self.assertEqual(move.call_args.kwargs["backlash_correction"], {})
        self.assertFalse(result.attrs["correction_backlash_enabled"])

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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                )

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
        self.assertIn("correction_move_started_at", result.attrs)
        self.assertIn("correction_move_finished_at", result.attrs)
        self.assertEqual(
            saved.attrs["correction_move_started_at"],
            result.attrs["correction_move_started_at"],
        )
        self.assertEqual(
            saved.attrs["correction_move_finished_at"],
            result.attrs["correction_move_finished_at"],
        )
        move_started = datetime.fromisoformat(
            str(saved.attrs["correction_move_started_at"])
        )
        move_finished = datetime.fromisoformat(
            str(saved.attrs["correction_move_finished_at"])
        )
        self.assertLessEqual(move_started, move_finished)
        self.assertIn("move_measured_delta_px", saved)
        self.assertIn("move_predicted_delta_px", saved)
        self.assertIn("move_model_residual_delta_px", saved)
        self.assertIn("move_predicted_weighted_response_px", saved)
        self.assertIn("move_measured_weighted_response_px", saved)
        self.assertIn("move_feedback_alpha", saved)
        self.assertIn("move_feedback_parallel_px", saved)
        self.assertIn("move_feedback_valid", saved)
        self.assertTrue(bool(saved["move_feedback_valid"].values[0]))

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

    def test_correction_history_updates_existing_run_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_path = Path(tmpdir) / "calibration.h5"
            history_path = correction_history_path(calibration_path)

            def result(move_count: int, image_value: int) -> xr.Dataset:
                iteration_count = move_count + 1
                move_values = (
                    np.arange(move_count * len(COMMAND_AXES), dtype=float).reshape(
                        move_count,
                        len(COMMAND_AXES),
                    )
                    / 1000.0
                )
                iteration_values = np.stack(
                    [x_shift(float(index)) for index in range(iteration_count)],
                    axis=0,
                )
                current_cam0 = np.full((4, 5), image_value, dtype=np.uint16)
                current_cam1 = np.full((6, 7), image_value + 1, dtype=np.uint16)
                updated = shift_dataset(x_shift(float(image_value))).assign(
                    current_cam0=(("y_cam0", "x_cam0"), current_cam0),
                    current_cam1=(("y_cam1", "x_cam1"), current_cam1),
                    move_command_delta_mm=(
                        ("move", "command_axis"),
                        move_values,
                        {"units": "commanded-mm"},
                    ),
                    move_feedback_valid=(
                        ("move",),
                        np.ones(move_count, dtype=bool),
                    ),
                    iteration_shift_px=(
                        ("iteration", "camera", "pixel_axis"),
                        iteration_values,
                        {"units": "px"},
                    ),
                    iteration_weighted_residual_px=(
                        ("iteration",),
                        np.linspace(2.0, 1.0, iteration_count),
                        {"units": "px"},
                    ),
                ).assign_coords(
                    move=np.arange(move_count, dtype=np.int64),
                    iteration=np.arange(iteration_count, dtype=np.int64),
                    command_axis=list(COMMAND_AXES),
                )
                return updated.assign_attrs(
                    {
                        "calibration_path": str(calibration_path),
                        "correction_history_completed": move_count > 1,
                    }
                )

            correct_module.save_correction_history_dataset(
                result(1, 4),
                history_path,
                run_id=0,
            )
            correct_module.save_correction_history_dataset(
                result(2, 9),
                history_path,
                run_id=0,
            )

            with h5py.File(history_path, "r") as history_file:
                self.assertEqual(int(history_file.attrs["latest_run_id"]), 0)
                group = history_file["run_000000"]
                self.assertEqual(group["move"].maxshape, (None,))
                self.assertEqual(group["iteration"].maxshape, (None,))
                self.assertIsNone(group["move_command_delta_mm"].maxshape[0])
                self.assertIsNone(group["iteration_shift_px"].maxshape[0])
                self.assertEqual(group["move_command_delta_mm"].shape[0], 2)
                self.assertEqual(group["iteration_shift_px"].shape[0], 3)

            loaded = load_latest_correction_history_dataset(calibration_path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.attrs["correction_history_completed"])
        self.assertEqual(loaded.sizes["move"], 2)
        self.assertEqual(loaded.sizes["iteration"], 3)
        np.testing.assert_allclose(
            loaded["move_command_delta_mm"].values,
            np.array([[0.0, 0.001, 0.002], [0.003, 0.004, 0.005]]),
        )
        np.testing.assert_array_equal(
            loaded["current_cam0"].values,
            np.full((4, 5), 9, dtype=np.uint16),
        )
        self.assertEqual(loaded["move_feedback_valid"].dtype, bool)

    def test_successful_deferred_correction_history_save_does_not_stage(self):
        result = shift_dataset(x_shift(1.0)).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "correction-history.h5"
            with patch(
                "merlin_track_position.tracking.correct.stage_dataset",
            ) as stage:
                persistence = correct_module.save_correction_history_dataset_deferred(
                    result,
                    history_path,
                    run_id=0,
                )

        stage.assert_not_called()
        self.assertTrue(persistence.flushed)
        self.assertFalse(persistence.pending)
        self.assertIsNone(persistence.spool_path)

    def test_next_correction_history_run_id_uses_latest_attr_with_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_path = Path(tmpdir) / "fast.h5"
            with h5py.File(fast_path, "w") as history_file:
                history_file.create_group("run_000003")
                history_file.create_group("run_999999")
                history_file.attrs["latest_run_id"] = 3

            legacy_path = Path(tmpdir) / "legacy.h5"
            with h5py.File(legacy_path, "w") as history_file:
                history_file.create_group("run_000002")
                history_file.create_group("run_000005")

            broken_attr_path = Path(tmpdir) / "broken-attr.h5"
            with h5py.File(broken_attr_path, "w") as history_file:
                history_file.create_group("run_000004")
                history_file.attrs["latest_run_id"] = 10

            self.assertEqual(
                correct_module._next_correction_history_run_id(fast_path),
                4,
            )
            self.assertEqual(
                correct_module._next_correction_history_run_id(legacy_path),
                6,
            )
            self.assertEqual(
                correct_module._next_correction_history_run_id(broken_attr_path),
                5,
            )

    def test_correction_timestamps_from_history_extracts_run_attrs(self):
        first = shift_dataset(x_shift(1.0)).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
                "correction_history_run_id": 2,
                "correction_started_at_utc": "2026-05-19T10:00:00+00:00",
                "correction_move_started_at": "2026-05-19T03:00:02-07:00",
                "correction_move_finished_at": "2026-05-19T03:00:05-07:00",
            }
        )
        second = shift_dataset(x_shift(2.0)).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
                "correction_history_run_id": 0,
                "correction_started_at_utc": "2026-05-19T10:10:00Z",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "correction-history.h5"
            correct_module.save_correction_history_dataset(
                first,
                history_path,
                run_id=2,
            )
            correct_module.save_correction_history_dataset(
                second,
                history_path,
                run_id=0,
            )
            timestamps = correction_timestamps_from_history(history_path)

        np.testing.assert_array_equal(timestamps["run_id"].values, np.array([0, 2]))
        self.assertEqual(
            timestamps["run_group"].values.tolist(),
            ["run_000000", "run_000002"],
        )
        np.testing.assert_array_equal(
            timestamps["correction_started_at_utc"].values,
            np.array(
                [
                    "2026-05-19T10:10:00",
                    "2026-05-19T10:00:00",
                ],
                dtype="datetime64[ns]",
            ),
        )
        self.assertTrue(
            np.isnat(timestamps["correction_move_started_at_utc"].values[0])
        )
        self.assertEqual(
            timestamps["correction_move_started_at_utc"].values[1],
            np.datetime64("2026-05-19T10:00:02", "ns"),
        )
        self.assertEqual(
            timestamps["correction_move_finished_at_utc"].values[1],
            np.datetime64("2026-05-19T10:00:05", "ns"),
        )

    def test_correction_total_move_by_axis_from_history(self):
        jacobian = px_per_cmd_mm()
        detected_move_delta_mm = np.array(
            [
                [0.0005, -0.001, 0.002],
                [0.0015, 0.0, -0.001],
            ],
            dtype=float,
        )
        detected_total_delta_px = measured_from_jacobian(
            detected_move_delta_mm.sum(axis=0, keepdims=True),
            jacobian,
        )[0]
        first = shift_dataset(x_shift(1.0)).assign(
            px_per_cmd_mm=(
                ("camera", "pixel_axis", "command_axis"),
                jacobian,
                {"units": "px/commanded-mm"},
            ),
            move_command_delta_mm=(
                ("move", "command_axis"),
                np.array(
                    [
                        [0.001, -0.002, 0.0],
                        [0.0, 0.003, -0.004],
                    ],
                    dtype=float,
                ),
                {"units": "commanded-mm"},
            ),
            iteration_shift_px=(
                ("iteration", "camera", "pixel_axis"),
                np.stack(
                    (
                        np.zeros_like(detected_total_delta_px),
                        detected_total_delta_px,
                    )
                ),
                {"units": "px"},
            ),
            iteration_weighted_residual_px=(
                ("iteration",),
                np.array([2.5, 0.75], dtype=float),
                {"units": "px"},
            ),
        ).assign_coords(
            move=np.arange(2, dtype=np.int64),
            iteration=np.arange(2, dtype=np.int64),
            command_axis=list(COMMAND_AXES),
        ).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
                "correction_history_run_id": 2,
                "correction_move_finished_at": "2026-05-19T03:00:05-07:00",
            }
        )
        second = shift_dataset(x_shift(2.0)).assign(
            px_per_cmd_mm=(
                ("camera", "pixel_axis", "command_axis"),
                jacobian,
                {"units": "px/commanded-mm"},
            ),
            move_command_delta_mm=(
                ("move", "command_axis"),
                np.empty((0, len(COMMAND_AXES)), dtype=float),
                {"units": "commanded-mm"},
            ),
            move_measured_delta_px=(
                ("move", "camera", "pixel_axis"),
                np.empty((0, len(CAMERAS), len(PIXEL_AXES)), dtype=float),
                {"units": "px"},
            ),
            iteration_weighted_residual_px=(
                ("iteration",),
                np.array([1.25], dtype=float),
                {"units": "px"},
            ),
        ).assign_coords(
            move=np.arange(0, dtype=np.int64),
            iteration=np.arange(1, dtype=np.int64),
            command_axis=list(COMMAND_AXES),
        ).assign_attrs(
            {
                "calibration_path": "calibration.h5",
                "correction_history_completed": True,
                "correction_history_run_id": 0,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "correction-history.h5"
            correct_module.save_correction_history_dataset(
                first,
                history_path,
                run_id=2,
            )
            correct_module.save_correction_history_dataset(
                second,
                history_path,
                run_id=0,
            )
            moves = correction_total_move_by_axis_from_history(history_path)

        self.assertEqual(moves.name, "total_move_mm")
        self.assertEqual(
            moves.dims,
            ("after_move_timestamp_utc", "move_source", "command_axis"),
        )
        np.testing.assert_array_equal(moves["run_id"].values, np.array([0, 2]))
        self.assertEqual(
            moves["move_source"].values.tolist(),
            ["commanded", "detected"],
        )
        self.assertEqual(moves["command_axis"].values.tolist(), list(COMMAND_AXES))
        np.testing.assert_allclose(
            moves["weighted_residual_px"].values,
            np.array([1.25, 0.75]),
        )
        np.testing.assert_allclose(
            moves.values,
            np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.001, 0.001, -0.004],
                        [0.002, -0.001, 0.001],
                    ],
                ]
            ),
            atol=1e-12,
        )
        self.assertTrue(np.isnat(moves["after_move_timestamp_utc"].values[0]))
        self.assertEqual(
            moves["after_move_timestamp_utc"].values[1],
            np.datetime64("2026-05-19T10:00:05", "ns"),
        )

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
                    result = do_correction(
                        path,
                        capture_count=1,
                        correction_mode="camera",
                        max_moves=1,
                    )
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
                    self.assertEqual(pending_entry_count(), 1)
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
                    correction_mode="camera",
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
                do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                )

            loaded = load_latest_correction_history_dataset(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.attrs["correction_history_completed"])
        self.assertTrue(loaded.attrs["correction_applied"])
        self.assertEqual(loaded.sizes["move"], 1)
        self.assertEqual(loaded["move_feedback_valid"].dtype, bool)

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
                result = do_correction(
                    path,
                    capture_count=1,
                    correction_mode="camera",
                    max_moves=1,
                )

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
