import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CallableCameraPlugin,
    CameraPairPlugin,
)
from merlin_track_position.interface.calibration_panel import _validate_calibration_dataset
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    PIXEL_AXES,
    STAGE_AXES,
    _resolve_n_jobs,
    estimate_stage_offset,
    fit_calibration_from_images,
    get_correction,
)
from merlin_track_position.tracking.calibrate import (
    _make_calibration_path,
    calibration_sample_count,
    run_calibration,
)
from merlin_track_position.tracking.correct import do_correction
from merlin_track_position.tracking.sample_calibration import (
    build_sample_calibration_dataset,
)

ORIGIN_STABILITY_UM = 5.0


def textured_image(seed=10, shape=(160, 176)):
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="wrap")
    y, x = np.indices(shape)
    image += 0.3 * np.sin(x / 9.0) + 0.2 * np.cos(y / 13.0)
    image -= image.min()
    image /= image.max()
    return image


def stereo_stage_to_pixel():
    return np.array(
        [
            [[0.27, -0.14, 0.07], [0.09, 0.31, -0.12]],
            [[-0.21, 0.18, 0.33], [0.24, 0.05, 0.16]],
        ],
        dtype=float,
    )


def calibration_stage():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [-30.0, 0.0, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, -30.0, 0.0],
            [0.0, 0.0, 30.0],
            [0.0, 0.0, -30.0],
            [30.0, 30.0, 30.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def make_stereo_images(stage, stage_to_pixel, *, seed0=20, seed1=21):
    reference_cam0 = textured_image(seed=seed0, shape=(160, 176))
    reference_cam1 = textured_image(seed=seed1, shape=(144, 192))
    images_cam0 = []
    images_cam1 = []
    for stage_row in stage:
        shifts = np.einsum("cpk,k->cp", stage_to_pixel, stage_row)
        du0, dv0 = shifts[0]
        du1, dv1 = shifts[1]
        images_cam0.append(
            ndimage.shift(reference_cam0, shift=(dv0, du0), order=3, mode="wrap")
        )
        images_cam1.append(
            ndimage.shift(reference_cam1, shift=(dv1, du1), order=3, mode="wrap")
        )
    return images_cam0, images_cam1


def as_capture_stacks(images):
    return [np.asarray(image)[None, ...] for image in images]


def camera_pair_from_pair_source(pair_source):
    pending_pair = []

    def capture_cam0():
        if pending_pair:
            raise AssertionError("cam0 was captured before cam1 consumed the pair")
        pending_pair.append(pair_source())
        return pending_pair[0][0]

    def capture_cam1():
        if not pending_pair:
            pending_pair.append(pair_source())
        image_cam0, image_cam1 = pending_pair.pop(0)
        del image_cam0
        return image_cam1

    return CameraPairPlugin(
        CallableCameraPlugin("cam0", capture_cam0),
        CallableCameraPlugin("cam1", capture_cam1),
    )


class CalibrationTests(unittest.TestCase):
    def test_calibration_path_prioritizes_positive_y_motion(self):
        path = _make_calibration_path(3, 10.0)

        np.testing.assert_allclose(
            path,
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
        )

        moves = np.diff(np.vstack(([0.0, 0.0, 0.0], path)), axis=0)
        self.assertEqual(np.count_nonzero(moves[:, 1] < 0.0), 1)

    def test_calibration_path_preserves_sample_positions(self):
        path = _make_calibration_path(5, 10.0)
        rows = {tuple(row) for row in path}
        offsets = (-20.0, -10.0, 10.0, 20.0)
        expected_rows = {
            (0.0, 0.0, z_offset)
            for z_offset in offsets
        } | {
            (0.0, y_offset, 0.0)
            for y_offset in offsets
        } | {
            (x_offset, 0.0, 0.0)
            for x_offset in offsets
        } | {
            (x_offset, y_offset, z_offset)
            for x_offset in (-10.0, 10.0)
            for y_offset in (-10.0, 10.0)
            for z_offset in (-10.0, 10.0)
        } | {
            (0.0, 0.0, 0.0)
        }

        self.assertEqual(rows, expected_rows)
        self.assertEqual(path.shape[0], len(expected_rows))
        moves = np.diff(np.vstack(([0.0, 0.0, 0.0], path)), axis=0)
        self.assertEqual(np.count_nonzero(moves[:, 1] < 0.0), 1)

    def test_calibration_sample_count_matches_path_shape(self):
        self.assertEqual(calibration_sample_count(3), 16)
        self.assertEqual(calibration_sample_count(5), 22)

    def test_calibration_sample_count_rejects_too_few_axis_points(self):
        with self.assertRaisesRegex(ValueError, "n must be >= 2"):
            calibration_sample_count(1)

    def test_run_calibration_records_initial_stage_and_tilt_attrs(self):
        captured_kwargs = {}
        generator_calls = 0

        def fake_pair_source():
            nonlocal generator_calls
            generator_calls += 1
            image = np.full((8, 8), generator_calls, dtype=float)
            return image, image + 100.0

        def fake_move_motors_and_wait(motor_aliases, goals, **kwargs):
            del motor_aliases, kwargs
            return tuple(float(goal) for goal in goals)

        def fake_fit_calibration_from_images(**kwargs):
            captured_kwargs.update(kwargs)
            return xr.Dataset(attrs=kwargs["additional_context"])

        with (
            patch(
                "merlin_track_position.tracking.calibrate.get_positions",
                return_value=(1.2, 2.3, 3.4, 4.5, 6.7, 5.0),
            ),
            patch(
                "merlin_track_position.tracking.calibrate.move_motors_and_wait",
                side_effect=fake_move_motors_and_wait,
            ),
            patch(
                "merlin_track_position.tracking.calibrate.fit_calibration_from_images",
                side_effect=fake_fit_calibration_from_images,
            ),
            patch("merlin_track_position.tracking.calibrate.time.sleep"),
        ):
            calibration = run_calibration(
                2,
                10.0,
                camera_pair_from_pair_source(fake_pair_source),
                origin_stability_um=ORIGIN_STABILITY_UM,
                capture_count=1,
            )

        self.assertEqual(calibration.attrs["initial_x_mm"], 1.2)
        self.assertEqual(calibration.attrs["initial_y_mm"], 2.3)
        self.assertEqual(calibration.attrs["initial_z_mm"], 3.4)
        self.assertEqual(calibration.attrs["polar"], 4.5)
        self.assertEqual(calibration.attrs["tilt"], 6.7)
        np.testing.assert_allclose(captured_kwargs["stage_um"][0], [0.0, 0.0, 0.0])

    def test_run_calibration_captures_default_images_per_step_and_reports_once(self):
        captured_kwargs = {}
        callback_steps = []
        generator_calls = 0

        def fake_pair_source():
            nonlocal generator_calls
            generator_calls += 1
            value = float(generator_calls)
            return (
                np.full((2, 2), value, dtype=np.float32),
                np.full((2, 3), value + 100.0, dtype=np.float32),
            )

        def fake_move_motors_and_wait(motor_aliases, goals, **kwargs):
            del motor_aliases, kwargs
            return tuple(float(goal) for goal in goals)

        def fake_fit_calibration_from_images(**kwargs):
            captured_kwargs.update(kwargs)
            return xr.Dataset()

        def step_callback(idx, dx, dy, dz, image_cam0, image_cam1):
            callback_steps.append((idx, dx, dy, dz, image_cam0, image_cam1))

        with (
            patch(
                "merlin_track_position.tracking.calibrate.get_positions",
                return_value=(1.2, 2.3, 3.4, 4.5, 6.7, 5.0),
            ),
            patch(
                "merlin_track_position.tracking.calibrate.move_motors_and_wait",
                side_effect=fake_move_motors_and_wait,
            ),
            patch(
                "merlin_track_position.tracking.calibrate.fit_calibration_from_images",
                side_effect=fake_fit_calibration_from_images,
            ),
            patch("merlin_track_position.tracking.calibrate.time.sleep"),
        ):
            run_calibration(
                2,
                10.0,
                camera_pair_from_pair_source(fake_pair_source),
                origin_stability_um=ORIGIN_STABILITY_UM,
                step_callback=step_callback,
            )

        expected_steps = calibration_sample_count(2)
        expected_capture_count = constants.DEFAULT_CAPTURE_COUNT
        self.assertEqual(generator_calls, expected_steps * expected_capture_count)
        self.assertEqual(len(callback_steps), expected_steps)
        self.assertEqual(len(captured_kwargs["images_cam0"]), expected_steps)
        self.assertEqual(
            captured_kwargs["images_cam0"][0].shape,
            (expected_capture_count, 2, 2),
        )
        self.assertEqual(captured_kwargs["images_cam0"][0].dtype, np.float32)
        expected_representative_value = (expected_capture_count + 1.0) / 2.0
        np.testing.assert_array_equal(
            callback_steps[0][4],
            np.full((2, 2), expected_representative_value, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            callback_steps[0][5],
            np.full((2, 3), expected_representative_value + 100.0, dtype=np.float32),
        )

    def test_run_calibration_reports_full_display_images_from_cropped_pair(self):
        captured_kwargs = {}
        callback_steps = []
        generator_calls = 0

        def fake_pair_source():
            nonlocal generator_calls
            generator_calls += 1
            value = 1.0 if generator_calls % 2 else 10.0
            return (
                np.full((4, 5), value, dtype=np.float32),
                np.full((6, 7), value + 100.0, dtype=np.float32),
            )

        def fake_move_motors_and_wait(motor_aliases, goals, **kwargs):
            del motor_aliases, kwargs
            return tuple(float(goal) for goal in goals)

        def fake_fit_calibration_from_images(**kwargs):
            captured_kwargs.update(kwargs)
            return xr.Dataset()

        def step_callback(idx, dx, dy, dz, image_cam0, image_cam1):
            callback_steps.append((idx, dx, dy, dz, image_cam0, image_cam1))

        def processing_callback(completed, total):
            del completed, total

        camera_pair = camera_pair_from_pair_source(fake_pair_source).cropped(
            (1.0, 1.0, 3.0, 2.0),
            (2.0, 3.0, 2.0, 2.0),
        )
        with (
            patch(
                "merlin_track_position.tracking.calibrate.get_positions",
                return_value=(1.2, 2.3, 3.4, 4.5, 6.7, 5.0),
            ),
            patch(
                "merlin_track_position.tracking.calibrate.move_motors_and_wait",
                side_effect=fake_move_motors_and_wait,
            ),
            patch(
                "merlin_track_position.tracking.calibrate.fit_calibration_from_images",
                side_effect=fake_fit_calibration_from_images,
            ),
            patch("merlin_track_position.tracking.calibrate.time.sleep"),
        ):
            run_calibration(
                2,
                10.0,
                camera_pair,
                origin_stability_um=ORIGIN_STABILITY_UM,
                capture_count=2,
                step_callback=step_callback,
                processing_callback=processing_callback,
            )

        self.assertEqual(captured_kwargs["images_cam0"][0].shape, (2, 2, 3))
        self.assertEqual(captured_kwargs["images_cam1"][0].shape, (2, 2, 2))
        self.assertIs(captured_kwargs["progress_callback"], processing_callback)
        np.testing.assert_array_equal(
            callback_steps[0][4],
            np.full((4, 5), 5.5, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            callback_steps[0][5],
            np.full((6, 7), 105.5, dtype=np.float32),
        )

    def test_fit_calibration_from_stereo_numpy_arrays(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)

        calibration = fit_calibration_from_images(
            as_capture_stacks(images_cam0),
            as_capture_stacks(images_cam1),
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
        )

        self.assertEqual(calibration.attrs["format_version"], "1")
        self.assertEqual(calibration["stage_to_pixel"].dims, ("camera", "pixel_axis", "stage_axis"))
        for derived_name in (
            "condition_number",
            "origin_stability_um",
            "pixel_to_stage",
            "predicted_shift_px",
            "residual_shift_px",
            "residual_stage_um",
            "return_to_origin_image_error_norm_um",
            "return_to_origin_motor_error_norm_um",
            "repeatability_mean_rms_std_px",
            "repeatability_max_rms_std_px",
        ):
            self.assertNotIn(derived_name, calibration.data_vars)
        self.assertEqual(calibration["stage_um"].shape[1], 3)
        np.testing.assert_allclose(
            calibration["stage_to_pixel"].values,
            stage_to_pixel,
            atol=0.04,
        )
        np.testing.assert_allclose(calibration["stage_um"].values, stage)
        self.assertNotIn("return_to_origin_motor_error_um", calibration.data_vars)

        shift = np.einsum("cpk,k->cp", stage_to_pixel, np.array([30.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            estimate_stage_offset(calibration, shift),
            [30.0, 0.0, 0.0],
            atol=1.0,
        )

    def test_fit_calibration_uses_median_shift_and_mean_images_from_capture_stack(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        capture_deltas = np.array([-0.2, -0.1, 0.0, 0.1, 200.0], dtype=float)
        capture_markers = (0, 1, 10, 11, 12)
        capture_marker_to_index = {
            marker: capture_index
            for capture_index, marker in enumerate(capture_markers)
        }
        images_cam0 = []
        images_cam1 = []
        for sample_index in range(stage.shape[0]):
            images_cam0.append(
                np.stack(
                    [
                        np.full(
                            (4, 4),
                            sample_index * 100 + marker,
                            dtype=np.float32,
                        )
                        for marker in capture_markers
                    ],
                    axis=0,
                )
            )
            images_cam1.append(
                np.stack(
                    [
                        np.full(
                            (5, 6),
                            10000 + sample_index * 100 + marker,
                            dtype=np.float32,
                        )
                        for marker in capture_markers
                    ],
                    axis=0,
                )
            )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, kwargs
            marker = int(round(float(np.asarray(current)[0, 0])))
            camera_index = 1 if marker >= 10000 else 0
            if camera_index == 1:
                marker -= 10000
            sample_index, capture_marker = divmod(marker, 100)
            capture_index = capture_marker_to_index[capture_marker]
            base_shift = stage_to_pixel[camera_index] @ stage[sample_index]
            return xr.Dataset(
                data_vars={
                    "shift_px": (
                        ("pixel_axis",),
                        base_shift + capture_deltas[capture_index],
                    ),
                },
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ) as estimate_shift:
            calibration = fit_calibration_from_images(
                images_cam0,
                images_cam1,
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
                n_jobs=1,
            )

        expected_shifts = np.einsum("cpk,sk->scp", stage_to_pixel, stage)
        self.assertEqual(estimate_shift.call_count, stage.shape[0] * len(CAMERAS) * 5)
        self.assertEqual(calibration.attrs["capture_count"], 5)
        np.testing.assert_allclose(calibration["measured_shift_px"].values, expected_shifts)
        np.testing.assert_allclose(calibration["capture_shift_mad_px"].values, 0.1)
        self.assertEqual(calibration["image_cam0"].dtype, np.float32)
        self.assertEqual(calibration["image_cam1"].dtype, np.float32)
        np.testing.assert_array_equal(
            calibration["image_cam0"].values[0],
            np.full((4, 4), 6.8, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            calibration["image_cam1"].values[0],
            np.full((5, 6), 10006.8, dtype=np.float32),
        )

    def test_fit_calibration_reports_measurement_warning_step_and_camera(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        image = np.zeros((16, 16), dtype=float)
        images = as_capture_stacks([image for _ in range(stage.shape[0])])
        call_count = 0

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, current, kwargs
            nonlocal call_count
            sample_index, camera_index = divmod(call_count, len(CAMERAS))
            call_count += 1
            warnings = ""
            if sample_index == 2 and camera_index == 1:
                warnings = "low texture\nregistration error is not finite"
            return xr.Dataset(
                data_vars={
                    "shift_px": (
                        ("pixel_axis",),
                        stage_to_pixel[camera_index] @ stage[sample_index],
                    ),
                },
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": warnings},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            calibration = fit_calibration_from_images(
                images,
                images,
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
                n_jobs=1,
            )

        warning_lines = calibration.attrs["warnings"].splitlines()
        self.assertEqual(call_count, stage.shape[0] * len(CAMERAS))
        self.assertIn(
            "step 3 (x=-30, y=0, z=0 um), cam1: low texture",
            warning_lines,
        )
        self.assertIn(
            "step 3 (x=-30, y=0, z=0 um), cam1: registration error is not finite",
            warning_lines,
        )
        self.assertNotIn("one or more shift measurements", calibration.attrs["warnings"])
        self.assertEqual(
            calibration["measurement_warnings"].values[2, 1],
            "low texture\nregistration error is not finite",
        )

    def test_fit_calibration_parallel_matches_sequential_and_reports_progress(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        capture_stacks_cam0 = as_capture_stacks(images_cam0)
        capture_stacks_cam1 = as_capture_stacks(images_cam1)

        sequential = fit_calibration_from_images(
            capture_stacks_cam0,
            capture_stacks_cam1,
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
            n_jobs=1,
        )
        progress = []
        parallel = fit_calibration_from_images(
            capture_stacks_cam0,
            capture_stacks_cam1,
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
            progress_callback=lambda completed, total: progress.append(
                (completed, total)
            ),
            n_jobs=2,
        )

        xr.testing.assert_allclose(parallel, sequential)
        self.assertEqual(parallel.attrs, sequential.attrs)
        expected_total = stage.shape[0] * len(CAMERAS)
        self.assertEqual(progress[0], (0, expected_total))
        self.assertEqual(progress[-1], (expected_total, expected_total))
        self.assertEqual([completed for completed, _ in progress], list(range(expected_total + 1)))
        self.assertTrue(all(total == expected_total for _, total in progress))

    def test_fit_calibration_default_worker_count_comes_from_constants(self):
        self.assertEqual(_resolve_n_jobs(None), constants.CALIBRATION_FIT_N_JOBS)
        with patch.object(constants, "CALIBRATION_FIT_N_JOBS", 3):
            self.assertEqual(_resolve_n_jobs(None), 3)

        self.assertEqual(_resolve_n_jobs(2), 2)
        with self.assertRaisesRegex(ValueError, "n_jobs"):
            _resolve_n_jobs(0)

    def test_h5_roundtrip_preserves_stereo_schema(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        dataset = fit_calibration_from_images(
            as_capture_stacks(images_cam0),
            as_capture_stacks(images_cam1),
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.h5"
            dataset.to_netcdf(path, engine="h5netcdf")

            with xr.open_dataset(path, engine="h5netcdf") as dataset_on_disk:
                loaded = dataset_on_disk.load()

        self.assertEqual(loaded["image_cam0"].dims, ("sample", "y_cam0", "x_cam0"))
        self.assertEqual(loaded["image_cam1"].dims, ("sample", "y_cam1", "x_cam1"))
        np.testing.assert_allclose(loaded["stage_um"].values, stage)
        np.testing.assert_allclose(
            loaded["stage_to_pixel"].values,
            dataset["stage_to_pixel"].values,
        )

    def test_get_correction_uses_two_camera_shift(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        calibration = fit_calibration_from_images(
            as_capture_stacks(images_cam0),
            as_capture_stacks(images_cam1),
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
        )
        offset_um = np.array([8.0, -4.0, 6.0])
        shifts = np.einsum("cpk,k->cp", stage_to_pixel, offset_um)
        current_cam0 = ndimage.shift(
            images_cam0[0],
            shift=(shifts[0, 1], shifts[0, 0]),
            order=3,
            mode="wrap",
        )
        current_cam1 = ndimage.shift(
            images_cam1[0],
            shift=(shifts[1, 1], shifts[1, 0]),
            order=3,
            mode="wrap",
        )

        result = get_correction(
            calibration,
            calibration["image_cam0"].isel(sample=0).values,
            current_cam0[None, ...],
            calibration["image_cam1"].isel(sample=0).values,
            current_cam1[None, ...],
            check_tiles=False,
            clip_percentiles=None,
        )

        np.testing.assert_allclose(
            result["estimated_stage_offset_um"].values,
            offset_um,
            atol=1.0,
        )
        np.testing.assert_allclose(
            result["correction_um"].values,
            -offset_um,
            atol=1.0,
        )
        self.assertEqual(result["shift_px"].shape, (len(CAMERAS), len(PIXEL_AXES)))
        np.testing.assert_allclose(result["current_cam0"].values, current_cam0)
        np.testing.assert_allclose(result["current_cam1"].values, current_cam1)
        self.assertNotIn("method", result.attrs)

    def test_fit_calibration_rejects_legacy_2d_inputs(self):
        stage = calibration_stage()
        image = np.zeros((16, 16), dtype=float)

        with self.assertRaisesRegex(ValueError, "must be 3D"):
            fit_calibration_from_images(
                [image for _ in range(stage.shape[0])],
                as_capture_stacks([image for _ in range(stage.shape[0])]),
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
            )

    def test_get_correction_rejects_legacy_2d_current_images(self):
        image = np.zeros((16, 16), dtype=float)
        calibration = xr.Dataset(
            data_vars={
                "stage_to_pixel": (
                    ("camera", "pixel_axis", "stage_axis"),
                    stereo_stage_to_pixel(),
                ),
            },
            coords={
                "camera": list(CAMERAS),
                "pixel_axis": list(PIXEL_AXES),
                "stage_axis": list(STAGE_AXES),
            },
        )

        with self.assertRaisesRegex(ValueError, "current_cam0\\[0\\] must be 3D"):
            get_correction(
                calibration,
                image,
                image,
                image,
                image[None, ...],
                check_tiles=False,
                clip_percentiles=None,
            )

    def test_get_correction_uses_median_shift_from_capture_stack(self):
        stage_to_pixel = stereo_stage_to_pixel()
        calibration = xr.Dataset(
            data_vars={
                "stage_to_pixel": (
                    ("camera", "pixel_axis", "stage_axis"),
                    stage_to_pixel,
                ),
            },
            coords={
                "camera": list(CAMERAS),
                "pixel_axis": list(PIXEL_AXES),
                "stage_axis": list(STAGE_AXES),
            },
        )
        offset_um = np.array([8.0, -4.0, 6.0])
        base_shifts = np.einsum("cpk,k->cp", stage_to_pixel, offset_um)
        capture_deltas = np.array([-0.2, -0.1, 0.0, 0.1, 200.0], dtype=float)
        reference_cam0 = np.zeros((4, 4), dtype=np.float32)
        reference_cam1 = np.zeros((5, 6), dtype=np.float32)
        current_cam0 = np.stack(
            [
                np.full(reference_cam0.shape, capture_index, dtype=np.float32)
                for capture_index in range(5)
            ],
            axis=0,
        )
        current_cam1 = np.stack(
            [
                np.full(reference_cam1.shape, 100 + capture_index, dtype=np.float32)
                for capture_index in range(5)
            ],
            axis=0,
        )

        def fake_estimate_shift(reference, current, **kwargs):
            del reference, kwargs
            marker = int(round(float(np.asarray(current)[0, 0])))
            camera_index = 1 if marker >= 100 else 0
            capture_index = marker - 100 if camera_index == 1 else marker
            return xr.Dataset(
                data_vars={
                    "shift_px": (
                        ("pixel_axis",),
                        base_shifts[camera_index] + capture_deltas[capture_index],
                    ),
                },
                coords={"pixel_axis": list(PIXEL_AXES)},
                attrs={"warnings": ""},
            )

        with patch(
            "merlin_track_position.tracking.calibration_core.estimate_shift",
            side_effect=fake_estimate_shift,
        ):
            result = get_correction(
                calibration,
                reference_cam0,
                current_cam0,
                reference_cam1,
                current_cam1,
                check_tiles=False,
                clip_percentiles=None,
            )

        self.assertEqual(result.attrs["capture_count"], 5)
        np.testing.assert_allclose(result["shift_px"].values, base_shifts)
        np.testing.assert_allclose(result["capture_shift_mad_px"].values, 0.1)
        np.testing.assert_allclose(result["estimated_stage_offset_um"].values, offset_um)
        np.testing.assert_allclose(result["correction_um"].values, -offset_um)
        np.testing.assert_array_equal(
            result["current_cam0"].values,
            np.full(reference_cam0.shape, 2.0, dtype=np.float32),
        )

    def test_do_correction_uses_reference_images_and_applies_motor_move(self):
        reference_cam0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        reference_cam1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        other_cam0 = reference_cam0 + 10.0
        other_cam1 = reference_cam1 + 10.0
        current_cam0 = reference_cam0 + 0.5
        current_cam1 = reference_cam1 + 0.5
        calibration = xr.Dataset(
            data_vars={
                "image_cam0": (
                    ("sample", "y_cam0", "x_cam0"),
                    np.stack([reference_cam0, other_cam0], axis=0),
                ),
                "image_cam1": (
                    ("sample", "y_cam1", "x_cam1"),
                    np.stack([reference_cam1, other_cam1], axis=0),
                ),
            },
            attrs={
                "roi_cam0_x": 0.0,
                "roi_cam0_y": 0.0,
                "roi_cam0_width": 1.0,
                "roi_cam0_height": 1.0,
                "roi_cam1_x": 0.0,
                "roi_cam1_y": 0.0,
                "roi_cam1_width": 1.0,
                "roi_cam1_height": 1.0,
            },
        )
        correction_dataset = xr.Dataset(
            data_vars={
                "correction_um": (
                    ("stage_axis",),
                    np.array([10.0, -20.0, 30.0]),
                ),
            },
            coords={"stage_axis": list(STAGE_AXES)},
        )
        captured = {}

        def pair_source():
            return current_cam0, current_cam1

        def fake_get_correction(
            calibration_arg,
            reference_cam0_arg,
            current_cam0_arg,
            reference_cam1_arg,
            current_cam1_arg,
            **shift_kwargs,
        ):
            captured["calibration"] = calibration_arg
            captured["reference_cam0"] = reference_cam0_arg
            captured["current_cam0"] = current_cam0_arg
            captured["reference_cam1"] = reference_cam1_arg
            captured["current_cam1"] = current_cam1_arg
            captured["shift_kwargs"] = shift_kwargs
            return correction_dataset

        with (
            patch(
                "merlin_track_position.tracking.correct.get_correction",
                side_effect=fake_get_correction,
            ),
            patch(
                "merlin_track_position.tracking.correct.get_positions",
                return_value=(1.0, 2.0, 3.0),
            ),
            patch(
                "merlin_track_position.tracking.correct.move_motors_and_wait",
                return_value=(1.01, 1.98, 3.03),
            ) as move_motors_and_wait,
        ):
            result = do_correction(
                calibration,
                camera_pair_from_pair_source(pair_source),
                move_tolerance_um=(1.0, 2.0, 3.0),
                max_retries=7,
                capture_count=1,
                check_tiles=False,
                clip_percentiles=None,
            )

        self.assertIs(captured["calibration"], calibration)
        np.testing.assert_allclose(captured["reference_cam0"], reference_cam0)
        np.testing.assert_allclose(captured["reference_cam1"], reference_cam1)
        np.testing.assert_array_equal(captured["current_cam0"], current_cam0[None, ...])
        np.testing.assert_array_equal(captured["current_cam1"], current_cam1[None, ...])
        self.assertEqual(
            captured["shift_kwargs"],
            {"check_tiles": False, "clip_percentiles": None},
        )
        move_motors_and_wait.assert_called_once_with(
            ("x", "y", "z"),
            (1.01, 1.98, 3.03),
            tolerance=(0.001, 0.002, 0.003),
            max_retries=7,
        )
        self.assertTrue(result.attrs["correction_applied"])
        self.assertEqual(result.attrs["pre_move_x_mm"], 1.0)
        self.assertEqual(result.attrs["pre_move_y_mm"], 2.0)
        self.assertEqual(result.attrs["pre_move_z_mm"], 3.0)
        self.assertEqual(result.attrs["requested_x_mm"], 1.01)
        self.assertEqual(result.attrs["requested_y_mm"], 1.98)
        self.assertEqual(result.attrs["requested_z_mm"], 3.03)
        self.assertEqual(result.attrs["final_x_mm"], 1.01)
        self.assertEqual(result.attrs["final_y_mm"], 1.98)
        self.assertEqual(result.attrs["final_z_mm"], 3.03)

    def test_do_correction_crops_raw_images_from_calibration_roi_metadata(self):
        raw_cam0 = np.arange(5 * 6, dtype=float).reshape(5, 6)
        raw_cam1 = np.arange(6 * 7, dtype=float).reshape(6, 7)
        reference_cam0 = raw_cam0[2:4, 1:4]
        reference_cam1 = raw_cam1[3:5, 2:4]
        calibration = xr.Dataset(
            data_vars={
                "image_cam0": (
                    ("sample", "y_cam0", "x_cam0"),
                    reference_cam0[None, ...],
                ),
                "image_cam1": (
                    ("sample", "y_cam1", "x_cam1"),
                    reference_cam1[None, ...],
                ),
            },
            attrs={
                "roi_cam0_x": 1.0,
                "roi_cam0_y": 2.0,
                "roi_cam0_width": 3.0,
                "roi_cam0_height": 2.0,
                "roi_cam1_x": 2.0,
                "roi_cam1_y": 3.0,
                "roi_cam1_width": 2.0,
                "roi_cam1_height": 2.0,
            },
        )
        correction_dataset = xr.Dataset(
            data_vars={
                "correction_um": (
                    ("stage_axis",),
                    np.array([0.0, 0.0, 0.0]),
                ),
            },
            coords={"stage_axis": list(STAGE_AXES)},
        )
        captured = {}

        def pair_source():
            return raw_cam0, raw_cam1

        def fake_get_correction(
            calibration_arg,
            reference_cam0_arg,
            current_cam0_arg,
            reference_cam1_arg,
            current_cam1_arg,
            **shift_kwargs,
        ):
            del calibration_arg, reference_cam0_arg, reference_cam1_arg, shift_kwargs
            captured["current_cam0"] = current_cam0_arg
            captured["current_cam1"] = current_cam1_arg
            return correction_dataset

        with (
            patch(
                "merlin_track_position.tracking.correct.get_correction",
                side_effect=fake_get_correction,
            ),
            patch(
                "merlin_track_position.tracking.correct.get_positions",
                return_value=(1.0, 2.0, 3.0),
            ),
            patch(
                "merlin_track_position.tracking.correct.move_motors_and_wait",
                return_value=(1.0, 2.0, 3.0),
            ),
        ):
            do_correction(
                calibration,
                camera_pair_from_pair_source(pair_source),
                capture_count=1,
            )

        np.testing.assert_array_equal(
            captured["current_cam0"],
            reference_cam0[None, ...],
        )
        np.testing.assert_array_equal(
            captured["current_cam1"],
            reference_cam1[None, ...],
        )
        self.assertFalse(np.shares_memory(captured["current_cam0"], raw_cam0))
        self.assertFalse(np.shares_memory(captured["current_cam1"], raw_cam1))

    def test_do_correction_computes_expected_correction_from_current_images(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        calibration = fit_calibration_from_images(
            as_capture_stacks(images_cam0),
            as_capture_stacks(images_cam1),
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
        )
        offset_um = np.array([8.0, -4.0, 6.0])
        shifts = np.einsum("cpk,k->cp", stage_to_pixel, offset_um)
        current_cam0 = ndimage.shift(
            images_cam0[0],
            shift=(shifts[0, 1], shifts[0, 0]),
            order=3,
            mode="wrap",
        )
        current_cam1 = ndimage.shift(
            images_cam1[0],
            shift=(shifts[1, 1], shifts[1, 0]),
            order=3,
            mode="wrap",
        )

        def pair_source():
            return current_cam0, current_cam1

        with (
            patch(
                "merlin_track_position.tracking.correct.get_positions",
                return_value=(1.0, 2.0, 3.0),
            ),
            patch(
                "merlin_track_position.tracking.correct.move_motors_and_wait",
                return_value=(0.992, 2.004, 2.994),
            ) as move_motors_and_wait,
        ):
            result = do_correction(
                calibration,
                camera_pair_from_pair_source(pair_source),
                move_tolerance_um=2.5,
                capture_count=1,
                check_tiles=False,
                clip_percentiles=None,
            )

        np.testing.assert_allclose(
            result["estimated_stage_offset_um"].values,
            offset_um,
            atol=1.0,
        )
        np.testing.assert_allclose(
            result["correction_um"].values,
            -offset_um,
            atol=1.0,
        )
        _, requested_position_mm = move_motors_and_wait.call_args.args
        np.testing.assert_allclose(
            requested_position_mm,
            np.array([1.0, 2.0, 3.0]) + result["correction_um"].values * 1e-3,
        )
        self.assertEqual(move_motors_and_wait.call_args.kwargs["tolerance"], 0.0025)
        self.assertEqual(move_motors_and_wait.call_args.kwargs["max_retries"], 4)
        self.assertTrue(result.attrs["correction_applied"])

    def test_do_correction_rejects_non_finite_correction_before_move(self):
        image = np.zeros((2, 2), dtype=float)
        calibration = xr.Dataset(
            data_vars={
                "image_cam0": (("sample", "y_cam0", "x_cam0"), image[None, ...]),
                "image_cam1": (("sample", "y_cam1", "x_cam1"), image[None, ...]),
            }
        )
        correction_dataset = xr.Dataset(
            data_vars={
                "correction_um": (
                    ("stage_axis",),
                    np.array([1.0, np.nan, 3.0]),
                ),
            },
            coords={"stage_axis": list(STAGE_AXES)},
        )

        def pair_source():
            return image, image

        with (
            patch(
                "merlin_track_position.tracking.correct.get_correction",
                return_value=correction_dataset,
            ),
            patch(
                "merlin_track_position.tracking.correct.get_positions",
            ) as get_positions,
            patch(
                "merlin_track_position.tracking.correct.move_motors_and_wait",
            ) as move_motors_and_wait,
        ):
            with self.assertRaisesRegex(ValueError, "finite values"):
                do_correction(
                    calibration,
                    camera_pair_from_pair_source(pair_source),
                    capture_count=1,
                )

        get_positions.assert_not_called()
        move_motors_and_wait.assert_not_called()

    def test_fit_calibration_requires_independent_stage_axes(self):
        stage = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        reference_cam0 = textured_image(seed=70, shape=(96, 104))
        reference_cam1 = textured_image(seed=71, shape=(96, 104))

        with self.assertRaisesRegex(ValueError, "three independent motor axes"):
            fit_calibration_from_images(
                as_capture_stacks([reference_cam0] * len(stage)),
                as_capture_stacks([reference_cam1] * len(stage)),
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
            )

    def test_fit_calibration_requires_full_rank_camera_matrix(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage_to_pixel[:, :, 2] = 0.0
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)

        with self.assertRaisesRegex(ValueError, "matrix must have rank 3"):
            fit_calibration_from_images(
                as_capture_stacks(images_cam0),
                as_capture_stacks(images_cam1),
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
            )

    def test_fit_calibration_requires_first_stage_row_at_origin(self):
        stage = calibration_stage()
        stage[0, 0] = 1.0
        reference_cam0 = textured_image(seed=95, shape=(96, 104))
        reference_cam1 = textured_image(seed=96, shape=(96, 104))

        with self.assertRaisesRegex(ValueError, "stage_um\\[0\\]"):
            fit_calibration_from_images(
                as_capture_stacks([reference_cam0] * len(stage)),
                as_capture_stacks([reference_cam1] * len(stage)),
                stage,
                origin_stability_um=ORIGIN_STABILITY_UM,
                check_tiles=False,
                clip_percentiles=None,
            )

    def test_fit_calibration_warns_for_return_to_origin_image_error(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        final_error_um = np.array([10.0, 0.0, 0.0])
        final_shifts = np.einsum("cpk,k->cp", stage_to_pixel, final_error_um)
        images_cam0[-1] = ndimage.shift(
            images_cam0[0],
            shift=(final_shifts[0, 1], final_shifts[0, 0]),
            order=3,
            mode="wrap",
        )
        images_cam1[-1] = ndimage.shift(
            images_cam1[0],
            shift=(final_shifts[1, 1], final_shifts[1, 0]),
            order=3,
            mode="wrap",
        )

        calibration = fit_calibration_from_images(
            as_capture_stacks(images_cam0),
            as_capture_stacks(images_cam1),
            stage,
            origin_stability_um=ORIGIN_STABILITY_UM,
            check_tiles=False,
            clip_percentiles=None,
        )

        self.assertIn("return-to-origin image error", calibration.attrs["warnings"])
        self.assertNotIn("return_to_origin_image_error_norm_um", calibration.data_vars)

    def test_sample_calibration_generator_uses_two_camera_schema(self):
        dataset = build_sample_calibration_dataset(
            image_shape_cam0=(48, 64),
            image_shape_cam1=(54, 72),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample_calibration.h5"
            dataset.to_netcdf(path, engine="h5netcdf")
            with xr.open_dataset(path, engine="h5netcdf") as dataset_on_disk:
                loaded = dataset_on_disk.load()

        _validate_calibration_dataset(loaded)
        self.assertEqual(loaded.attrs["format_version"], "1")
        self.assertNotIn("format", loaded.attrs)
        self.assertNotIn("model", loaded.attrs)
        self.assertEqual(loaded["stage_um"].shape, (8, len(STAGE_AXES)))
        self.assertEqual(loaded["stage_to_pixel"].shape, (2, 2, 3))
        self.assertNotIn("pixel_to_stage", loaded.data_vars)
        self.assertNotIn("origin_stability_um", loaded.data_vars)
        self.assertNotIn("repeatability_mean_rms_std_px", loaded.data_vars)
        self.assertEqual(loaded["image_cam0"].dtype, np.float32)
        self.assertEqual(loaded["image_cam1"].dtype, np.float32)
        self.assertEqual(loaded["image_cam0"].shape, (8, 48, 64))
        self.assertEqual(loaded["image_cam1"].shape, (8, 54, 72))


if __name__ == "__main__":
    unittest.main()
