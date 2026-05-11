import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position.interface.calibration_panel import _validate_calibration_dataset
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    PIXEL_AXES,
    STAGE_AXES,
    correct,
    estimate_stage_offset,
    fit_calibration_from_images,
)
from merlin_track_position.tracking.calibrate import calibration_sample_count, run_calibration
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


class CalibrationTests(unittest.TestCase):
    def test_calibration_sample_count_matches_path_shape(self):
        self.assertEqual(calibration_sample_count(3), 16)
        self.assertEqual(calibration_sample_count(5), 22)

    def test_calibration_sample_count_rejects_too_few_axis_points(self):
        with self.assertRaisesRegex(ValueError, "n must be >= 2"):
            calibration_sample_count(1)

    def test_run_calibration_records_initial_stage_and_tilt_attrs(self):
        image = np.zeros((8, 8), dtype=float)
        captured_kwargs = {}

        def fake_image_generator():
            return image, image

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
                fake_image_generator,
                origin_stability_um=ORIGIN_STABILITY_UM,
            )

        self.assertEqual(calibration.attrs["initial_x_mm"], 1.2)
        self.assertEqual(calibration.attrs["initial_y_mm"], 2.3)
        self.assertEqual(calibration.attrs["initial_z_mm"], 3.4)
        self.assertEqual(calibration.attrs["polar"], 4.5)
        self.assertEqual(calibration.attrs["tilt"], 6.7)
        np.testing.assert_allclose(captured_kwargs["stage_um"][0], [0.0, 0.0, 0.0])

    def test_fit_calibration_from_stereo_numpy_arrays(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)

        calibration = fit_calibration_from_images(
            images_cam0,
            images_cam1,
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

    def test_fit_calibration_reports_measurement_warning_step_and_camera(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        image = np.zeros((16, 16), dtype=float)
        images = [image for _ in range(stage.shape[0])]
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

    def test_h5_roundtrip_preserves_stereo_schema(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        dataset = fit_calibration_from_images(
            images_cam0,
            images_cam1,
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

    def test_correct_uses_two_camera_shift(self):
        stage_to_pixel = stereo_stage_to_pixel()
        stage = calibration_stage()
        images_cam0, images_cam1 = make_stereo_images(stage, stage_to_pixel)
        calibration = fit_calibration_from_images(
            images_cam0,
            images_cam1,
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

        result = correct(
            calibration,
            calibration["image_cam0"].isel(sample=0).values,
            current_cam0,
            calibration["image_cam1"].isel(sample=0).values,
            current_cam1,
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
                [reference_cam0] * len(stage),
                [reference_cam1] * len(stage),
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
                images_cam0,
                images_cam1,
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
                [reference_cam0] * len(stage),
                [reference_cam1] * len(stage),
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
            images_cam0,
            images_cam1,
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
