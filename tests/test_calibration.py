import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position import (
    correct,
    estimate_stage_offset,
    fit_calibration_from_images,
    fit_calibration_from_measurements,
)


def textured_image(seed=10, shape=(160, 176)):
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="wrap")
    y, x = np.indices(shape)
    image += 0.3 * np.sin(x / 9.0) + 0.2 * np.cos(y / 13.0)
    image -= image.min()
    image /= image.max()
    return image


class CalibrationTests(unittest.TestCase):
    def test_fit_calibration_from_measurements(self):
        stage_to_pixel = np.array([[0.45, -0.12], [0.08, 0.37]])
        stage = np.array(
            [
                [0.0, 0.0],
                [50.0, 0.0],
                [-50.0, 0.0],
                [0.0, 50.0],
                [0.0, -50.0],
                [50.0, 50.0],
                [-50.0, 50.0],
            ]
        )
        pixels = stage @ stage_to_pixel.T

        calibration = fit_calibration_from_measurements(stage, pixels)

        np.testing.assert_allclose(
            calibration["stage_to_pixel"].values, stage_to_pixel, atol=1e-12
        )
        np.testing.assert_allclose(
            estimate_stage_offset(calibration, [22.5, 4.0]), [50.0, 0.0], atol=1e-10
        )
        self.assertLess(float(calibration["condition_number"].values), 2.0)

    def test_poor_condition_warning(self):
        stage_to_pixel = np.array([[1.0, 0.999], [0.0, 0.001]])
        stage = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        pixels = stage @ stage_to_pixel.T

        calibration = fit_calibration_from_measurements(
            stage,
            pixels,
            condition_warning_threshold=10.0,
        )

        self.assertIn("poorly conditioned", calibration.attrs["warnings"])

    def test_fit_calibration_from_numpy_arrays(self):
        reference = textured_image(seed=20)
        stage_to_pixel = np.array([[0.27, -0.14], [0.09, 0.31]])
        stage = np.array(
            [
                [0.0, 0.0],
                [30.0, 0.0],
                [-30.0, 0.0],
                [0.0, 30.0],
                [0.0, -30.0],
                [30.0, 30.0],
                [-30.0, 30.0],
            ]
        )
        images = []
        for stage_row in stage:
            du, dv = stage_to_pixel @ stage_row
            images.append(
                ndimage.shift(reference, shift=(dv, du), order=3, mode="wrap")
            )

        calibration = fit_calibration_from_images(
            images,
            stage,
            check_tiles=False,
            clip_percentiles=None,
        )

        np.testing.assert_allclose(
            calibration["stage_to_pixel"].values, stage_to_pixel, atol=0.03
        )
        np.testing.assert_allclose(
            calibration["image"].values, np.stack(images), atol=0.0
        )
        np.testing.assert_allclose(calibration["stage_um"].values, stage)
        self.assertEqual(calibration.attrs["reference_index"], 0)

    def test_xarray_dataset_includes_image_stack(self):
        reference = textured_image(seed=40, shape=(96, 104))
        stage_to_pixel = np.array([[0.3, -0.1], [0.08, 0.22]])
        stage = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]])
        images = [
            ndimage.shift(
                reference,
                shift=tuple((stage_to_pixel @ row)[::-1]),
                order=3,
                mode="wrap",
            )
            for row in stage
        ]

        dataset = fit_calibration_from_images(
            images,
            stage,
            check_tiles=False,
            clip_percentiles=None,
        )

        self.assertEqual(dataset["image"].dims, ("sample", "y", "x"))
        self.assertEqual(dataset["stage_to_pixel"].dims, ("pixel_axis", "stage_axis"))
        np.testing.assert_allclose(dataset["image"].values, np.stack(images))
        np.testing.assert_allclose(dataset["stage_um"].values, stage)

    def test_h5_roundtrip_preserves_images_and_calibration(self):
        reference = textured_image(seed=50, shape=(96, 104))
        stage_to_pixel = np.array([[0.29, -0.12], [0.07, 0.25]])
        stage = np.array(
            [[0.0, 0.0], [25.0, 0.0], [-25.0, 0.0], [0.0, 25.0], [25.0, 25.0]]
        )
        images = [
            ndimage.shift(
                reference,
                shift=tuple((stage_to_pixel @ row)[::-1]),
                order=3,
                mode="wrap",
            )
            for row in stage
        ]
        dataset = fit_calibration_from_images(
            images,
            stage,
            check_tiles=False,
            clip_percentiles=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.h5"
            dataset.to_netcdf(path, engine="h5netcdf")

            with xr.open_dataset(path, engine="h5netcdf") as dataset_on_disk:
                loaded = dataset_on_disk.load()

        np.testing.assert_allclose(loaded["image"].values, np.stack(images))
        np.testing.assert_allclose(
            loaded["stage_um"].values, dataset["stage_um"].values
        )
        np.testing.assert_allclose(
            loaded["stage_to_pixel"].values, dataset["stage_to_pixel"].values
        )
        np.testing.assert_allclose(loaded["bias_px"].values, dataset["bias_px"].values)

    def test_correct_uses_xarray_dataset(self):
        reference = textured_image(seed=60, shape=(128, 136))
        stage_to_pixel = np.array([[0.25, -0.1], [0.05, 0.2]])
        stage = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]])
        images = [
            ndimage.shift(
                reference,
                shift=tuple((stage_to_pixel @ row)[::-1]),
                order=3,
                mode="wrap",
            )
            for row in stage
        ]
        calibration = fit_calibration_from_images(
            images, stage, check_tiles=False, clip_percentiles=None
        )
        current = ndimage.shift(
            reference,
            shift=tuple((stage_to_pixel @ np.array([8.0, -4.0]))[::-1]),
            order=3,
            mode="wrap",
        )

        result = correct(
            calibration, reference, current, check_tiles=False, clip_percentiles=None
        )

        np.testing.assert_allclose(
            result["estimated_stage_offset_um"].values, [8.0, -4.0], atol=1.0
        )
        np.testing.assert_allclose(
            result["correction_um"].values, [-8.0, 4.0], atol=1.0
        )

    def test_fit_calibration_requires_independent_stage_axes(self):
        stage = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        pixels = np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 2.0]])

        with self.assertRaises(ValueError):
            fit_calibration_from_measurements(stage, pixels)


if __name__ == "__main__":
    unittest.main()
