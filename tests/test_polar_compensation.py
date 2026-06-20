import unittest

import numpy as np

from merlin_track_position.tracking.polar_compensation import (
    apply_polar_compensation_model,
    fit_polar_compensation_model,
    polar_rotation_matrix,
    predict_polar_compensation_from_attrs,
    predict_polar_compensation_xz,
)


class PolarCompensationTests(unittest.TestCase):
    def test_fit_recovers_anchor_to_center_vector_and_center(self):
        anchor_polar = 0.0
        anchor = np.asarray([1.25, -0.4])
        radius = np.asarray([0.6, -0.2])
        polar = np.asarray([-20.0, -12.5, -5.0, anchor_polar])
        predicted = predict_polar_compensation_xz(
            polar,
            anchor_polar_deg=anchor_polar,
            anchor_xz_mm=anchor,
            anchor_to_center_xz_mm=radius,
        )

        model = fit_polar_compensation_model(
            polar,
            predicted[:, 0],
            np.full(polar.shape, 2.0),
            predicted[:, 1],
            anchor_polar_deg=anchor_polar,
        )

        self.assertAlmostEqual(
            model.attrs["polar_compensation_anchor_to_center_x_mm"],
            radius[0],
        )
        self.assertAlmostEqual(
            model.attrs["polar_compensation_anchor_to_center_z_mm"],
            radius[1],
        )
        self.assertAlmostEqual(
            model.attrs["polar_compensation_center_x_mm"],
            anchor[0] - radius[0],
        )
        self.assertAlmostEqual(
            model.attrs["polar_compensation_center_z_mm"],
            anchor[1] - radius[1],
        )
        self.assertLess(model.attrs["polar_compensation_residual_max_um"], 1e-8)

    def test_rotation_sign_matches_project_polar_convention(self):
        rotation = polar_rotation_matrix(90.0)
        np.testing.assert_allclose(
            rotation @ np.asarray([1.0, 0.0]),
            [0.0, -1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rotation @ np.asarray([0.0, 1.0]),
            [1.0, 0.0],
            atol=1e-12,
        )

    def test_anchor_point_is_not_a_fit_row(self):
        anchor_polar = 0.0
        anchor = np.asarray([1.0, 2.0])
        radius = np.asarray([0.3, 0.5])
        non_anchor = np.asarray([-20.0, -12.5])
        polar = np.asarray([*non_anchor, anchor_polar])
        predicted = predict_polar_compensation_xz(
            polar,
            anchor_polar_deg=anchor_polar,
            anchor_xz_mm=anchor,
            anchor_to_center_xz_mm=radius,
        )

        model = fit_polar_compensation_model(
            polar,
            predicted[:, 0],
            np.zeros(polar.shape),
            predicted[:, 1],
            anchor_polar_deg=anchor_polar,
        )

        residual = np.asarray(model["polar_compensation_residual_mm"].values)
        np.testing.assert_allclose(residual[-1], [0.0, 0.0], atol=1e-12)

    def test_rejects_duplicate_or_missing_anchor_angles(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_polar_compensation_model(
                [-5.0, -5.0, 0.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                anchor_polar_deg=0.0,
            )
        with self.assertRaisesRegex(ValueError, "anchor_polar_deg"):
            fit_polar_compensation_model(
                [-20.0, -12.5, -5.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                anchor_polar_deg=0.0,
            )

    def test_apply_model_to_calibration_and_predict_from_attrs(self):
        anchor = np.asarray([1.0, 2.0])
        radius = np.asarray([0.5, -0.25])
        polar = np.asarray([-20.0, -12.5, -5.0, 0.0])
        predicted = predict_polar_compensation_xz(
            polar,
            anchor_polar_deg=0.0,
            anchor_xz_mm=anchor,
            anchor_to_center_xz_mm=radius,
        )
        model = fit_polar_compensation_model(
            polar,
            predicted[:, 0],
            np.zeros(polar.shape),
            predicted[:, 1],
            anchor_polar_deg=0.0,
        )
        calibration = apply_polar_compensation_model(model.drop_vars(model.data_vars), model)

        np.testing.assert_allclose(
            predict_polar_compensation_from_attrs(calibration, -10.0),
            predict_polar_compensation_xz(
                -10.0,
                anchor_polar_deg=0.0,
                anchor_xz_mm=anchor,
                anchor_to_center_xz_mm=radius,
            ),
        )

    def test_fit_stores_probe_current_images(self):
        anchor = np.asarray([1.0, 2.0])
        radius = np.asarray([0.5, -0.25])
        polar = np.asarray([-20.0, -12.5, -5.0, 0.0])
        predicted = predict_polar_compensation_xz(
            polar,
            anchor_polar_deg=0.0,
            anchor_xz_mm=anchor,
            anchor_to_center_xz_mm=radius,
        )
        current_cam0 = np.arange(polar.size * 2 * 3, dtype=np.uint16).reshape(
            polar.size,
            2,
            3,
        )
        current_cam1 = np.arange(polar.size * 3 * 4 * 3, dtype=np.uint16).reshape(
            polar.size,
            3,
            4,
            3,
        )

        model = fit_polar_compensation_model(
            polar,
            predicted[:, 0],
            np.zeros(polar.shape),
            predicted[:, 1],
            anchor_polar_deg=0.0,
            current_cam0=current_cam0,
            current_cam1=current_cam1,
        )
        calibration = apply_polar_compensation_model(
            model.drop_vars(model.data_vars),
            model,
        )

        self.assertEqual(
            calibration["polar_compensation_current_cam0"].dims,
            (
                "polar_compensation_probe",
                "polar_compensation_y_cam0",
                "polar_compensation_x_cam0",
            ),
        )
        self.assertEqual(
            calibration["polar_compensation_current_cam1"].dims,
            (
                "polar_compensation_probe",
                "polar_compensation_y_cam1",
                "polar_compensation_x_cam1",
                "polar_compensation_channel_cam1",
            ),
        )
        np.testing.assert_array_equal(
            calibration["polar_compensation_current_cam0"].values,
            current_cam0,
        )
        np.testing.assert_array_equal(
            calibration["polar_compensation_current_cam1"].values,
            current_cam1,
        )

    def test_rejects_mismatched_probe_image_count(self):
        anchor = np.asarray([1.0, 2.0])
        radius = np.asarray([0.5, -0.25])
        polar = np.asarray([-20.0, -12.5, -5.0, 0.0])
        predicted = predict_polar_compensation_xz(
            polar,
            anchor_polar_deg=0.0,
            anchor_xz_mm=anchor,
            anchor_to_center_xz_mm=radius,
        )

        with self.assertRaisesRegex(ValueError, "one image per polar probe"):
            fit_polar_compensation_model(
                polar,
                predicted[:, 0],
                np.zeros(polar.shape),
                predicted[:, 1],
                anchor_polar_deg=0.0,
                current_cam0=np.zeros((polar.size - 1, 2, 3), dtype=np.uint16),
            )


if __name__ == "__main__":
    unittest.main()
