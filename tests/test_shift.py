import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy import ndimage

from merlin_track_position.tracking.shift import estimate_shift, normalize_intensity


CORPUS_ARRAYS_DIR = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "shift_image_corpus" / "arrays"
)


def corpus_image(name: str) -> np.ndarray:
    with np.load(CORPUS_ARRAYS_DIR / f"{name}.npz") as archive:
        return archive["image"].copy()


def corpus_grayscale_image() -> np.ndarray:
    return corpus_image("khan_cns_calibration_local_cam0")


def corpus_uint16_image() -> np.ndarray:
    return corpus_image("khan_cns_calibration_local_cam1")


def corpus_rgb_image() -> np.ndarray:
    return corpus_image("khan_sro2_azim47p5_reference_cam1")


class ShiftTests(unittest.TestCase):
    def test_integer_shift(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(7, -11), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        np.testing.assert_allclose(result["shift_px"].values, [-11.0, 7.0], atol=0.15)
        self.assertEqual(set(result.data_vars), {"shift_px"})
        self.assertEqual(set(result.attrs), {"warnings"})

    def test_subpixel_shift(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(-4.4, 6.25), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        np.testing.assert_allclose(result["shift_px"].values, [6.25, -4.4], atol=0.25)

    def test_opencv_phase_registration_sign_convention(self):
        reference = corpus_grayscale_image()

        with patch(
            "merlin_track_position.tracking.shift.cv2.phaseCorrelate",
            side_effect=(
                ((1.25, -2.5), 0.95),
                ((-1.25, 2.5), 0.95),
            ),
        ) as phase_correlate:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
            )

        self.assertEqual(phase_correlate.call_count, 2)
        self.assertEqual(
            phase_correlate.call_args_list[0].args[0].dtype,
            np.dtype(np.float32),
        )
        self.assertEqual(
            phase_correlate.call_args_list[0].args[1].dtype,
            np.dtype(np.float32),
        )
        np.testing.assert_allclose(result["shift_px"].values, [1.25, -2.5])

    def test_uint16_phase_registration_uses_float32_work_images(self):
        reference = corpus_uint16_image()

        with patch(
            "merlin_track_position.tracking.shift.cv2.phaseCorrelate",
            side_effect=(
                ((1.25, -2.5), 0.95),
                ((-1.25, 2.5), 0.95),
            ),
        ) as phase_correlate:
            estimate_shift(reference, reference.copy(), check_tiles=False)

        self.assertEqual(phase_correlate.call_args_list[0].args[0].ndim, 2)
        self.assertEqual(phase_correlate.call_args_list[0].args[1].ndim, 2)
        self.assertEqual(
            phase_correlate.call_args_list[0].args[0].dtype,
            np.dtype(np.float32),
        )
        self.assertEqual(
            phase_correlate.call_args_list[0].args[1].dtype,
            np.dtype(np.float32),
        )

    def test_tile_consistency_uses_opencv_phase_registration(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(1.25, -1.75), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=True)

        np.testing.assert_allclose(result["shift_px"].values, [-1.75, 1.25], atol=0.45)
        self.assertNotIn(
            "tile shift estimates are inconsistent", result.attrs["warnings"]
        )

    def test_ecc_tile_consistency_uses_phase_baseline(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(1.25, -1.75), order=3, mode="wrap")
        ecc_shift = np.asarray(
            [
                [1.0, 0.0, 25.0],
                [0.0, 1.0, -20.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            return_value=(0.99, ecc_shift),
        ):
            result = estimate_shift(
                reference,
                current,
                check_tiles=True,
                use_ecc_refinement=True,
            )

        np.testing.assert_allclose(result["shift_px"].values, [25.0, -20.0])
        self.assertNotIn(
            "tile shift estimates are inconsistent", result.attrs["warnings"]
        )

    def test_ecc_refinement_keeps_translation_sign_convention(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(2.75, -4.25), order=3, mode="wrap")

        result = estimate_shift(
            reference,
            current,
            check_tiles=False,
            use_ecc_refinement=True,
        )

        np.testing.assert_allclose(result["shift_px"].values, [-4.25, 2.75], atol=0.1)

    def test_ecc_refinement_returns_homography_displacement_at_center(self):
        reference = corpus_grayscale_image().astype(np.float32)
        height, width = reference.shape
        center_x = (width - 1.0) / 2.0
        center_y = (height - 1.0) / 2.0
        scale = 1.015
        expected_shift = np.asarray([-3.25, 2.5], dtype=np.float64)
        warp = np.asarray(
            [
                [
                    scale,
                    0.0,
                    expected_shift[0] + (1.0 - scale) * center_x,
                ],
                [
                    0.0,
                    scale,
                    expected_shift[1] + (1.0 - scale) * center_y,
                ],
            ],
            dtype=np.float32,
        )
        current = cv2.warpAffine(
            reference.astype(np.float32),
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        result = estimate_shift(
            reference,
            current,
            check_tiles=False,
            use_ecc_refinement=True,
            clip_percentiles=None,
        )

        np.testing.assert_allclose(result["shift_px"].values, expected_shift, atol=0.1)

    def test_ecc_refinement_returns_homography_displacement_at_reference_point(self):
        reference = corpus_grayscale_image().astype(np.float32)
        height, width = reference.shape
        center_x = (width - 1.0) / 2.0
        center_y = (height - 1.0) / 2.0
        scale = 1.012
        center_shift = np.asarray([-2.0, 3.0], dtype=np.float64)
        warp = np.asarray(
            [
                [
                    scale,
                    0.0,
                    center_shift[0] + (1.0 - scale) * center_x,
                ],
                [
                    0.0,
                    scale,
                    center_shift[1] + (1.0 - scale) * center_y,
                ],
            ],
            dtype=np.float32,
        )
        current = cv2.warpAffine(
            reference.astype(np.float32),
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        point = np.asarray([center_x + 24.0, center_y - 16.0], dtype=np.float64)
        expected_shift = warp.astype(np.float64) @ np.r_[point, 1.0] - point

        result = estimate_shift(
            reference,
            current,
            check_tiles=False,
            use_ecc_refinement=True,
            ecc_reference_point_px=(float(point[0]), float(point[1])),
            clip_percentiles=None,
        )

        np.testing.assert_allclose(result["shift_px"].values, expected_shift, atol=0.1)

    def test_ecc_refinement_uses_homography_motion_model(self):
        reference = corpus_grayscale_image()
        point = np.asarray([76.25, 88.5], dtype=np.float64)
        homography = np.asarray(
            [
                [1.01, 0.015, -2.0],
                [-0.01, 0.995, 3.0],
                [1.0e-4, -2.0e-4, 1.0],
            ],
            dtype=np.float32,
        )
        mapped_homogeneous = homography.astype(np.float64) @ np.r_[point, 1.0]
        expected_shift = mapped_homogeneous[:2] / mapped_homogeneous[2] - point

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            return_value=(1.0, homography),
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_reference_point_px=point,
            )

        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_HOMOGRAPHY)
        self.assertEqual(find_ecc.call_args.args[2].shape, (3, 3))
        np.testing.assert_allclose(result["shift_px"].values, expected_shift)

    def test_rgb_input_uses_basler_grayscale_phase_seed(self):
        reference = corpus_rgb_image()
        red, green, blue = np.moveaxis(reference, -1, 0)
        basler_gray = 0.25 * red + 0.625 * green + 0.125 * blue
        rec709_gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue

        with patch(
            "merlin_track_position.tracking.shift.cv2.phaseCorrelate",
            side_effect=(
                ((1.5, -2.25), 0.95),
                ((-1.5, 2.25), 0.95),
            ),
        ) as phase_correlate:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                clip_percentiles=None,
            )

        self.assertEqual(phase_correlate.call_args_list[0].args[0].ndim, 2)
        self.assertEqual(phase_correlate.call_args_list[0].args[1].ndim, 2)
        self.assertEqual(
            phase_correlate.call_args_list[0].args[0].dtype,
            np.dtype(np.float32),
        )
        np.testing.assert_allclose(
            phase_correlate.call_args_list[0].args[0],
            normalize_intensity(basler_gray, clip_percentiles=None),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertFalse(
            np.allclose(
                phase_correlate.call_args_list[0].args[0],
                normalize_intensity(rec709_gray, clip_percentiles=None),
                rtol=1e-6,
                atol=1e-6,
            )
        )
        np.testing.assert_allclose(result["shift_px"].values, [1.5, -2.25])

    def test_uint16_grayscale_ecc_refinement_uses_native_images(self):
        reference = corpus_uint16_image()
        initial_shift = np.asarray([3.0, -4.0], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=initial_shift,
            )

        self.assertEqual(find_ecc.call_args.args[0].ndim, 2)
        self.assertEqual(find_ecc.call_args.args[1].ndim, 2)
        self.assertEqual(find_ecc.call_args.args[0].dtype, np.dtype(np.uint16))
        self.assertEqual(find_ecc.call_args.args[1].dtype, np.dtype(np.uint16))
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_int32_grayscale_ecc_refinement_converts_to_float32(self):
        reference = corpus_uint16_image().astype(np.int32)
        initial_shift = np.asarray([3.0, -4.0], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=initial_shift,
            )

        reference_work = find_ecc.call_args.args[0]
        current_work = find_ecc.call_args.args[1]
        self.assertEqual(reference_work.dtype, np.dtype(np.float32))
        self.assertEqual(current_work.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(reference_work, reference.astype(np.float32))
        np.testing.assert_allclose(current_work, reference.astype(np.float32))
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_rgb_ecc_refinement_uses_native_color_images(self):
        reference = corpus_rgb_image()
        initial_shift = np.asarray([3.0, -4.0], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=initial_shift,
            )

        self.assertEqual(find_ecc.call_args.args[0].ndim, 3)
        self.assertEqual(find_ecc.call_args.args[1].ndim, 3)
        self.assertEqual(find_ecc.call_args.args[0].shape[-1], 3)
        self.assertEqual(find_ecc.call_args.args[0].dtype, np.dtype(np.uint16))
        self.assertEqual(find_ecc.call_args.args[1].dtype, np.dtype(np.uint16))
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_ecc_window_converts_to_float32_and_applies_hanning_window(self):
        reference = np.full((5, 5), 100, dtype=np.uint16)
        initial_shift = np.asarray([3.0, -4.0], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=initial_shift,
                ecc_use_window=True,
            )

        reference_work = find_ecc.call_args.args[0]
        current_work = find_ecc.call_args.args[1]
        self.assertEqual(reference_work.dtype, np.dtype(np.float32))
        self.assertEqual(current_work.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(reference_work[0, :], 0.0)
        np.testing.assert_allclose(reference_work[:, 0], 0.0)
        self.assertAlmostEqual(float(reference_work[2, 2]), 100.0)
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_ecc_refinement_uses_affine_motion_model(self):
        reference = corpus_grayscale_image()
        point = np.asarray([71.5, 83.25], dtype=np.float64)
        affine = np.asarray(
            [
                [1.01, 0.015, -2.0],
                [-0.01, 0.995, 3.0],
            ],
            dtype=np.float32,
        )
        expected_shift = affine.astype(np.float64) @ np.r_[point, 1.0] - point

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            return_value=(1.0, affine),
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_motion_model="affine",
                ecc_reference_point_px=point,
            )

        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_AFFINE)
        self.assertEqual(find_ecc.call_args.args[2].shape, (2, 3))
        np.testing.assert_allclose(result["shift_px"].values, expected_shift)

    def test_ecc_refinement_uses_explicit_initial_shift_for_homography(self):
        reference = corpus_grayscale_image()
        initial_shift = np.asarray([8.25, -5.5], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=initial_shift,
            )

        initial_warp = find_ecc.call_args.args[2]
        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_HOMOGRAPHY)
        np.testing.assert_allclose(initial_warp[:2, 2], initial_shift)
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_ecc_refinement_uses_explicit_initial_shift_for_affine(self):
        reference = corpus_grayscale_image()
        initial_shift = np.asarray([-6.5, 3.75], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_motion_model="affine",
                ecc_initial_shift_px=initial_shift,
            )

        initial_warp = find_ecc.call_args.args[2]
        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_AFFINE)
        np.testing.assert_allclose(initial_warp[:, 2], initial_shift)
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_ecc_refinement_passes_gauss_filter_size_to_opencv(self):
        reference = corpus_grayscale_image()
        initial_shift = np.asarray([-6.5, 3.75], dtype=np.float64)

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_motion_model="affine",
                ecc_initial_shift_px=initial_shift,
                ecc_gauss_filter_size=3,
            )

        self.assertEqual(find_ecc.call_args.args[6], 3)
        np.testing.assert_allclose(result["shift_px"].values, initial_shift)

    def test_ecc_refinement_rejects_even_gauss_filter_size(self):
        reference = corpus_grayscale_image()

        with self.assertRaisesRegex(
            ValueError,
            "ecc_gauss_filter_size must be a positive odd integer",
        ):
            estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_gauss_filter_size=4,
            )

    def test_ecc_refinement_uses_explicit_initial_affine_warp(self):
        reference = corpus_grayscale_image()
        point = np.asarray([61.25, 70.5], dtype=np.float64)
        initial_warp = np.asarray(
            [
                [0.95, -0.12, 8.0],
                [0.08, 1.02, -6.0],
            ],
            dtype=np.float64,
        )
        expected_shift = initial_warp @ np.r_[point, 1.0] - point

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            result = estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_motion_model="affine",
                ecc_reference_point_px=point,
                ecc_initial_warp=initial_warp,
            )

        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_AFFINE)
        np.testing.assert_allclose(find_ecc.call_args.args[2], initial_warp)
        np.testing.assert_allclose(result["shift_px"].values, expected_shift, atol=1e-5)

    def test_ecc_refinement_converts_affine_initial_warp_for_homography(self):
        reference = corpus_grayscale_image()
        initial_warp = np.asarray(
            [
                [1.0, 0.05, -2.0],
                [-0.03, 0.98, 4.0],
            ],
            dtype=np.float64,
        )

        def echo_initial_warp(*args):
            return 1.0, args[2].copy()

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=echo_initial_warp,
        ) as find_ecc:
            estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_warp=initial_warp,
            )

        expected = np.eye(3, dtype=np.float64)
        expected[:2, :] = initial_warp
        self.assertEqual(find_ecc.call_args.args[3], cv2.MOTION_HOMOGRAPHY)
        np.testing.assert_allclose(find_ecc.call_args.args[2], expected)

    def test_ecc_refinement_rejects_invalid_initial_warp(self):
        reference = corpus_grayscale_image()

        with self.assertRaisesRegex(ValueError, "ecc_initial_warp"):
            estimate_shift(
                reference,
                reference.copy(),
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_warp=np.zeros((2, 2), dtype=float),
            )

    def test_ecc_refinement_failure_falls_back_to_phase_shift(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(-1.5, 2.0), order=3, mode="wrap")

        baseline = estimate_shift(reference, current, check_tiles=False)
        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=RuntimeError("forced failure"),
        ):
            result = estimate_shift(
                reference,
                current,
                check_tiles=False,
                use_ecc_refinement=True,
            )

        np.testing.assert_allclose(
            result["shift_px"].values, baseline["shift_px"].values
        )
        self.assertIn("ECC refinement failed: forced failure", result.attrs["warnings"])

    def test_ecc_refinement_failure_can_disable_phase_fallback(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(-1.5, 2.0), order=3, mode="wrap")

        with patch(
            "merlin_track_position.tracking.shift.cv2.findTransformECC",
            side_effect=RuntimeError("forced failure"),
        ):
            result = estimate_shift(
                reference,
                current,
                check_tiles=False,
                use_ecc_refinement=True,
                ecc_initial_shift_px=(0.0, 0.0),
                ecc_fallback_to_phase_shift=False,
            )

        self.assertTrue(np.isnan(result["shift_px"].values).all())
        self.assertIn("ECC refinement failed: forced failure", result.attrs["warnings"])

    def test_opencv_phase_registration_does_not_emit_high_error_warning(self):
        reference = corpus_grayscale_image()
        current = ndimage.shift(reference, shift=(0.4, -0.6), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        self.assertNotIn("high registration error", result.attrs["warnings"])

    def test_brightness_and_noise_robustness(self):
        reference = corpus_grayscale_image()
        shifted = ndimage.shift(reference, shift=(3.5, -5.5), order=3, mode="wrap")
        rng = np.random.default_rng(4)
        current = 2.5 * shifted + 10.0 + rng.normal(scale=0.02, size=shifted.shape)

        result = estimate_shift(reference, current, check_tiles=False)

        np.testing.assert_allclose(result["shift_px"].values, [-5.5, 3.5], atol=0.35)

    def test_low_texture_warning(self):
        reference = np.ones((96, 96))
        current = np.ones((96, 96))

        result = estimate_shift(reference, current, check_tiles=False)

        warnings = result.attrs["warnings"]
        self.assertTrue(np.isnan(result["shift_px"].values).all())
        self.assertIn("little or no intensity contrast", warnings)
        self.assertIn("low texture", warnings)
        self.assertIn("registration skipped", warnings)


if __name__ == "__main__":
    unittest.main()
