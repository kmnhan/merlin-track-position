import unittest
from inspect import signature

import numpy as np
from scipy import ndimage

from merlin_track_position.tracking.shift import estimate_shift


def textured_image(seed=1, shape=(192, 224)):
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(size=shape), sigma=2.0, mode="wrap")
    y, x = np.indices(shape)
    image += 0.3 * np.sin(x / 7.0) + 0.2 * np.cos(y / 11.0)
    return image


class ShiftTests(unittest.TestCase):
    def test_optional_registration_checks_are_disabled_by_default(self):
        parameters = signature(estimate_shift).parameters

        self.assertEqual(parameters["clip_percentiles"].default, (1.0, 99.0))
        self.assertIs(parameters["use_window"].default, False)
        self.assertEqual(parameters["upsample_factor"].default, 50)
        self.assertEqual(parameters["normalization"].default, "phase")
        self.assertIs(parameters["check_tiles"].default, False)

    def test_integer_shift(self):
        reference = textured_image()
        current = ndimage.shift(reference, shift=(7, -11), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        np.testing.assert_allclose(result["shift_px"].values, [-11.0, 7.0], atol=0.15)
        self.assertEqual(set(result.data_vars), {"shift_px"})
        self.assertEqual(set(result.attrs), {"warnings"})

    def test_subpixel_shift(self):
        reference = textured_image(seed=2)
        current = ndimage.shift(reference, shift=(-4.4, 6.25), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        np.testing.assert_allclose(result["shift_px"].values, [6.25, -4.4], atol=0.25)

    def test_default_phase_registration_does_not_emit_high_error_warning(self):
        reference = textured_image(seed=5)
        current = ndimage.shift(reference, shift=(0.4, -0.6), order=3, mode="wrap")

        result = estimate_shift(reference, current, check_tiles=False)

        self.assertNotIn("high registration error", result.attrs["warnings"])

    def test_brightness_and_noise_robustness(self):
        reference = textured_image(seed=3)
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
        self.assertIn("registration error is not finite", warnings)


if __name__ == "__main__":
    unittest.main()
