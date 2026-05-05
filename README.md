# merlin-track-position

Offline single-camera 2D shift detection for grayscale image pairs.

The package estimates the subpixel image displacement between a reference image
and a current image, then converts that image shift into two calibrated motor
axis corrections in microns.

Depth-of-field information is intentionally ignored in this first version.

## Quick Start

```python
import numpy as np
from merlin_track_position import estimate_shift

# reference and current are 2D grayscale NumPy arrays.
reference = np.load("reference.npy")
current = np.load("current.npy")

shift = estimate_shift(reference, current)
print(shift["shift_px"].values)  # [du_px, dv_px]
print(shift.attrs["warnings"])
```

## Calibration

Fit calibration directly from image arrays and known motor positions:

```python
import numpy as np
from merlin_track_position import correct, fit_calibration_from_images

# images is a list of 2D grayscale NumPy arrays.
# stage_um has one [stage_a_um, stage_b_um] row per image.
images = [img_center, img_a_plus, img_a_minus, img_b_plus, img_b_minus]
stage_um = np.array([
    [0.0, 0.0],
    [50.0, 0.0],
    [-50.0, 0.0],
    [0.0, 50.0],
    [0.0, -50.0],
])

calibration = fit_calibration_from_images(images, stage_um)
result = correct(calibration, reference, current)

print(result["shift_px"].values)
print(result["estimated_stage_offset_um"].values)
print(result["correction_um"].values)
print(result.attrs["warnings"])
```

The calibration model is:

```text
pixel_shift = J @ [stage_a_um, stage_b_um] + bias
motor_correction_um = -inv(J) @ measured_pixel_shift
```

No assumption is made that the motor axes are aligned with camera pixel axes.

## Xarray And HDF5

Calibration results are xarray datasets. The dataset keeps the calibration image
stack, motor coordinates, measured shifts, fitted matrix, residuals, and
warnings together.

```python
import xarray as xr

calibration.to_netcdf("calibration.h5", engine="h5netcdf")

with xr.open_dataset("calibration.h5", engine="h5netcdf") as dataset_on_disk:
    calibration = dataset_on_disk.load()
```

The main dataset variables are:

- `image(sample, y, x)`
- `stage_um(sample, stage_axis)`
- `measured_shift_px(sample, pixel_axis)`
- `predicted_shift_px(sample, pixel_axis)`
- `residual_shift_px(sample, pixel_axis)`
- `residual_stage_um(sample, stage_axis)`
- `stage_to_pixel(pixel_axis, stage_axis)`
- `pixel_to_stage(stage_axis, pixel_axis)`
- `bias_px(pixel_axis)`

## Recommended Data Collection

Run a scout sweep from center on each motor axis with `5, 10, 25, 50, 100,
200 um`. Pick a base step that produces roughly `20-100 px` image displacement
while keeping the same features visible.

For the first calibration, capture images at center, `+/-S` and `+/-2S` on each
motor axis, and the four `(+/-S_A, +/-S_B)` corners. Three repeats per position
are recommended for repeatability diagnostics.

Motor movement and backlash-safe approach are handled outside this code. The
stage positions you pass beside the arrays should record the final settled
offsets.

## Diagnostics

Shift and calibration results include:

- measured pixel shift
- motor correction in microns
- skimage registration error and phase difference
- calibration residuals in pixels and microns
- repeatability by calibration position
- calibration matrix condition number
- warnings for low texture, ambiguous peaks, low confidence, inconsistent local
  shifts, and poorly conditioned calibration

## Accuracy And Morphology Limits

`estimate_shift` uses `skimage.registration.phase_cross_correlation` for the
subpixel solve. The default `upsample_factor=50` resolves the correlation peak
on a `1/50 px` grid, so clean, well-textured synthetic images usually land
within a few hundredths of a pixel. Real sample images should be judged from
calibration residuals, repeat captures, and tile-consistency warnings.

Whole-frame matching is most reliable when the moving sample structure dominates
the field of view. If static background, repeated patterns, focus changes, or
non-rigid sample deformation dominate, the returned shift can be biased even
when a numerical correlation peak exists.

## Tests

```bash
uv run python -m unittest discover -v
```
