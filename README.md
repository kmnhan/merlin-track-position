# merlin-track-position

Two-camera 3D sample shift detection and calibration for grayscale image pairs.

The package estimates subpixel image displacement in two camera views, then
converts the four observed pixel shifts into calibrated `x`, `y`, and `z` motor
axis corrections in microns.

## Quick Start

```python
from merlin_track_position.tracking.calibration_core import correct

result = correct(
    calibration,
    reference_cam0,
    current_cam0,
    reference_cam1,
    current_cam1,
)

print(result["shift_px"].values)  # shape: (camera, pixel_axis)
print(result["estimated_stage_offset_um"].values)  # [x_um, y_um, z_um]
print(result["correction_um"].values)
print(result.attrs["warnings"])
```

## Calibration

Fit calibration directly from two image stacks and known motor positions:

```python
from merlin_track_position.tracking.calibration_core import fit_calibration_from_images

calibration = fit_calibration_from_images(
    images_cam0,
    images_cam1,
    stage_um,
    origin_stability_um=5.0,
)
```

The calibration model is:

```text
[du_cam0, dv_cam0, du_cam1, dv_cam1] = J @ [x_um, y_um, z_um]
motor_correction_um = -pinv(J) @ measured_pixel_shift
```

No assumption is made that motor axes are aligned with camera pixel axes.

## Xarray And HDF5

Calibration results are xarray datasets. `format_version` remains `"1"` for the
current two-camera schema.

The main dataset variables are:

- `image_cam0(sample, y_cam0, x_cam0)`
- `image_cam1(sample, y_cam1, x_cam1)`
- `stage_um(sample, stage_axis)`
- `measured_shift_px(sample, camera, pixel_axis)`
- `stage_to_pixel(camera, pixel_axis, stage_axis)`
- `measurement_warnings(sample, camera)` when image matching reports warnings

Saved calibration attributes include `format_version`, `warnings`, initial
motor context (`initial_x_mm`, `initial_y_mm`, `initial_z_mm`, `polar`,
`tilt`), and GUI ROI bounds (`roi_cam0_*`, `roi_cam1_*`) when created from the
GUI.

## Hardware Notes

Camera 0 uses the existing framegrabber path. Camera 1 has a Basler placeholder:
development mode uses the simulator, while acquisition-PC mode raises
`NotImplementedError` until the Basler framework is connected.

Active motor correction from the reconstructed `x/y/z` displacement is still
deferred; the UI move-trigger handler currently acknowledges the trigger and
contains the TODO for that future control loop.

## Tests

```bash
uv run python -m unittest discover -v
```
