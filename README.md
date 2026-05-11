# merlin-track-position

Two-camera 3D sample shift detection and calibration for grayscale image pairs.

The package estimates subpixel image displacement in two camera views, then
converts the four observed pixel shifts into calibrated `x`, `y`, and `z` motor
axis corrections in microns.

## Quick Start

```python
from merlin_track_position.tracking.calibration import correct

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
from merlin_track_position.tracking.calibration import fit_calibration_from_images

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
- `predicted_shift_px(sample, camera, pixel_axis)`
- `residual_shift_px(sample, camera, pixel_axis)`
- `residual_stage_um(sample, stage_axis)`
- `stage_to_pixel(camera, pixel_axis, stage_axis)`
- `pixel_to_stage(stage_axis, observation_axis)`
- `origin_stability_um`
- `return_to_origin_motor_error_um(stage_axis)`
- `return_to_origin_motor_error_norm_um`
- `return_to_origin_image_error_px(camera, pixel_axis)`
- `return_to_origin_image_error_um(stage_axis)`
- `return_to_origin_image_error_norm_um`

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
