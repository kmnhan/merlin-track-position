# Image-Space Visual Servoing for MERLIN Sample-Position Correction

This repository implements image-based sample-position tracking and correction
for Beamline 4.0.3 MERLIN at the Advanced Light Source. The method estimates a
local visual Jacobian between commanded motor displacements and two-camera image
displacements, then applies closed-loop damped visual servoing in the same
command coordinate system used by the beamline control software.

The imaging system consists of the existing sample-view camera exposed through
the FrameGrabber window and a Basler acA1440-73gm camera viewing the same
viewport from a different angle. The two views have distinct pixel scales,
fields of view, and image content; therefore, all correction is formulated in a
joint four-dimensional image space rather than in either individual camera
coordinate system.

## Method

### Variables

Let the two-camera image residual be

$$
\mathbf p =
\begin{bmatrix}
\Delta u_{\mathrm{cam0}} \\
\Delta v_{\mathrm{cam0}} \\
\Delta u_{\mathrm{cam1}} \\
\Delta v_{\mathrm{cam1}}
\end{bmatrix}.
$$

Let the commanded motor state be

$$
\mathbf a =
\begin{bmatrix}
a_x \\
a_y \\
a_z
\end{bmatrix},
$$

where each component denotes an absolute BCS motor command in millimeters. The
calibration and correction algorithms use command increments

$$
\Delta \mathbf a =
\begin{bmatrix}
\Delta a_x \\
\Delta a_y \\
\Delta a_z
\end{bmatrix}
$$

in the same commanded-mm coordinate system.

The local image-space response is modeled as

$$
\Delta \mathbf p \approx J\,\Delta \mathbf a,
$$

with

$$
J \in \mathbb R^{4\times3}
$$

with units of pixels per commanded millimeter. In the xarray calibration
dataset, `J` is stored as
`visual_jacobian_px_per_cmd_mm(camera, pixel_axis, command_axis)` and is
reshaped to a \(4\times3\) observation matrix only for linear algebra.

### Calibration Routine

Calibration estimates \(J\) from before/after image pairs acquired around
single-axis commanded-mm probe moves. The acquisition procedure is:

1. read the initial BCS `x`, `y`, and `z` command positions;
2. acquire `reference_cam0` and `reference_cam1`;
3. generate repeated positive and negative single-axis probes for `x`, `y`, and
   `z`;
4. for each probe, acquire pre-move images, command the move, acquire post-move
   images, and register the post-move images against the pre-move images;
5. estimate `visual_jacobian_px_per_cmd_mm` from the valid
   `(probe_command_delta_mm, probe_measured_delta_px)` rows;
6. persist the calibration dataset to disk and reload it from that path.

The default probe magnitude is

$$
\Delta a_x = \Delta a_y = \Delta a_z = 0.5\ \mathrm{mm}
$$

for each plus/minus probe direction, with
`DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION = 3`.

Motor readback is retained only as diagnostic metadata:

```python
pre_readback_position_mm
post_readback_position_mm
```

These readback quantities are excluded from the Jacobian fit. The regression
uses commanded deltas, defined by

```python
post_commanded_position_mm - pre_commanded_position_mm
    == probe_command_delta_mm
```

The fitted model is

$$
J =
\frac{\partial \mathbf p}{\partial \mathbf a},
$$

and not

$$
\frac{\partial \mathbf p}{\partial \mathbf x_{\mathrm{physical}}}.
$$

The estimator is a robust iteratively reweighted least-squares regression over
probe residuals. Calibration is rejected if:

- a probe image response is below `DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX`
  (`2.0 px` by default);
- the commanded probe deltas do not span the three command axes;
- the fitted visual Jacobian has rank less than 3;
- the condition number exceeds
  `DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING` (`100.0` by default).

### Axis Scale

For numerical conditioning, correction is posed in normalized command
coordinates. Each calibration dataset therefore stores

```text
axis_scale_cmd_mm(command_axis)
```

and subsequent correction runs reuse this saved scale.

Let

$$
S = \text{diag}(s_x, s_y, s_z),
$$

where

$$
\mathbf s =
\begin{bmatrix}
s_x \\
s_y \\
s_z
\end{bmatrix}
$$

as represented by `axis_scale_cmd_mm`.

The scale is derived from the fitted visual Jacobian. The quantity

$$
c_j = \|J_{:,j}\|_2
$$

is the image sensitivity of command axis \(j\), expressed in
px/commanded-mm. Given the calibration probe magnitude

$$
h_j = \max_i |\Delta a_{i,j}|,
$$

the observed per-axis probe response scale is

$$
r_j = c_j h_j.
$$

The target response is the median response over axes:

$$
r_\star = \text{median}(r_x, r_y, r_z).
$$

The unclamped command scale is

$$
s_j^{\mathrm{raw}} = \frac{r_\star}{c_j}.
$$

The saved correction scale is clamped to configured operational bounds:

$$
s_j =
\text{clip}
\left(
s_j^{\mathrm{raw}},
s_{j,\min},
s_{j,\max}
\right).
$$

The configured bounds are:

```text
x: 0.1 to 0.8 commanded-mm
y: 0.3 to 1.0 commanded-mm
z: 0.1 to 0.8 commanded-mm
```

The calibration dataset stores only the final `axis_scale_cmd_mm`. The
unclamped scale, sensitivity, bounds, and target response are not persisted
because they are exactly derivable from `J`, `probe_command_delta_mm`, and the
configured constants. The GUI recomputes those diagnostics when displaying a
calibration.

### Weighted Correction Objective

Correction minimizes the weighted image-space residual

$$
\rho(\mathbf p) =
\sqrt{\mathbf p^\mathsf T W \mathbf p}.
$$

When no weights are specified,

$$
W = I_4.
$$

Weights, when supplied, must contain four finite nonnegative observation
weights with at least one positive entry. They may also have shape
`(camera, pixel_axis)`, in which case they are flattened to the same
observation order as \(\mathbf p\).

Convergence is declared when

$$
\rho(\mathbf p_k)
\le
\text{tol},
$$

where `tol` is `DEFAULT_CORRECTION_PIXEL_TOLERANCE_PX`, with a default value
of `0.5 px`.

This is the only correction residual recorded in the iteration history:

```text
iteration_weighted_residual_px
```

Unweighted RMS residuals are not part of the correction output.

### Damped Normalized Command Solve

Let

$$
\Delta \mathbf a = S\,\Delta \mathbf q
$$

and

$$
J_q = J S.
$$

At correction iteration \(k\), the command increment is obtained from

$$
\Delta \mathbf q
=
-\lambda
\left(
J_q^\mathsf T W J_q + \mu I
\right)^{-1}
J_q^\mathsf T W \mathbf p.
$$

The commanded-mm correction is then

$$
\Delta \mathbf a = S\,\Delta \mathbf q.
$$

The default numerical parameters are:

```text
lambda = DEFAULT_CORRECTION_GAIN = 0.3
lambda_min = DEFAULT_CORRECTION_MIN_GAIN = 0.05
mu = DEFAULT_CORRECTION_DAMPING_MU = 1e-2
max |Delta q_j| = DEFAULT_CORRECTION_MAX_NORMALIZED_STEP = 0.5
min per-axis predicted response =
    DEFAULT_CORRECTION_MIN_AXIS_PREDICTED_SHIFT_PX = 0.25 px
min command norm = DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM = 1e-9 mm
max_moves = DEFAULT_CORRECTION_MAX_MOVES = 12
```

Before a motor command is issued, two safeguards are applied to the
least-squares proposal. First, the full normalized vector is scaled when

$$
\max_j |\Delta q_j|
>
\texttt{DEFAULT\_CORRECTION\_MAX\_NORMALIZED\_STEP}.
$$

Second, axis components are suppressed when their predicted weighted image
effect is below the configured threshold:

$$
|\Delta a_j|\sqrt{J_{:,j}^\mathsf T WJ_{:,j}}
<
\texttt{DEFAULT\_CORRECTION\_MIN\_AXIS\_PREDICTED\_SHIFT\_PX}.
$$

Components at or below the configured correction command deadband are also set to zero.
If the remaining command vector is effectively zero, correction stops before
issuing another motor command and reports non-convergence with a warning.

The absolute BCS-mm target sent to the motors is

$$
\mathbf a_{\mathrm{request}}
=
\mathbf a_{\mathrm{commanded}} + \Delta \mathbf a.
$$

The internal `commanded_position_mm` state is initialized from the BCS `x`,
`y`, and `z` values and is subsequently advanced to each requested absolute
target. Post-move readback is not used as the command-state anchor.

### Correction Loop

The closed-loop correction algorithm proceeds as follows:

1. load and validate a saved calibration dataset;
2. require a real calibration file path, because accepted Jacobian refinements
   are persisted to disk;
3. capture images and compute `shift_px`;
4. compute
   $\rho_k = \sqrt{\mathbf p_k^\mathsf T W \mathbf p_k}$;
5. stop immediately if $\rho_k$ is at or below the pixel tolerance;
6. compute the damped normalized command correction;
7. apply the normalized-step cap and suppress low-impact axis components;
8. stop before motion if the remaining command vector is below the minimum
   command norm;
9. send absolute BCS-mm targets only for axes with nonzero correction components;
10. re-image and compute the new residual;
11. if the residual decreased, refit the visual Jacobian from the calibration
   probes plus accepted correction observations and save the refined
   calibration dataset to disk;
12. if the residual increased or stayed flat, skip the Jacobian update, halve
    the gain down to `DEFAULT_CORRECTION_MIN_GAIN`, double `mu`, and continue
    from the newly measured image state;
13. save the correction result into the sibling correction-history file.

No rollback is performed after a residual-increasing move. The image
measurement acquired after that move defines the next closed-loop state.

If convergence is not reached after `max_moves`, correction returns a dataset
with

```text
correction_converged = False
```

and an associated warning. Lack of convergence after motion is therefore
reported as data rather than raised as an exception.

### Jacobian Refinement

After a move, the measured image change is

$$
\Delta \mathbf p_{\mathrm{meas}}
=
\mathbf p_{k+1} - \mathbf p_k.
$$

Accepted correction moves are treated as additional observations of the same
linear model used during calibration:

$$
\Delta \mathbf p_i \approx J \Delta \mathbf a_i.
$$

The refined Jacobian is not overwritten from one correction move. It is refit
by pooling the original calibration probes with all accepted correction
observations:

$$
\widehat{J}
=
\arg\min_J
\sum_i
\rho
\left(
\left\|
\Delta \mathbf p_i - J\Delta \mathbf a_i
\right\|_2
\right),
$$

where the first rows are the large calibration probes and later rows are
closed-loop correction moves. The estimate is obtained with the same
Huber-style iteratively reweighted least-squares procedure used during
calibration.

This weighting is important for small corrections. In the pooled least-squares
normal equation, an observation contributes through
$\Delta\mathbf a_i\Delta\mathbf a_i^\mathsf T$. A 5 micron correction therefore
has approximately $(0.005/0.5)^2 = 10^{-4}$ the Jacobian leverage of a 0.5 mm
calibration probe, preventing a single small move from dominating a Jacobian
column.

Refinement is only attempted when the weighted image residual decreased. If the
refit is rank-deficient, poorly conditioned, or otherwise invalid, it is skipped
and the warning is recorded. Accepted refinements update
`visual_jacobian_px_per_cmd_mm`, increment `jacobian_refinement_count`, mark
`jacobian_refined = "true"`, validate the dataset, and save it back to the
calibration file path.

### Dataset Schema

The required calibration variables are:

```text
visual_jacobian_px_per_cmd_mm(camera, pixel_axis, command_axis)
axis_scale_cmd_mm(command_axis)
reference_cam0(y_cam0, x_cam0)
reference_cam1(y_cam1, x_cam1)
probe_command_delta_mm(probe, command_axis)
probe_measured_delta_px(probe, camera, pixel_axis)
pre_commanded_position_mm(probe, command_axis)
post_commanded_position_mm(probe, command_axis)
pre_readback_position_mm(probe, command_axis)
post_readback_position_mm(probe, command_axis)
```

Calibration datasets also include non-derivable diagnostics:

```text
probe_capture_shift_mad_px(probe, camera, pixel_axis)
probe_registration_warnings(probe, camera)
```

Persisted calibration datasets omit values that can be recomputed exactly, such
as predicted probe shifts, probe residuals, axis sensitivities, scale bounds,
and condition number. `calibration_path` is also not written into the file; it
is attached only to loaded in-memory datasets.

Correction outputs are expressed in commanded-mm units:

```text
estimated_command_offset_mm(command_axis)
correction_cmd_mm(command_axis)
shift_px(camera, pixel_axis)
iteration_shift_px(iteration, camera, pixel_axis)
iteration_weighted_residual_px(iteration)
move_command_delta_mm(move, command_axis)
move_requested_position_mm(move, command_axis)
move_final_readback_position_mm(move, command_axis)
move_pre_weighted_residual_px(move)
move_post_weighted_residual_px(move)
move_predicted_delta_px(move, camera, pixel_axis)
move_measured_delta_px(move, camera, pixel_axis)
move_visual_jacobian_before_px_per_cmd_mm(move, camera, pixel_axis, command_axis)
move_visual_jacobian_after_px_per_cmd_mm(move, camera, pixel_axis, command_axis)
move_gain(move)
move_damping_mu(move)
move_max_normalized_component(move)
move_active_axis_mask(move, command_axis)
move_jacobian_refined(move)
```

The correction result keeps per-move diagnostic state because that dataset is
also used as the on-disk correction log. For a calibration file such as
`calibration.h5`, corrections are saved next to it in
`calibration_corrections.h5`. Each correction run is written to an HDF5 group
named `run_000000`, `run_000001`, and so on. During an active correction, the
active run group is rewritten after every completed motor move, so the file
contains the latest available residual trace, move commands, measured image
response, and Jacobian before/after any accepted pooled least-squares
refinement.

At result assembly, the reported command offset is computed from the final image
residual as

$$
\widehat{\Delta \mathbf a}
=
\left(J^\mathsf T WJ\right)^+
J^\mathsf T W\mathbf p,
$$

where `+` denotes the pseudoinverse. This is stored as
`estimated_command_offset_mm`. The reported `correction_cmd_mm` is zero when
the loop has converged; otherwise it is the next damped correction proposal
computed with the final gain and damping values.

### References

[1] S. Duan, S. Wang, Y. Yang, C. Huang, L. Gu, H. Liu, and W. Zhang, "A sample-position-autocorrection system with precision better than $1\,\mu\mathrm{m}$ in angle-resolved photoemission experiments," *Review of Scientific Instruments* **93**, 103905, 2022. <https://doi.org/10.1063/5.0106299>

[2] F. Chaumette and S. Hutchinson, "Visual Servo Control, Part I: Basic Approaches," *IEEE Robotics & Automation Magazine* **13**(4), 82--90, 2006. <https://doi.org/10.1109/MRA.2006.250573>

[3] M. Jagersand, O. Fuentes, and R. Nelson, "Experimental Evaluation of Uncalibrated Visual Servoing for Precision Manipulation," *Proceedings of the IEEE International Conference on Robotics and Automation*, 1997, pp. 2874--2880. <https://doi.org/10.1109/ROBOT.1997.606723>

[4] K. Levenberg, "A Method for the Solution of Certain Non-Linear Problems in Least Squares," *Quarterly of Applied Mathematics* **2**, 164--168, 1944. <https://doi.org/10.1090/qam/10666>; D. W. Marquardt, "An Algorithm for Least-Squares Estimation of Nonlinear Parameters," *Journal of the Society for Industrial and Applied Mathematics* **11**(2), 431--441, 1963. <https://doi.org/10.1137/0111030>

[5] S. Chiaverini, B. Siciliano, and O. Egeland, "Review of the Damped Least-Squares Inverse Kinematics with Experiments on an Industrial Robot Manipulator," *IEEE Transactions on Control Systems Technology* **2**(2), 123--134, 1994. <https://doi.org/10.1109/87.294335>

[6] P. J. Huber, "Robust Estimation of a Location Parameter," *The Annals of Mathematical Statistics* **35**(1), 73--101, 1964. <https://doi.org/10.1214/aoms/1177703732>

[7] J. Music, M. Bonkovic, and M. Cecic, "Comparison of Uncalibrated Model-Free Visual Servoing Methods for Small-Amplitude Movements: A Simulation Study," *International Journal of Advanced Robotic Systems* **11**, 2014. <https://doi.org/10.5772/58822>

[8] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006. <https://doi.org/10.1007/978-0-387-40065-5>

## Programmatic Use

```python
from merlin_track_position.tracking.correct import do_correction

result = do_correction("visual_jacobian_calibration.h5")

print(result["shift_px"].values)  # shape: (camera, pixel_axis)
print(result["estimated_command_offset_mm"].values)  # [x_mm, y_mm, z_mm]
print(result["correction_cmd_mm"].values)
print(result.attrs["correction_converged"])
print(result.attrs["warnings"])
```

## Calibration Entry Point

Calibration can be initiated from before/after commanded-mm
probe moves:

```python
from merlin_track_position.tracking.calibrate import run_calibration

calibration = run_calibration(
    output_path="visual_jacobian_calibration.h5",
)
```

The calibration model is:

```text
[du_cam0, dv_cam0, du_cam1, dv_cam1] = J @ [dx_cmd_mm, dy_cmd_mm, dz_cmd_mm]
correction_cmd_mm = -lambda * damped_wls(J, measured_pixel_shift)
```

No assumption is made that motor axes are aligned with camera pixel axes. Motor
readback is recorded as a diagnostic only; the fitted Jacobian uses commanded
BCS-mm deltas and measured image deltas.

## Xarray and HDF5

Calibration results are xarray datasets.

The main dataset variables are:

- `visual_jacobian_px_per_cmd_mm(camera, pixel_axis, command_axis)`
- `axis_scale_cmd_mm(command_axis)`
- `reference_cam0(y_cam0, x_cam0)`
- `reference_cam1(y_cam1, x_cam1)`
- `probe_command_delta_mm(probe, command_axis)`
- `probe_measured_delta_px(probe, camera, pixel_axis)`
- `pre_commanded_position_mm(probe, command_axis)`
- `post_commanded_position_mm(probe, command_axis)`
- `pre_readback_position_mm(probe, command_axis)`
- `post_readback_position_mm(probe, command_axis)`

Saved calibration datasets intentionally omit values that can be recomputed
exactly, such as predicted probe shifts, probe residuals, axis-scale
diagnostics, condition number, and the file path. Those diagnostics are
computed on demand by the GUI and correction code.

Saved calibration attributes include `warnings`, initial motor context
(`initial_x_mm`, `initial_y_mm`, `initial_z_mm`, `polar`, `tilt`), and GUI ROI
bounds (`roi_cam0_*`, `roi_cam1_*`) when created from the GUI. Accepted
closed-loop pooled least-squares refinements rewrite the calibration file so the
refined Jacobian persists across correction runs.

## Hardware Notes

Camera 0 uses the existing framegrabber path. Camera 1 has a Basler placeholder:
development mode uses the simulator, while acquisition-PC mode raises
`NotImplementedError` until the Basler framework is connected.

Python continues to command motors through BCS `MoveMotor` in millimeters. Y
backlash correction is assumed to be handled by BCS; Python-side backlash
compensation remains only for axes configured in `MOTOR_BACKLASH_CORRECTION`.

## Tests

```bash
uv run pytest -q
```
