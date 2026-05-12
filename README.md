# merlin-track-position

Software for tracking the sample position at Beamline 4.0.3 MERLIN at the Advanced Light
Source. The main goal is to implement a visual closed loop for sample position
correction, using the existing motor system and two cameras.

One camera is the existing camera that looks at the sample (#5), exposed through the
FrameGrabber window. The second camera is a new camera (Basler ac1440-73gm) that loos
through the same viewport at a slightly different angle and has a different field of
view and resolution.

## Design

### Variables

The image residual for the two cameras is

$$
\mathbf p =
\begin{bmatrix}
\Delta u_{\mathrm{cam0}} \\
\Delta v_{\mathrm{cam0}} \\
\Delta u_{\mathrm{cam1}} \\
\Delta v_{\mathrm{cam1}}
\end{bmatrix}.
$$

The absolute motor command state is

$$
\mathbf a =
\begin{bmatrix}
a_x \\
a_y \\
a_z
\end{bmatrix},
$$

where each component is the absolute commanded value sent for that motor.
Calibration and correction use command deltas

$$
\Delta \mathbf a =
\begin{bmatrix}
\Delta a_x \\
\Delta a_y \\
\Delta a_z
\end{bmatrix}
$$

in mm units.

The visual model is

$$
\Delta \mathbf p \approx J\,\Delta \mathbf a,
$$

with

$$
J \in \mathbb R^{4\times3}
$$

in units of pixels per commanded-mm. In the xarray calibration dataset,
`J` is stored as
`visual_jacobian_px_per_cmd_mm(camera, pixel_axis, command_axis)` and is
flattened to the 4 by 3 observation matrix only for linear algebra.

### Calibration Routine

Calibration learns `J` from images captured before and after moves commanded in mm
coordinates. The current acquisition path is:

1. read the initial BCS `x`, `y`, and `z` positions
2. capture `reference_cam0` and `reference_cam1`
3. build repeated plus/minus single-axis probe delta coordinates for `x`, `y`, and `z`
4. for each probe, capture before images, command the move,
   capture after images, and register after-vs-before;
5. fit `visual_jacobian_px_per_cmd_mm` from the valid
   `(probe_command_delta_mm, probe_measured_delta_px)` rows;
6. save the calibration dataset to disk and reload it from that path.

The default probe step is currently

$$
\Delta a_x = \Delta a_y = \Delta a_z = 0.5\ \mathrm{mm}
$$

for each plus/minus probe direction, with
`DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION = 3`.

Motor readback is saved as diagnostics only:

```python
pre_readback_position_mm
post_readback_position_mm
```

It is not used to fit `J`. The fit uses commanded deltas:

```python
post_commanded_position_mm - pre_commanded_position_mm
    == probe_command_delta_mm
```

The fitted model is

$$
J =
\frac{\partial \mathbf p}{\partial \mathbf a},
$$

not

$$
\frac{\partial \mathbf p}{\partial \mathbf x_{\mathrm{physical}}}.
$$

The core fit uses a robust iteratively reweighted least-squares solve over
probe residuals. Calibration fails if:

- a probe image response is below `DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX`
  (`2.0 px` by default);
- the commanded probe deltas do not span the three command axes;
- the fitted visual Jacobian has rank less than 3;
- the condition number exceeds
  `DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING` (`100.0` by default).

### Axis Scale

Correction is solved in normalized command coordinates. The calibration
therefore stores

```text
axis_scale_cmd_mm(command_axis)
```

and correction reuses that saved value.

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

which is stored in `axis_scale_cmd_mm`.

The code derives this scale from the fitted visual Jacobian:

$$
c_j = \|J_{:,j}\|_2
$$

is the image sensitivity of command axis `j` in px/commanded-mm. With
calibration probe step magnitude

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

The current bounds are:

```text
x: 0.1 to 0.8 commanded-mm
y: 0.3 to 1.0 commanded-mm
z: 0.1 to 0.8 commanded-mm
```

The calibration dataset stores only the final `axis_scale_cmd_mm`. The
unclamped scale, sensitivity, bounds, and target response are not persisted
because they are exactly derivable from `J`, `probe_command_delta_mm`, and the
current constants. The GUI recomputes those diagnostics when displaying a
calibration.

### Weighted Correction Objective

Correction uses the formal weighted image residual

$$
\rho(\mathbf p) =
\sqrt{\mathbf p^\mathsf T W \mathbf p}.
$$

If no weights are passed, the code uses

$$
W = I_4.
$$

If weights are supplied, they must contain four finite nonnegative observation
weights, with at least one positive value, or have shape `(camera, pixel_axis)`,
which is then flattened to the same observation order as `p`.

Convergence is declared when

$$
\rho(\mathbf p_k)
\le
\text{tol},
$$

Where `tol` is `DEFAULT_CORRECTION_PIXEL_TOLERANCE_PX`, which is currently `0.5 px`.

This is the only correction residual recorded in the iteration history:

```text
iteration_weighted_residual_px
```

The previous unweighted RMS correction residual is no longer part of the
current correction output.

### Damped Normalized Command Solve

Let

$$
\Delta \mathbf a = S\,\Delta \mathbf q
$$

and

$$
J_q = J S.
$$

Each correction proposal solves

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

The current defaults are:

```text
lambda = DEFAULT_CORRECTION_GAIN = 0.3
lambda_min = DEFAULT_CORRECTION_MIN_GAIN = 0.05
mu = DEFAULT_CORRECTION_DAMPING_MU = 1e-2
max_moves = DEFAULT_CORRECTION_MAX_MOVES = 8
```

The absolute BCS-mm target sent to the motors is

$$
\mathbf a_{\mathrm{request}}
=
\mathbf a_{\mathrm{commanded}} + \Delta \mathbf a.
$$

The implementation initializes `commanded_position_mm` from the current BCS
`x/y/z` values and then updates that internal command state to each requested
absolute target. It does not use post-move readback as the next command-state
anchor.

### Correction Loop

The implemented correction loop is:

1. load and validate a saved calibration dataset;
2. require a real calibration file path, because accepted Jacobian refinements
   are persisted back to disk;
3. capture current images and compute `shift_px`;
4. compute
   $\rho_k = \sqrt{\mathbf p_k^\mathsf T W \mathbf p_k}$;
5. stop immediately if $\rho_k$ is at or below the pixel tolerance;
6. compute the damped normalized command correction;
7. send the absolute BCS-mm target for `x`, `y`, and `z`;
8. re-image and compute the new residual;
9. if the residual decreased, apply a guarded blended Broyden update and save
   the refined calibration dataset to disk;
10. if the residual increased or stayed flat, skip the Jacobian update, halve
    the gain down to `DEFAULT_CORRECTION_MIN_GAIN`, double `mu`, and continue
    from the newly measured image state.

The code does not roll back the motor position after a bad move. The camera
measurement after that move becomes the next state.

If convergence is not reached after `max_moves`, correction returns a dataset
with

```text
correction_converged = False
```

and a warning. It does not raise merely because the closed loop did not
converge after motion.

### Broyden Refinement

After a move, the measured image change is

$$
\Delta \mathbf p_{\mathrm{meas}}
=
\mathbf p_{k+1} - \mathbf p_k.
$$

The predicted image change is

$$
\Delta \mathbf p_{\mathrm{pred}}
=
J_k \Delta \mathbf a_k.
$$

The implemented update is the blended rank-one Broyden update

$$
J_{k+1}
=
J_k
+
\alpha
\frac{
\left(
\Delta \mathbf p_{\mathrm{meas}}
- J_k\Delta \mathbf a_k
\right)
\Delta \mathbf a_k^\mathsf T
}{
\Delta \mathbf a_k^\mathsf T\Delta \mathbf a_k
}.
$$

The blend $\alpha$ is defined by `DEFAULT_BROYDEN_UPDATE_BLEND`, which is currently
`0.5` for a 50/50 blend of the previous Jacobian and the new rank-one update.

The update is only attempted when the weighted image residual decreased. The update also
requires a finite command vector with norm at least `1e-9 commanded-mm`. If the update
fails validation, it is skipped and the warning is recorded.

Accepted updates call `assign_refined_visual_jacobian(...)`, which updates
`visual_jacobian_px_per_cmd_mm`, increments `broyden_update_count`, marks
`jacobian_refined = "true"`, validates the dataset, and saves it back to the calibration
file path.

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

The current calibration writer also includes non-derivable diagnostics:

```text
probe_capture_shift_mad_px(probe, camera, pixel_axis)
probe_registration_warnings(probe, camera)
```

The writer deliberately omits values that can be recomputed exactly, such as
predicted probe shifts, probe residuals, axis sensitivities, scale bounds, and
condition number. `calibration_path` is also not written into the file; it is
attached only to loaded in-memory datasets.

The current correction output uses commanded-mm names:

```text
estimated_command_offset_mm(command_axis)
correction_cmd_mm(command_axis)
shift_px(camera, pixel_axis)
iteration_shift_px(iteration, camera, pixel_axis)
iteration_weighted_residual_px(iteration)
move_command_delta_mm(move, command_axis)
move_requested_position_mm(move, command_axis)
move_final_readback_position_mm(move, command_axis)
move_gain(move)
move_damping_mu(move)
move_jacobian_updated(move)
```

The correction result omits redundant per-move state. Residual decrease can be
recovered from consecutive `iteration_weighted_residual_px` values, and the
final gain/damping are stored as summary attributes.

At result assembly, the reported command offset is computed from the current
image residual as

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

[6] C. G. Broyden, "A Class of Methods for Solving Nonlinear Simultaneous Equations," *Mathematics of Computation* **19**(92), 577--593, 1965. <https://doi.org/10.1090/S0025-5718-1965-0198670-6>

[7] J. Music, M. Bonkovic, and M. Cecic, "Comparison of Uncalibrated Model-Free Visual Servoing Methods for Small-Amplitude Movements: A Simulation Study," *International Journal of Advanced Robotic Systems* **11**, 2014. <https://doi.org/10.5772/58822>

[8] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006. <https://doi.org/10.1007/978-0-387-40065-5>

## Quick Start

```python
from merlin_track_position.tracking.correct import do_correction

result = do_correction("visual_jacobian_calibration.h5")

print(result["shift_px"].values)  # shape: (camera, pixel_axis)
print(result["estimated_command_offset_mm"].values)  # [x_mm, y_mm, z_mm]
print(result["correction_cmd_mm"].values)
print(result.attrs["correction_converged"])
print(result.attrs["warnings"])
```

## Calibration

Run visual-Jacobian calibration from before/after commanded-mm probe moves:

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

## Xarray And HDF5

Calibration results are xarray datasets with one current visual-Jacobian
schema.

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
closed-loop Broyden updates rewrite the calibration file so the refined
Jacobian persists across correction runs.

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
