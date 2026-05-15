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

```math
\mathbf p =
\begin{bmatrix}
\Delta u_{\mathrm{cam0}} \\
\Delta v_{\mathrm{cam0}} \\
\Delta u_{\mathrm{cam1}} \\
\Delta v_{\mathrm{cam1}}
\end{bmatrix}.
```

This is the measured displacement of the sample in the two camera images, expressed in
pixels. Let the command vector be

```math
\mathbf a =
\begin{bmatrix}
a_x \\
a_y \\
a_z
\end{bmatrix},
```

where each component denotes an absolute BCS motor move command in millimeters. The
correction algorithm works with increments:

```math
\Delta \mathbf a =
\begin{bmatrix}
\Delta a_x \\
\Delta a_y \\
\Delta a_z
\end{bmatrix}.
```

The local image-space response is approximated by

```math
\Delta \mathbf p \approx J\,\Delta \mathbf a,
```

with

```math
J \in \mathbb R^{4\times3}
```

so that $J$ maps a 3-component motor command displacement to a 4-component image
displacement.

In matrix form,

```math
\begin{bmatrix}
\Delta u_{\mathrm{cam0}} \\
\Delta v_{\mathrm{cam0}} \\
\Delta u_{\mathrm{cam1}} \\
\Delta v_{\mathrm{cam1}}
\end{bmatrix}
\approx
\begin{bmatrix}
J_{u0,x} & J_{u0,y} & J_{u0,z} \\
J_{v0,x} & J_{v0,y} & J_{v0,z} \\
J_{u1,x} & J_{u1,y} & J_{u1,z} \\
J_{v1,x} & J_{v1,y} & J_{v1,z}
\end{bmatrix}
\begin{bmatrix}
\Delta a_x \\
\Delta a_y \\
\Delta a_z
\end{bmatrix}.
```

Each entry has the units of pixels per commanded millimeter. Note that this is not the
readback from the encoders, but the commanded move. Encoder readback is not reliable due
to backlash, hysteresis, and other nonlinearities, so the Jacobian is fit only from
commanded moves and measured image responses.

### Calibration Routine

Prior to any correction, the visual Jacobian \(J\) must be estimated from data. The
process is essentially a finite-difference linear response measurement.

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

```math
\Delta a_x = \Delta a_y = \Delta a_z = 0.5\ \mathrm{mm}
```

for each plus/minus probe direction, with

```text
DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION = 3
```

For each probe move \(i\), the commanded motor displacement is known:

```math
\Delta \mathbf a_i.
```

The image registration gives the measured image displacement:

```math
\Delta \mathbf p_i.
```

The calibration fit uses

```math
\Delta \mathbf p_i \approx J\Delta \mathbf a_i
```

across all valid probe moves.

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

The estimator is a robust iteratively reweighted least-squares regression over probe
residuals. Robust fitting is used because image registration can produce occasional
outliers from low contrast, poor features, illumination changes, partial occlusion, or
other nonideal image effects.

Calibration is rejected if:

- a probe image response is below `DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX`
  (`2.0 px` by default);
- the commanded probe deltas do not span the three command axes;
- the fitted visual Jacobian has rank less than 3;
- the condition number exceeds
  `DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING` (`100.0` by default).

The rank and condition-number checks are observability checks. Rank less than 3 means
that the three command directions cannot be distinguished in image space. A large
condition number means that at least one command direction is only weakly observable, so
small image noise could produce a large inferred motor correction.

### Axis Scale

The three motor axes can have different image sensitivities. A one-millimeter
move along `x` might produce a much larger image shift than a one-millimeter
move along `z`, for example.

To avoid solving the inverse problem in poorly scaled coordinates, correction is
posed in normalized command coordinates. Each calibration dataset stores

```text
axis_scale_cmd_mm(command_axis)
```

and later correction runs reuse this saved scale.

Let

```math
S = \operatorname{diag}(s_x, s_y, s_z),
```

where

```math
\mathbf s =
\begin{bmatrix}
s_x \\
s_y \\
s_z
\end{bmatrix}
```

is represented by `axis_scale_cmd_mm`.

The normalized command increment \(\Delta\mathbf q\) is dimensionless, and the
actual commanded-mm increment is

```math
\Delta \mathbf a = S\Delta \mathbf q.
```

Substituting this into the image-response model gives

```math
\Delta\mathbf p \approx J S\Delta\mathbf q.
```

Define

```math
J_q = JS.
```

Then

```math
\Delta\mathbf p \approx J_q\Delta\mathbf q.
```

The scale is derived from the fitted visual Jacobian. For command axis \(j\),

```math
c_j = \|J_{:,j}\|_2
```

is the image sensitivity of that axis, expressed in px/commanded-mm.

Given the calibration probe magnitude

```math
h_j = \max_i |\Delta a_{i,j}|,
```

the observed per-axis probe response scale is

```math
r_j = c_j h_j.
```

The target response is the median response over axes:

```math
r_\star = \operatorname{median}(r_x, r_y, r_z).
```

The unclamped command scale is

```math
s_j^{\mathrm{raw}} = \frac{r_\star}{c_j}.
```

The saved correction scale is clamped to configured operational bounds:

```math
s_j =
\operatorname{clip}
\left(
s_j^{\mathrm{raw}},
s_{j,\min},
s_{j,\max}
\right).
```

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

### Weighted Image Residual

Correction minimizes the weighted image-space residual

```math
\rho(\mathbf p) =
\sqrt{\mathbf p^\mathsf T W\mathbf p}.
```

When no weights are specified,

```math
W = I_4.
```

In that default case,

```math
\rho(\mathbf p) =
\sqrt{
(\Delta u_{\mathrm{cam0}})^2 +
(\Delta v_{\mathrm{cam0}})^2 +
(\Delta u_{\mathrm{cam1}})^2 +
(\Delta v_{\mathrm{cam1}})^2
}.
```

Weights allow individual image observations to be trusted more or less. For example, a
camera direction with poor registration quality can be downweighted.

Weights, when supplied, must contain four finite nonnegative observation weights with at
least one positive entry. They may also have shape `(camera, pixel_axis)`, in which case
they are flattened to the same observation order as \(\mathbf p\).

Convergence is declared when

```math
\rho(\mathbf p_k) \le \mathrm{tol},
```

where `tol` is `DEFAULT_CORRECTION_PIXEL_TOLERANCE_PX`, with default value

```text
0.5 px
```

This is the only correction residual recorded in the iteration history:

```text
iteration_weighted_residual_px
```

Unweighted RMS residuals are not part of the correction output.

### Damped Normalized Command Solve

At a correction step, the current measured residual is \(\mathbf p\). The linearized
prediction for the residual after a move is

```math
\mathbf p_{\mathrm{new}}
\approx
\mathbf p + J\Delta\mathbf a.
```

To reduce the residual, the controller wants

```math
J\Delta\mathbf a \approx -\mathbf p.
```

In normalized coordinates,

```math
\Delta \mathbf a = S\Delta \mathbf q
```

and

```math
J_q = JS.
```

The controller solves a damped weighted least-squares inverse problem. At correction
iteration $k$, the normalized command increment is

```math
\Delta \mathbf q
=
-\lambda
\left(
J_q^\mathsf T WJ_q + \mu I
\right)^{-1}
J_q^\mathsf T W\mathbf p.
```

The commanded-mm correction is then

```math
\Delta \mathbf a = S\Delta \mathbf q.
```

This is equivalent to choosing a motor move that approximately minimizes

```math
\left\|
\mathbf p + J_q\Delta\mathbf q
\right\|_W^2
+
\mu\left\|\Delta\mathbf q\right\|^2,
```

then applying only a fraction \(\lambda\) of that full correction.

The first term says: reduce the predicted image residual. The second term says: do not
take an excessively large normalized motor step. The damping parameter \(\mu\)
stabilizes the inverse when the Jacobian is noisy or ill-conditioned. The gain
\(\lambda\) makes the correction conservative, which helps when the linear model is only
locally accurate.

The default numerical parameters are:

```text
lambda = DEFAULT_CORRECTION_GAIN = 0.3
lambda_min = DEFAULT_CORRECTION_MIN_GAIN = 0.15
mu = DEFAULT_CORRECTION_DAMPING_MU = 1.0
max |Delta q_j| = DEFAULT_CORRECTION_MAX_NORMALIZED_STEP = 0.5
min per-axis predicted response =
    DEFAULT_CORRECTION_MIN_AXIS_PREDICTED_SHIFT_PX = 0.15 px
min total predicted response =
    DEFAULT_CORRECTION_MIN_TOTAL_PREDICTED_SHIFT_PX = 0.30 px
min feedback alpha = DEFAULT_CORRECTION_MIN_FEEDBACK_ALPHA = 0.25
min feedback parallel response =
    DEFAULT_CORRECTION_MIN_FEEDBACK_PARALLEL_SHIFT_PX = 0.15 px
min command norm = DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM = 1e-9 mm
max_moves = DEFAULT_CORRECTION_MAX_MOVES = 12
```

Before a motor command is issued, safeguards are applied to the least-squares proposal.

First, the full normalized vector is scaled down if

```math
\max_j |\Delta q_j|
>
\texttt{DEFAULT\_CORRECTION\_MAX\_NORMALIZED\_STEP}.
```

Second, individual axis components are suppressed if their predicted weighted
image effect is below the configured threshold:

```math
|\Delta a_j|
\sqrt{J_{:,j}^\mathsf T WJ_{:,j}}
<
\texttt{DEFAULT\_CORRECTION\_MIN\_AXIS\_PREDICTED\_SHIFT\_PX}.
```

Components are also set to zero when the corresponding estimated command offset is at or
below the configured correction deadband. If the remaining command vector is effectively
zero, correction stops before issuing another motor command and reports non-convergence
with a warning.

### Image-Response Observability

For a proposed correction \(\Delta\mathbf a\), the predicted image response is

```math
\widehat{\Delta \mathbf p} =
J\Delta\mathbf a.
```

Before motion, the controller computes its weighted magnitude:

```math
r_{\mathrm{pred}} =
\sqrt{
\widehat{\Delta \mathbf p}^{\mathsf T}
W
\widehat{\Delta \mathbf p}
}.
```

If \(r_{\mathrm{pred}}\) is below `DEFAULT_CORRECTION_MIN_TOTAL_PREDICTED_SHIFT_PX`, the
move is not expected to produce enough image motion to be useful feedback. In that case,
correction stops before issuing the motor command. The controller does not treat a
below-noise image change as evidence about the motor direction or the Jacobian.

After a commanded move and one post-move image capture, the measured image
change is

```math
\Delta \mathbf p_{\mathrm{meas}} =
\mathbf p_{k+1} - \mathbf p_k.
```

The predicted response was \(\widehat{\Delta\mathbf p}\). The controller compares the
measured and predicted responses before deciding whether the feedback is trustworthy.

The scalar innovation gain is

```math
\alpha
=
\frac{
\widehat{\Delta \mathbf p}^{\mathsf T}
W
\Delta \mathbf p_{\mathrm{meas}}
}{
\widehat{\Delta \mathbf p}^{\mathsf T}
W
\widehat{\Delta \mathbf p}
}.
```

This is the weighted least-squares scalar that best maps the predicted image change onto
the measured image change. If the measured response matches the prediction, \(\alpha
\approx 1\). If the measured response is smaller but in the same direction, \(0 < \alpha
< 1\). If the measured response is opposite the prediction, \(\alpha < 0\).

The parallel measured response is

```math
r_{\parallel}
=
\frac{
\widehat{\Delta \mathbf p}^{\mathsf T}
W
\Delta \mathbf p_{\mathrm{meas}}
}{
r_{\mathrm{pred}}
}.
```

This is the measured image motion projected along the predicted image-motion
direction, in weighted pixel units.

Post-move feedback is accepted only when

```text
r_pred >= DEFAULT_CORRECTION_MIN_TOTAL_PREDICTED_SHIFT_PX
alpha >= DEFAULT_CORRECTION_MIN_FEEDBACK_ALPHA
r_parallel >= DEFAULT_CORRECTION_MIN_FEEDBACK_PARALLEL_SHIFT_PX
```

Invalid feedback is recorded in the correction history, but it is not used for
Jacobian refinement. It also does not cause the loop to keep stacking additional
correction moves on top of an unobservable or anti-aligned image response.

The absolute BCS-mm target sent to the motors is

```math
\mathbf a_{\mathrm{request}}
=
\mathbf a_{\mathrm{commanded}} + \Delta \mathbf a.
```

The internal `commanded_position_mm` state is initialized from the BCS `x`, `y`,
and `z` values and is subsequently advanced to each requested absolute target.
Post-move readback is not used as the command-state anchor.

### Correction Loop

The closed-loop correction algorithm proceeds as follows:

1. load and validate a saved calibration dataset;
2. require a real calibration file path, because accepted Jacobian refinements
   are persisted to disk;
3. capture images and compute `shift_px`;
4. form the image residual \(\mathbf p_k\);
5. compute

   ```math
   \rho_k = \sqrt{\mathbf p_k^\mathsf T W\mathbf p_k};
   ```

6. stop immediately if \(\rho_k\) is at or below the pixel tolerance;
7. compute the damped normalized command correction;
8. apply the normalized-step cap and suppress low-impact axis components;
9. stop before motion if the remaining command vector is below the minimum
   command norm or the predicted total image response is below the observable
   feedback threshold;
10. send absolute BCS-mm targets only for axes with nonzero correction
    components;
11. re-image once and compute the new residual plus feedback innovation
    diagnostics;
12. if feedback is valid and the residual decreased, refit the visual Jacobian
    from the calibration probes plus accepted correction observations and save
    the refined calibration dataset to disk;
13. if feedback is valid but the residual increased or stayed flat, skip the
    Jacobian update, halve the gain down to `DEFAULT_CORRECTION_MIN_GAIN`,
    double `mu`, and continue from the newly measured image state;
14. if feedback is invalid, record the move diagnostics and stop unless the new
    residual is already within tolerance;
15. save the correction result into the sibling correction-history file.

No rollback is performed after a residual-increasing move. The image measurement
acquired after that move defines the next closed-loop state.

If convergence is not reached after `max_moves`, correction returns a dataset
with

```text
correction_converged = False
```

and an associated warning. Lack of convergence after motion is therefore
reported as data rather than raised as an exception.

### Jacobian Refinement

Every accepted correction move is also a new measurement of the same local
linear response:

```math
\Delta \mathbf p_i \approx J\Delta \mathbf a_i.
```

After a move, the measured image change is

```math
\Delta \mathbf p_{\mathrm{meas}}
=
\mathbf p_{k+1} - \mathbf p_k.
```

The refined Jacobian is not overwritten from one correction move. Instead, the
software pools:

- the original calibration probe observations;
- all accepted closed-loop correction observations.

It then refits the Jacobian:

```math
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
```

where the first rows are the large calibration probes and later rows are
closed-loop correction moves. The estimate is obtained with the same
Huber-style iteratively reweighted least-squares procedure used during
calibration.

This pooling is intentionally conservative. In the pooled least-squares normal
equation, an observation contributes roughly through

```math
\Delta\mathbf a_i\Delta\mathbf a_i^\mathsf T.
```

A 5 micron correction therefore has approximately

```math
\left(\frac{0.005}{0.5}\right)^2 = 10^{-4}
```

the Jacobian leverage of a 0.5 mm calibration probe. This prevents a single
small correction move from dominating a Jacobian column.

Refinement is only attempted when the post-move feedback is valid and the
weighted image residual decreased. If the refit is rank-deficient, poorly
conditioned, or otherwise invalid, it is skipped and the warning is recorded.

Accepted refinements update:

```text
visual_jacobian_px_per_cmd_mm
jacobian_refinement_count
jacobian_refined = "true"
```

The refined dataset is validated and saved back to the calibration file path.

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
move_predicted_weighted_response_px(move)
move_measured_weighted_response_px(move)
move_feedback_alpha(move)
move_feedback_parallel_px(move)
move_feedback_valid(move)
move_visual_jacobian_before_px_per_cmd_mm(move, camera, pixel_axis, command_axis)
move_visual_jacobian_after_px_per_cmd_mm(move, camera, pixel_axis, command_axis)
move_gain(move)
move_damping_mu(move)
move_max_normalized_component(move)
move_active_axis_mask(move, command_axis)
move_jacobian_refined(move)
```

The correction result keeps per-move diagnostic state because that dataset is
also used as the on-disk correction log.

For a calibration file such as

```text
calibration.h5
```

corrections are saved next to it in

```text
calibration_corrections.h5
```

Each correction run is written to an HDF5 group named

```text
run_000000
run_000001
...
```

During an active correction, the active run group is rewritten after every
completed motor move. Thus the file contains the latest available residual
trace, move commands, measured image response, and Jacobian before/after any
accepted pooled least-squares refinement.

At result assembly, the reported command offset is computed from the final image
residual as

```math
\widehat{\Delta \mathbf a}
=
\left(J^\mathsf T WJ\right)^+
J^\mathsf T W\mathbf p,
```

where \(+\) denotes the pseudoinverse.

This quantity is stored as

```text
estimated_command_offset_mm
```

It answers: under the current Jacobian, what commanded-mm offset would explain
the final image residual?

The reported

```text
correction_cmd_mm
```

is zero when the loop has converged. Otherwise, it is the next damped correction
proposal computed with the final gain and damping values.

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

Calibration can be initiated from before/after commanded-mm probe moves:

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

Saved calibration attributes include:

- `warnings`;
- initial motor context, such as `initial_x_mm`, `initial_y_mm`,
  `initial_z_mm`, `polar`, and `tilt`;
- GUI ROI bounds, such as `roi_cam0_*` and `roi_cam1_*`, when created from the
  GUI.

Accepted closed-loop pooled least-squares refinements rewrite the calibration
file so the refined Jacobian persists across correction runs.

## Tests

```bash
uv run pytest -q
```

## References

[1] S. Duan, S. Wang, Y. Yang, C. Huang, L. Gu, H. Liu, and W. Zhang,
"A sample-position-autocorrection system with precision better than
$1\,\mu\mathrm{m}$ in angle-resolved photoemission experiments,"
*Review of Scientific Instruments* **93**, 103905, 2022.
<https://doi.org/10.1063/5.0106299>

[2] F. Chaumette and S. Hutchinson, "Visual Servo Control, Part I: Basic
Approaches," *IEEE Robotics & Automation Magazine* **13**(4), 82--90, 2006.
<https://doi.org/10.1109/MRA.2006.250573>

[3] M. Jagersand, O. Fuentes, and R. Nelson, "Experimental Evaluation of
Uncalibrated Visual Servoing for Precision Manipulation," *Proceedings of the
IEEE International Conference on Robotics and Automation*, 1997, pp. 2874--2880.
<https://doi.org/10.1109/ROBOT.1997.606723>

[4] K. Levenberg, "A Method for the Solution of Certain Non-Linear Problems in
Least Squares," *Quarterly of Applied Mathematics* **2**, 164--168, 1944.
<https://doi.org/10.1090/qam/10666>; D. W. Marquardt, "An Algorithm for
Least-Squares Estimation of Nonlinear Parameters," *Journal of the Society for
Industrial and Applied Mathematics* **11**(2), 431--441, 1963.
<https://doi.org/10.1137/0111030>

[5] S. Chiaverini, B. Siciliano, and O. Egeland, "Review of the Damped
Least-Squares Inverse Kinematics with Experiments on an Industrial Robot
Manipulator," *IEEE Transactions on Control Systems Technology* **2**(2),
123--134, 1994. <https://doi.org/10.1109/87.294335>

[6] P. J. Huber, "Robust Estimation of a Location Parameter," *The Annals of
Mathematical Statistics* **35**(1), 73--101, 1964.
<https://doi.org/10.1214/aoms/1177703732>

[7] J. Music, M. Bonkovic, and M. Cecic, "Comparison of Uncalibrated Model-Free
Visual Servoing Methods for Small-Amplitude Movements: A Simulation Study,"
*International Journal of Advanced Robotic Systems* **11**, 2014.
<https://doi.org/10.5772/58822>

[8] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
2006. <https://doi.org/10.1007/978-0-387-40065-5>
