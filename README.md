# Sample-Position Correction System for Beamline 4.0.3 MERLIN at the Advanced Light Source

This repository implements image-based sample-position tracking and correction
for Beamline 4.0.3 MERLIN at the Advanced Light Source. The controller uses a
locally calibrated visual Jacobian between commanded motor displacements and
two-camera image displacements, then applies a closed-loop Linear Quadratic
Regulator (LQR) in the controllable image subspace.

The imaging system consists of the existing sample-view camera exposed through
the FrameGrabber window and a Basler acA1440-73gm camera viewing the same
viewport from a different angle. The two views have distinct pixel scales,
fields of view, and image content, so correction is formulated in a joint
four-channel image space rather than in either camera coordinate system alone.

The correction loop is local. It is intended for small displacements around the
operator point where the calibrated Jacobian is valid. If the sample is outside
that local region, the controller should abort or hand off to coarse alignment
before applying feedback.

## Method

### Variables

At correction iteration `k`, the two-camera image measurement is

```math
e_k^{\mathrm{meas}} =
\begin{bmatrix}
\Delta u_{\mathrm{cam0},k} \\
\Delta v_{\mathrm{cam0},k} \\
\Delta u_{\mathrm{cam1},k} \\
\Delta v_{\mathrm{cam1},k}
\end{bmatrix}
\in \mathbb{R}^4.
```

Each component is a sub-pixel shift between the current image and the stored
reference image. The reference images are captured at the desired operator
point, so zero image error means the sample is aligned to that point in the two
views.

The commanded motor increment is

```math
\Delta a_k =
\begin{bmatrix}
\Delta a_{x,k} \\
\Delta a_{y,k} \\
\Delta a_{z,k}
\end{bmatrix},
```

where each component is a commanded BCS motor displacement in millimeters. The
local command-to-image response is

```math
\Delta e_k \approx J\Delta a_k,
\quad
J \in \mathbb{R}^{4\times3}.
```

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

The Jacobian entries have units of pixels per commanded millimeter. The
calibration fit uses commanded moves, not encoder readback. Readback is retained
as diagnostic metadata because backlash, hysteresis, and stale status can make
readback a poor representation of the command state used by the beamline
software.

### Reference Images and Sign Convention

At the operator point, store reference images

```math
I_0^\star,\quad I_1^\star.
```

At correction iteration `k`, acquire current images

```math
I_{0,k},\quad I_{1,k}.
```

For each camera, compute the sub-pixel shift against the reference image:

```math
(\Delta u_{i,k}, \Delta v_{i,k})
= \text{CrossCorrShift}(I_{i,k}, I_i^\star),
\quad i \in \{0,1\}.
```

The implemented sign convention must satisfy

```math
e_{k+1}^{\mathrm{meas}} - e_k^{\mathrm{meas}}
\approx
J\Delta a_k
```

for small test commands near the operator point. Before enabling feedback,
measure `e_k`, apply a small known command, remeasure, and verify that the
observed image change has the same sign and approximate magnitude as
`J @ delta_a`.

### Calibration Routine

Before correction, the visual Jacobian `J` is estimated from finite-difference
probe moves:

1. read the initial BCS `x`, `y`, and `z` command positions;
2. acquire `reference_cam0` and `reference_cam1`;
3. generate repeated positive and negative single-axis probes for `x`, `y`, and
   `z`;
4. for each probe, acquire pre-move images, command the move, acquire post-move
   images, and register the post-move images against the pre-move images;
5. estimate `visual_jacobian_px_per_cmd_mm` from the valid
   `(probe_command_delta_mm, probe_measured_delta_px)` rows;
6. persist the calibration dataset to disk and reload it from that path.

The default probe magnitudes are

```text
x: 0.3 commanded-mm
y: 0.5 commanded-mm
z: 0.3 commanded-mm
```

with

```text
DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION = 3
```

For probe `i`, the commanded motor displacement is `Delta a_i` and image
registration gives `Delta e_i`. The calibration fit identifies the local model

```math
\Delta e_i = J\Delta a_i + r_i,
```

where `r_i` collects image-registration noise, backlash, finite settling error,
local nonlinearity, and other residual effects. The nominal LQR design uses the
fitted `J`; the residuals are not assumed to be zero at runtime.

Calibration is rejected if:

- a probe image response is below `DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX`
  (`2.0 px` by default);
- the commanded probe deltas do not span the three command axes;
- the fitted visual Jacobian has rank less than 3;
- the condition number exceeds
  `DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING` (`100.0` by default).

The rank and condition-number checks are observability checks. Rank less than 3
means the three command directions cannot be distinguished in image space. A
large condition number means at least one command direction is weakly observable,
so image noise could produce a large inferred correction.

### Motor-Axis Scale

The three motor axes can have different image sensitivities. A one-millimeter
move along `x` may produce a much larger image shift than a one-millimeter move
along `z`.

The LQR design therefore uses normalized command coordinates. Each calibration
dataset stores

```text
axis_scale_cmd_mm(command_axis)
```

and correction reuses that saved scale. Let

```math
S_m = \text{diag}(s_x, s_y, s_z),
```

where

```math
s =
\begin{bmatrix}
s_x \\
s_y \\
s_z
\end{bmatrix}
```

is represented by `axis_scale_cmd_mm`. The normalized command is

```math
\tilde a_k = S_m^{-1}\Delta a_k,
\quad
\Delta a_k = S_m\tilde a_k.
```

The scale is derived from the fitted visual Jacobian. For command axis `j`,

```math
c_j = \|J_{:,j}\|_2
```

is the image sensitivity in pixels per commanded millimeter. Given the probe
magnitude

```math
h_j = \max_i |\Delta a_{i,j}|,
```

the observed probe response scale is

```math
r_j = c_jh_j.
```

The target response is the median over axes:

```math
r_\star = \text{median}(r_x,r_y,r_z).
```

The raw command scale is

```math
s_j^{\mathrm{raw}} = \frac{r_\star}{c_j}.
```

The saved scale is clamped to configured operational bounds:

```math
s_j = \text{clip}
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

## LQR Design

### Normalized Image Model

Image channels are also normalized. Let

```math
S_e = \text{diag}(e_{\mathrm{scale}}),
```

where the default scalar image scale is

```text
DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX = 0.1
```

Observation weights, when supplied, are folded into the image scale as

```math
e_{\mathrm{scale},i} =
\frac{\texttt{image\_scale\_px}}{\sqrt{w_i}}.
```

The normalized image measurement and normalized Jacobian are

```math
\tilde e_k^{\mathrm{meas}} = S_e^{-1}e_k^{\mathrm{meas}},
\quad
J_n = S_e^{-1}JS_m.
```

The nominal normalized update is

```math
\tilde e_{k+1} = \tilde e_k + J_n\tilde a_k.
```

This model is used to design the LQR gain. The actual runtime system may also
include backlash, drift, measurement noise, finite settling error, and local
model mismatch; closed-loop correction handles those effects by remeasuring
after every move.

### Controllable Image Subspace

The two-camera image error has four components, while the manipulator command
has three axes. Depending on the local camera geometry, not every image-space
direction is controllable. The controller therefore uses the controllable
subspace of the normalized Jacobian.

Compute

```math
J_n = U\Sigma V^\mathsf{T}.
```

The numerical rank is

```math
r_c = \#\{\sigma_i : \sigma_i >
\epsilon_{\mathrm{svd}}\sigma_1\},
```

with default

```text
DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE = 1e-6
```

Let

```math
U_c = U_{:,1:r_c}\in\mathbb{R}^{4\times r_c}
```

be the controllable image-subspace basis. The projected normalized state is

```math
s_k = U_c^\mathsf{T}\tilde e_k.
```

The LQR design model is

```math
s_{k+1} = A_ss_k + B_s\tilde a_k,
\quad
A_s = I_{r_c},
\quad
B_s = U_c^\mathsf{T}J_n.
```

### Riccati Design

The infinite-horizon discrete-time LQR cost is

```math
\mathcal{J} =
\sum_{k=0}^{\infty}
\left(
s_k^\mathsf{T}Q_ss_k
+
\tilde a_k^\mathsf{T}R_s\tilde a_k
\right).
```

The implementation uses

```math
Q_s = I_{r_c},
\quad
R_s = \lambda_m I_3.
```

The default motor penalty is

```text
DEFAULT_LQR_CORRECTION_MOTOR_PENALTY = 100.0
```

Larger `lambda_m` produces smaller, slower commands. Smaller `lambda_m` makes
the controller more aggressive.

The discrete algebraic Riccati equation is

```math
P = A_s^\mathsf{T}PA_s
- A_s^\mathsf{T}PB_s
(B_s^\mathsf{T}PB_s + R_s)^{-1}
B_s^\mathsf{T}PA_s
+ Q_s.
```

The feedback gain is

```math
K_s =
(B_s^\mathsf{T}PB_s + R_s)^{-1}
B_s^\mathsf{T}PA_s.
```

The closed-loop poles are checked from

```math
\lambda_i(A_s - B_sK_s),
```

and all must lie inside the unit circle.

### Command Law

Without a Kalman observer, the measured projected state is used directly:

```math
s_k^{\mathrm{meas}} =
U_c^\mathsf{T}S_e^{-1}e_k^{\mathrm{meas}}.
```

The normalized command is

```math
\tilde a_k =
-\alpha_{\mathrm{fb}}K_ss_k^{\mathrm{meas}}.
```

The commanded-mm move is

```math
\Delta a_k =
S_m\tilde a_k =
-\alpha_{\mathrm{fb}}S_mK_sU_c^\mathsf{T}S_e^{-1}e_k^{\mathrm{meas}}.
```

The default feedback multiplier is

```text
DEFAULT_LQR_CORRECTION_GAIN = 0.95
```

Before motion, the normalized command is capped:

```text
DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP = 0.5
```

The cap limits `max(abs(tilde_a_k))`. The final hardware request is an absolute
BCS target:

```math
a_{\mathrm{request}} =
a_{\mathrm{commanded}} + \Delta a_k.
```

The internal `commanded_position_mm` state is initialized from the BCS `x`, `y`,
and `z` command positions and is advanced to each requested absolute target.
Post-move readback is logged, but it is not used as the command-state anchor.

### Stopping Criterion

LQR convergence is evaluated in the controllable normalized image subspace:

```math
\|s_k\|_2 \le \varepsilon_s.
```

The default tolerance is

```text
DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE = 1.0
```

This means the controllable image error is within roughly one normalized image
tolerance unit. The pixel residual

```math
\sqrt{e_k^\mathsf{T}We_k}
```

is still recorded as a diagnostic, but the LQR stopping decision uses the
projected normalized error.

## Kalman Observer

Camera registration can be noisy. The measured image error should therefore be
distinguished from the true local image-space state:

```math
e_k^{\mathrm{meas}} \ne e_k^{\mathrm{true}}.
```

The optional Kalman observer treats the LQR controllable-subspace state as the
true state and treats the four camera channels as noisy measurements of that
state. This gives the controller a local LQG form: LQR supplies the feedback
gain and the Kalman filter supplies the state estimate.

### State and Measurement Models

Let

```math
x_k := s_k^{\mathrm{true}}\in\mathbb{R}^{r_c},
\quad
u_k := \tilde a_k^{\mathrm{applied}}.
```

The process model is

```math
x_{k+1} = x_k + B_su_k + w_k,
\quad
w_k \sim (0,Q).
```

The normalized full camera measurement is

```math
z_k = S_e^{-1}e_k^{\mathrm{meas}}\in\mathbb{R}^4.
```

The measurement model is

```math
z_k = U_cx_k + v_k,
\quad
v_k \sim (0,R).
```

Using the full four-channel measurement lets the observer account for
camera-direction covariance. If one image direction is noisy, or if two camera
directions move together because of optical or registration effects, that
information belongs in the full covariance matrix `R`.

With the observer enabled, the command law becomes

```math
\tilde a_k =
-\alpha_{\mathrm{fb}}K_s\hat x_k,
\quad
\Delta a_k =
-\alpha_{\mathrm{fb}}S_mK_s\hat x_k.
```

### Filter Equations

Prediction:

```math
\hat x_k^- =
\hat x_{k-1} + B_su_{k-1},
\quad
P_k^- = P_{k-1} + Q.
```

Innovation:

```math
\nu_k = z_k - U_c\hat x_k^-,
\quad
S_k = U_cP_k^-U_c^\mathsf{T} + R.
```

Kalman gain:

```math
L_k = P_k^-U_c^\mathsf{T}S_k^{-1}.
```

Update:

```math
\hat x_k = \hat x_k^- + L_k\nu_k,
\quad
P_k =
(I - L_kU_c)P_k^-(I - L_kU_c)^\mathsf{T}
+ L_kRL_k^\mathsf{T}.
```

The implementation also computes the normalized innovation distance

```math
\gamma_k = \nu_k^\mathsf{T}S_k^{-1}\nu_k.
```

If `gamma_k` exceeds the innovation gate, the measurement is rejected and the
predicted state is used. The default gate is

```text
DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE = 16.0
```

This catches correlation failures, images outside the local linear regime, and
measurements that are inconsistent with the calibrated model and covariance.

### Estimating Measurement Covariance

The most important Kalman parameter is the measurement covariance `R`.

With the motors fixed, temperature stable, and the sample near the reference
point, acquire `N` image measurements:

```math
e_i^{\mathrm{meas}} =
[\Delta u_{0,i}, \Delta v_{0,i},
 \Delta u_{1,i}, \Delta v_{1,i}]^\mathsf{T}.
```

Compute the raw covariance

```math
R_e =
\frac{1}{N-1}
\sum_i
(e_i - \bar e)(e_i - \bar e)^\mathsf{T}.
```

Then normalize it:

```math
R = S_e^{-1}R_eS_e^{-1}.
```

Use the full `4 x 4` covariance. Do not force `R` to be diagonal when the
camera channels are correlated. Off-diagonal terms such as
`cov(du_cam0, dv_cam0)` or `cov(du_cam0, du_cam1)` tell the filter which
directions should be trusted together or discounted together.

If no full covariance is supplied, the implementation starts from a scalar
measurement-noise level:

```text
DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_NOISE = 1.0
```

### Choosing Process Noise

The process covariance `Q` describes how much the true state can differ from the
prediction based on the applied motor command and local model. It represents
real state evolution, not camera readout noise:

- thermal drift during the correction interval;
- backlash or incomplete motion;
- local Jacobian mismatch;
- finite settling error;
- unmodeled mechanical coupling.

If motor tracking error is small and camera measurement noise dominates, start
with `Q << R`. A small `Q` smooths noisy camera measurements but follows real
drift more slowly. A larger `Q` follows measurements more quickly but filters
less.

The default scalar process-noise level is

```text
DEFAULT_LQR_CORRECTION_KALMAN_PROCESS_NOISE = 0.05
```

For a first implementation, `Q = qI` is usually enough. Later, residual logs can
be used to estimate a subspace covariance. A rough image-space residual is

```math
\delta z_k =
z_{k+1} - z_k - J_n\tilde a_k.
```

Because this includes measurement noise twice,

```math
\text{cov}(\delta z_k) \approx Q_{\mathrm{image}} + 2R.
```

A rough starting estimate is

```math
Q_s \approx
U_c^\mathsf{T}
\left(\text{cov}(\delta z_k)-2R\right)
U_c,
```

with negative eigenvalues clipped to zero.

## Runtime Correction Loop

The closed-loop LQR correction proceeds as follows:

1. load and validate a saved calibration dataset;
2. compute the normalized LQR design from `J`, `axis_scale_cmd_mm`,
   `image_scale_px`, `motor_penalty`, and observation weights;
3. acquire current images and compute `shift_px`;
4. form the normalized projected state, or initialize/update the Kalman state
   estimate if the observer is enabled;
5. stop immediately if the projected normalized error is within tolerance;
6. compute the LQR command in commanded-mm units;
7. cap the normalized command and stop if the remaining command is below the
   minimum command norm;
8. send absolute BCS-mm targets only for axes with nonzero correction;
9. wait for the motor move to return and acquire post-move images;
10. log the measured image change, nominal predicted change, and model
    residual;
11. update the Kalman observer from the post-move measurement when enabled;
12. repeat from the newly measured state until convergence or `max_moves`.

No rollback is performed after a residual-increasing move. The image
measurement acquired after that move defines the next closed-loop state.

If convergence is not reached after

```text
DEFAULT_CORRECTION_MAX_MOVES = 12
```

the correction result is returned with

```text
correction_converged = False
```

and a warning. Lack of convergence is reported as data rather than raised as an
exception.

### Runtime Residuals

After each move, the software logs

```math
r_k =
e_{k+1}^{\mathrm{meas}}
- e_k^{\mathrm{meas}}
- J\Delta a_k^{\mathrm{applied}}.
```

This residual is useful, but it must be interpreted carefully when camera noise
is significant:

```math
r_k^{\mathrm{meas}}
=
\text{true process residual}
+
(v_{k+1}-v_k).
```

Large residual variance can therefore come from camera measurement noise, not
only from motor error or backlash. When the Kalman observer is enabled, also
inspect the filtered subspace residual

```math
r_k^{\mathrm{KF}} =
\hat x_{k+1} - \hat x_k - B_s\tilde a_k.
```

This is not ground truth, but it is less contaminated by frame-to-frame camera
noise.

### Safety Conditions

The correction loop should stop or abort when:

- cross-correlation confidence is too low;
- the current image is outside the field of view;
- the image error is outside the local linear region of the Jacobian;
- the proposed motor command would violate position bounds;
- the command rounds or clips to an ineffective move;
- Kalman innovation gating rejects a measurement;
- closed-loop error increases repeatedly.

Do not declare success from the nominal prediction alone. Success is based on
measured image error or the corresponding Kalman state estimate after images are
acquired.

## Dataset Schema

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
and condition number. `calibration_path` is attached only to loaded in-memory
datasets.

Correction outputs are expressed in commanded-mm units:

```text
estimated_command_offset_mm(command_axis)
correction_cmd_mm(command_axis)
axis_scale_cmd_mm(command_axis)
initial_commanded_position_mm(command_axis)
final_commanded_position_mm(command_axis)
visual_jacobian_px_per_cmd_mm(camera, pixel_axis, command_axis)
shift_px(camera, pixel_axis)
iteration_shift_px(iteration, camera, pixel_axis)
iteration_weighted_residual_px(iteration)
iteration_correction_criterion_residual(iteration)
move_command_delta_mm(move, command_axis)
move_requested_position_mm(move, command_axis)
move_final_readback_position_mm(move, command_axis)
move_pre_weighted_residual_px(move)
move_post_weighted_residual_px(move)
move_predicted_delta_px(move, camera, pixel_axis)
move_measured_delta_px(move, camera, pixel_axis)
move_model_residual_delta_px(move, camera, pixel_axis)
move_predicted_weighted_response_px(move)
move_measured_weighted_response_px(move)
move_feedback_alpha(move)
move_feedback_parallel_px(move)
move_feedback_valid(move)
move_max_normalized_component(move)
move_active_axis_mask(move, command_axis)
```

When the Kalman observer is enabled, correction outputs also include:

```text
iteration_lqr_kalman_state(iteration, lqr_state)
iteration_lqr_kalman_predicted_state(iteration, lqr_state)
iteration_lqr_kalman_innovation(iteration, camera, pixel_axis)
iteration_lqr_kalman_innovation_mahalanobis(iteration)
iteration_lqr_kalman_measurement_accepted(iteration)
```

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

During active correction, the run group is rewritten after every completed
motor move. Thus the file contains the latest available residual trace, motor
commands, measured image response, feedback diagnostics, and Kalman diagnostics
when enabled.

At result assembly, the reported command offset is computed from the final image
residual and the current Jacobian as a diagnostic estimate:

```text
estimated_command_offset_mm
```

The reported

```text
correction_cmd_mm
```

is zero when the loop has converged. Otherwise, it is the next LQR correction
proposal from the final state.

## Programmatic Use

```python
from merlin_track_position.tracking.correct import do_correction

result = do_correction("visual_jacobian_calibration.h5")

print(result["shift_px"].values)  # shape: (camera, pixel_axis)
print(result["estimated_command_offset_mm"].values)  # [x_mm, y_mm, z_mm]
print(result["correction_cmd_mm"].values)
print(result.attrs["correction_criterion"])
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

The calibration and control model is:

```text
[du_cam0, dv_cam0, du_cam1, dv_cam1] = J @ [dx_cmd_mm, dy_cmd_mm, dz_cmd_mm]
Jn = inv(Se) @ J @ Sm
s = Uc.T @ inv(Se) @ measured_pixel_shift
correction_cmd_mm = -gain * Sm @ Ks @ s
```

With the Kalman observer enabled, replace the measured projected state `s` with
the filtered state estimate:

```text
correction_cmd_mm = -gain * Sm @ Ks @ x_hat
```

## Xarray and HDF5

Calibration results are xarray datasets. The main dataset variables are:

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

## Tests

```bash
uv run pytest -q
```

## References

[1] S. Duan, S. Wang, Y. Yang, C. Huang, L. Gu, H. Liu, and W. Zhang,
"A sample-position-autocorrection system with precision better than 1μm in angle-resolved photoemission experiments,"
*Review of Scientific Instruments* **93**, 103905, 2022.
<https://doi.org/10.1063/5.0106299>
