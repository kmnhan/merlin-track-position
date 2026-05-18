"""Defines port numbers for ZMQ communication and other constants."""

import sys
import pathlib

# Mapping from shorthand motor names to actual motor names at the beamline.
MOTOR_NAMES: dict[str, str] = {
    "x": "Sample X",
    "y": "Sample Y (Vert)",
    "z": "Sample Z",
    "p": "Polar",
    "t": "Tilt",
    "cam": "Video Switch",
    "TA": "Cryostat Temp A",
    "TB": "Cryostat Temp B",
    "TC": "Cryostat Temp C",
    "TD": "Cryostat Temp D",
}

# Manual backlash correction for motors that have no built-in correction in LabVIEW.
# Values are in each motor's command units. For x and z, this is mm.
MOTOR_BACKLASH_CORRECTION: dict[str, float] = {
    "x": 0.1,
    "z": 0.1,
}

# Stale-status readback fallback for motor moves. Values are keyed by motor alias
# and are in each motor's command units. For x/y/z this is mm.
MOTOR_STALE_READBACK_DEADBAND: dict[str, float] = {
    "x": 0.01,
    "y": 0.1,
    "z": 0.01,
    "p": 0.05,
    "t": 0.05,
    "cam": 0.001,
    "TA": 0.001,
    "TB": 0.001,
    "TC": 0.001,
    "TD": 0.001,
}
MOTOR_STALE_READBACK_DELAY_S: float = 10.0

# Damped-WLS command pruning thresholds. Values are in commanded mm and are
# applied to estimated command offsets before sending gain-scaled correction
# components. The LQR correction path bypasses these deadbands.
DAMPED_WLS_CORRECTION_COMMAND_DEADBAND_MM_BY_AXIS: dict[str, float] = {
    "x": 0.001,
    "y": 0.001,
    "z": 0.001,
}

# Default number of image pairs captured at each calibration/correction position.
DEFAULT_CAPTURE_COUNT: int = 3

# Default number of worker threads for calibration fitting. 1 works best on the DAQ PC.
CALIBRATION_FIT_N_JOBS: int = 1


# Default commanded-mm probe steps for calibration.
DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS: dict[str, float] = {
    "x": 0.2,
    "y": 0.2,
    "z": 0.2,
}

# Bounds used when deriving correction damping scales from the fitted Jacobian.
DEFAULT_AXIS_SCALE_BOUNDS_CMD_MM_BY_AXIS: dict[str, tuple[float, float]] = {
    "x": (0.1, 0.8),
    "y": (0.3, 1.0),
    "z": (0.1, 0.8),
}

# Number of repeated +axis/-axis probes per command axis.
DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION: int = 3

# Minimum two-camera image response accepted for a calibration probe.
DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX: float = 2.0

# Calibration condition number threshold above which the fit is rejected.
DEFAULT_JACOBIAN_CONDITION_WARNING: float = 100.0

# Shared closed-loop correction defaults in command-mm visual-servo space.
# Supported algorithms: "damped_wls" and "lqr".
CORRECTION_ALGORITHM: str = "lqr"
# Observation weights are ordered as cam0_du, cam0_dv, cam1_du, cam1_dv.
CORRECTION_OBSERVATION_WEIGHTS: tuple[float, float, float, float] | None = (
    0.80,
    1.33,
    1.21,
    0.66,
)
DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM: float = 1e-9
DEFAULT_CORRECTION_MAX_MOVES: int = 12

# Damped-WLS solver and legacy closed-loop guardrails. LQR still records
# feedback diagnostics, but it does not use these thresholds to prune/stop.
DEFAULT_DAMPED_WLS_CORRECTION_PIXEL_TOLERANCE_PX: float = 0.55
DEFAULT_DAMPED_WLS_CORRECTION_GAIN: float = 0.6
DEFAULT_DAMPED_WLS_CORRECTION_MIN_GAIN: float = 0.15
DEFAULT_DAMPED_WLS_CORRECTION_MAX_NORMALIZED_STEP: float = 0.5
DEFAULT_DAMPED_WLS_CORRECTION_DAMPING_MU: float = 1.0
DEFAULT_DAMPED_WLS_CORRECTION_MIN_AXIS_PREDICTED_SHIFT_PX: float = 0.04
DEFAULT_DAMPED_WLS_CORRECTION_MIN_TOTAL_PREDICTED_SHIFT_PX: float = 0.1
DEFAULT_DAMPED_WLS_CORRECTION_MIN_FEEDBACK_ALPHA: float = 0.25
DEFAULT_DAMPED_WLS_CORRECTION_MIN_FEEDBACK_PARALLEL_SHIFT_PX: float = 0.15

# LQR-only solver weights and numerical tolerance.
DEFAULT_LQR_CORRECTION_GAIN: float = 0.95
DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP: float = 0.5
DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX: float = 0.1
DEFAULT_LQR_CORRECTION_MOTOR_PENALTY: float = 100.0
DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE: float = 1e-6
DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE: float = 2.0

# LQR-only Kalman observer. Disabled by default so the nominal LQR command law
# is unchanged unless this is explicitly enabled.
DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER: bool = False
DEFAULT_LQR_CORRECTION_KALMAN_PROCESS_NOISE: float = 0.05
DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_NOISE: float = 1.0
DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_COVARIANCE: (
    tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    | None
) = None
DEFAULT_LQR_CORRECTION_KALMAN_INITIAL_COVARIANCE: float = 100.0
DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE: float = 16.0

# Image size for initial crop from each camera array.
IMAGE_WIDTH_CAM0: int = 704
IMAGE_HEIGHT_CAM0: int = 480
IMAGE_WIDTH_CAM1: int = 1440
IMAGE_HEIGHT_CAM1: int = 1080

# If you change this, make sure to also update FrameGrabbber FSM UI2.vi
FRAMEGRAB_SERVER_PORT: int = 6553

# Change this if you change the Basler camera
BASLER_CAMERA_SERIAL = "40055360"
BASLER_EXPOSURE: int = 300000  # exposure time


# If you change this, make sure to also update TrackTemperatureBL403 BCS Driver.vi
MOTOR_SERVER_PORT: int = 6554

# These are default settings for BCS API server, probably shouldn't need to be changed.
BCS_SERVER_HOST: str = "localhost"
BCS_SERVER_PORT: int = 5577
BCS_REQUEST_TIMEOUT_MS: int = 30_000
# When CPU intensive tasks like scheduled disk backup runs on the DAQ PC, BCS API
# responses can be very slow. Set a long timeout to avoid spurious failures.

# Flag to indicate whether we're running on the acquisition PC (True) or a development
# machine (False). This can be used to determine file paths and other
# environment-specific settings.
IS_DAQ_PC: bool = sys.platform == "win32"

if IS_DAQ_PC:
    # Assume we're on the acquisition PC, support files are on local.
    SUPPORT_FILE_BASE = pathlib.Path(
        r"C:\Beamline Controls\4.0.3 Arpes Endstation\4.0.3 Arpes Endstation Specific\Support Files"
    )
else:
    # Debugging on dev mac, assume mounted network drive to access support files.
    SUPPORT_FILE_BASE = pathlib.Path(
        "/Volumes/Beamline Controls/4.0.3 Arpes Endstation/4.0.3 Arpes Endstation Specific/Support Files"
    )

INSTR_SCAN_SETUP_PATH = SUPPORT_FILE_BASE / "Instrument Scan Setup.txt"
