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

# Apply MOTOR_BACKLASH_CORRECTION to correction moves when Python is moving
# motors directly through the BCS API. Delegated LabVIEW moves keep their own
# motor behavior.
CORRECTION_USE_BCS_API_BACKLASH: bool = False

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

# Default number of image pairs captured at each measurement position.
DEFAULT_CALIBRATION_CAPTURE_COUNT: int = 3
DEFAULT_CORRECTION_CAPTURE_COUNT: int = 3

# Default number of worker threads for calibration fitting. 1 works best on the DAQ PC.
CALIBRATION_FIT_N_JOBS: int = 1


# Default backlash-aware calibration trajectory controls.
DEFAULT_VISUAL_CALIBRATION_N: int = 5
DEFAULT_VISUAL_CALIBRATION_STEP_UM: float = 50.0

# Bounds used when deriving correction command normalization scales.
DEFAULT_AXIS_SCALE_BOUNDS_CMD_MM_BY_AXIS: dict[str, tuple[float, float]] = {
    "x": (0.1, 0.8),
    "y": (0.3, 1.0),
    "z": (0.1, 0.8),
}

DEFAULT_VISUAL_CALIBRATION_REPEATABILITY_WARNING_FRACTION: float = 0.10

# Minimum two-camera image response accepted for a calibration probe.
DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX: float = 0.1

# Calibration condition number threshold above which the fit is rejected.
DEFAULT_JACOBIAN_CONDITION_WARNING: float = 50.0

# Shared closed-loop LQR correction defaults in command-mm visual-servo space.
# Observation weights are ordered as cam0_du, cam0_dv, cam1_du, cam1_dv.
CORRECTION_OBSERVATION_WEIGHTS: tuple[float, float, float, float] | None = (
    1.0,
    1.0,
    1.0,
    1.0,
)
DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM: float = 1e-9
DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM: float = 0.1
DEFAULT_CORRECTION_MAX_MOVES: int = 20
DEFAULT_CORRECTION_MODE: str = "beam"

# Beam-frame correction geometry. At polar=0, the beam is +65 degrees from +z
# toward +x in the sample xz plane. Beam-frame tolerances are correction
# residual scales, in microns, for transverse-in-xz and vertical-y offsets.
DEFAULT_BEAM_XZ_ANGLE_FROM_ANALYZER_DEG: float = 65.0
DEFAULT_BEAM_TRANSVERSE_TOLERANCE_UM: float = 3.0
DEFAULT_BEAM_ANALYZER_TRANSVERSE_TOLERANCE_UM: float = 3.0
DEFAULT_BEAM_VERTICAL_TOLERANCE_UM: float = 3.0

# LQR solver weights and numerical tolerance.
DEFAULT_LQR_CORRECTION_GAIN: float = 0.50
DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP: float = 0.25
DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX: float = 0.1
DEFAULT_LQR_CORRECTION_MOTOR_PENALTY: float = 25.0
DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE: float = 1e-6
DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE: float = 1.0

# LQR-only Kalman observer. Measurement covariance is in raw pixel units and is
# ordered as cam0_du, cam0_dv, cam1_du, cam1_dv. The correction code normalizes
# it by the LQR image scale before updating/gating.
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
) = (
    (7.768352e-04, -2.612795e-04, -5.777776e-04, -1.457239e-04),
    (-2.612795e-04, 4.261280e-03, 4.355556e-03, -1.190572e-03),
    (-5.777776e-04, 4.355556e-03, 8.296296e-03, -1.940741e-03),
    (-1.457239e-04, -1.190572e-03, -1.940741e-03, 1.724983e-03),
)
DEFAULT_LQR_CORRECTION_KALMAN_INITIAL_COVARIANCE: float = 1.0
DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE: float = 25.0

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

# Whether to move motors from Python using the BCS API or by sending commands to the
# LabVIEW motor server.
MOTOR_SERVER_USE_BCS_API_BACKEND: bool = True

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
