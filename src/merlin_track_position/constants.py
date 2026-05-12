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

# Default no-op deadbands in each motor's command units. For x/y/z, this is mm.
MOTOR_MOVE_DEADBAND: dict[str, float] = {
    "x": 0.001,
    "y": 0.001,
    "z": 0.001,
}

# Default number of image pairs captured at each calibration/correction position.
DEFAULT_CAPTURE_COUNT: int = 3

# Default number of worker threads for calibration fitting.
CALIBRATION_FIT_N_JOBS: int = 1


# Default number of points along each axis for the calibration grid.
DEFAULT_CALIBRATION_N: int = 3

# Default step size in microns for the calibration grid.
DEFAULT_CALIBRATION_STEP_UM: float = 60.0

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
