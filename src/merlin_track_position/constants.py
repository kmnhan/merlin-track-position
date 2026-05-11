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

# Image size for initial crop from each camera array.
IMAGE_WIDTH_CAM0: int = 704
IMAGE_HEIGHT_CAM0: int = 480
IMAGE_WIDTH_CAM1: int = 1440
IMAGE_HEIGHT_CAM1: int = 1080

# If you change this, make sure to also update FrameGrabbber FSM UI2.vi
FRAMEGRAB_SERVER_PORT: int = 6553

# Change this if you change the Basler camera
BASLER_CAMERA_SERIAL = "40055360"
BASLER_EXPOSURE_US: float = 300000.0  # 300 ms exposure time


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
