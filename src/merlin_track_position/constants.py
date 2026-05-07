"""Defines port numbers for ZMQ communication and other constants."""

import sys
import pathlib

# Mapping from shorthand motor names to actual motor names at the beamline.
MOTOR_NAMES = {
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

# Image size for initial crop from the array returned by the framegrabber.
IMAGE_WIDTH: int = 704
IMAGE_HEIGHT: int = 480

# If you change this, make sure to also update FrameGrabbber FSM UI2.vi
FRAMEGRAB_SERVER_PORT = 6553

# If you change this, make sure to also update TrackTemperatureBL403 BCS Driver.vi
MOTOR_SERVER_PORT = 6554

# These are default settings for BCS API server, probably shouldn't need to be changed.
BCS_SERVER_HOST = "localhost"
BCS_SERVER_PORT = 5577

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
