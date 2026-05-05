"""Parse XML-like config files that store current settings.

Mostly used to read 'Instrument Scan Setup.txt' which contains info about the current
scan setting.
"""

import xml.etree.ElementTree as ET
import pathlib
import re

from merlin_track_position.constants import INSTR_SCAN_SETUP_PATH


def load_config(path):
    text = pathlib.Path(path).read_text(encoding="iso-8859-1")

    # The file is XML-ish, but not strict XML: it has raw '&' in text.
    text = re.sub(
        r"&(?!#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]+;)",
        "&amp;",
        text,
    )

    # Some files have multiple top-level elements, so wrap them.
    return ET.fromstring(f"<Root>{text}</Root>")


def _typed_value(node):
    """Try to cast into a python object based on the LabVIEW type tag."""
    tag = node.tag
    raw = node.findtext("Val", "")

    if tag == "Boolean":
        return bool(int(raw))
    if re.fullmatch(r"[UI]\d+", tag):  # U8, U16, U32, I32, etc.
        return int(raw)
    if tag in {"DBL", "SGL", "EXT"}:
        return float(raw)
    if tag in {"EW", "EL", "EB"}:
        return int(raw)  # enum/ring value index
    if tag == "Array":
        return [
            _typed_value(child)
            for child in node
            if child.tag not in {"Name", "Dimsize"}
        ]
    if tag == "Cluster":
        return {
            child.findtext("Name"): _typed_value(child)
            for child in node
            if child.tag not in {"Name", "NumElts"}
        }
    if tag == "Path":
        return pathlib.Path(raw)

    return raw


def _get_values(path, key):
    root = load_config(path)
    return [_typed_value(node) for node in root.iter() if node.findtext("Name") == key]


def _get_value(path, key):
    matches = _get_values(path, key)
    if not matches:
        raise KeyError(key)
    return matches[0]


def get_base_file_dir() -> pathlib.Path:
    """Get the base file directory for the current scan."""
    _get_value(INSTR_SCAN_SETUP_PATH, "Data file base directory")


def get_x_start() -> float:
    """Get the X Start position for the current scan."""
    return _get_value(INSTR_SCAN_SETUP_PATH, "X Start")
