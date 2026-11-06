"""Native Photo Desk camera-setting rows, including the recorded Kodak text."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from photodesk import camera, pipeline
from photodesk.gui.panels import ImageInfoPanel
from photodesk.gui.res import MissingResources, Resources


# Transcribed from the supplied native screenshots and verified against tag 1009
# in GQ432034.DCR. These are shot settings, not editable processing parameters.
CAPTURE_ROWS = (
    "Exposure Bias: Off",
    "ISO Speed:  160",
    "Aperture:  f1.8",
    "Shutter:  1/4000",
    "Exposure Mode:  A",
    "Meter Mode:  CW",
    "Compensation:  +1.0",
    "Flash Compensation:  +0.0",
    "Auto Bracket:  None",
    "Popup Flash:  Off",
    "Flash Sync Mode:  Front curtain",
    "Flash Fired:  No",
    "AF Function:  S",
    "Focal Length:  135",
    "Min Aperture:  f1.8",
    "Max Aperture:  f1.8",
    "Shooting Mode:  Single",
    "Lens Optimization: Auto",
    " (Auto strength = 10)",
    "White bal: Standard Daylight 5500",
    "Time: 11:13:25.790",
)


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    try:
        resources = Resources.load()
    except MissingResources as exc:
        pytest.skip(str(exc))
    widget = ImageInfoPanel(resources)
    yield widget, resources
    widget.close()
    widget.deleteLater()
    app.processEvents()


def entry(path, meta):
    return SimpleNamespace(path=path, meta=meta, sounds=[], settings=pipeline.Settings(),
                           model="DCS Pro SLR/n", make="Kodak", is_raw=True, file_type="R")


def displayed(panel):
    box = panel.b[1054]
    return [box.item(i).text() for i in range(box.count())]


def test_original_camera_rows_and_recorded_settings_are_preserved(panel, tmp_path):
    widget, resources = panel
    path = tmp_path / "capture.DCR"
    with path.open("wb") as file:
        file.truncate(round(12.8 * 1024 * 1024))
    meta = dict(serial="PSLRN-24584", firmware="5.4.1", image_number=2034,
                capture_timestamp="2026:10:05 11:13:25", timestamp="2000-01-01T00:00:00",
                width=4500, height=3000, camera_settings=CAPTURE_ROWS,
                iso=999, aperture=8.0, shutter=1.0, focal_length=24.0)
    image = entry(path, meta)
    widget.set_entry(image)
    s = resources.s
    expected = [s(50000) + "DCS Pro SLR/n", s(50001) + "PSLRN-24584",
                s(50002) + "5.4.1", s(50003) + "2034", s(50004) + "10/05/2026",
                s(50005) + "11:13:25", s(50006) + "4500", s(50007) + "3000",
                s(50010).replace("%.1f", "13.5"), s(50009).replace("%.1f", "12.8"),
                *CAPTURE_ROWS]
    assert displayed(widget) == expected
    image.settings.exposure = 2.0
    image.settings.white_point = 3200
    widget.set_entry(image)
    assert displayed(widget) == expected


def test_missing_capture_text_keeps_only_known_metadata(panel, tmp_path):
    widget, resources = panel
    path = tmp_path / "capture.DCR"
    path.write_bytes(b"x")
    image = entry(path, dict(timestamp="2020-02-03T04:05:06", iso=100, aperture=2.8,
                            shutter=0.008, focal_length=35, image_number=0))
    widget.set_entry(image)
    rows = displayed(widget)
    assert resources.s(50003) + "0" in rows
    assert resources.s(50004) + "02/03/2020" in rows
    assert rows[-4:] == ["ISO Speed:  100", "Aperture:  f2.8", "Shutter:  1/125", "Focal Length:  35"]
    assert not any("Flash" in row or "White bal" in row for row in rows)
    image.meta["timestamp"] = "invalid"
    widget.set_entry(image)
    assert not any(row.startswith(resources.s(50004)) for row in displayed(widget))
    widget.set_entry(None)
    assert displayed(widget) == []


def test_sample_capture_text_matches_native_screenshot(panel):
    path = os.environ.get("PHOTODESK_SLRN_SAMPLE")
    if not path:
        pytest.skip("PHOTODESK_SLRN_SAMPLE is not configured")
    from pathlib import Path
    metadata = camera.read_capture_metadata(path)
    if metadata.get("serial") != "PSLRN-24584" or metadata.get("image_number") != 2034:
        pytest.skip("The native screenshot reference is GQ432034 from PSLRN-24584")
    assert metadata["camera_settings"] == CAPTURE_ROWS
    metadata.update(width=4500, height=3000)
    widget, resources = panel
    widget.set_entry(entry(Path(path), metadata))
    rows = displayed(widget)
    assert rows[-len(CAPTURE_ROWS):] == list(CAPTURE_ROWS)
    assert resources.s(50009).replace("%.1f", "12.8") in rows
