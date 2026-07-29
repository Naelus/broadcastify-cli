from pathlib import Path
from struct import unpack
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]


def test_native_executable_has_multiresolution_application_icon() -> None:
    project = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "BroadcastifyCli.WinUI.csproj"
    ).getroot()
    application_icon = next(
        element
        for element in project.iter()
        if element.tag.endswith("ApplicationIcon")
    )
    assert application_icon.text == r"Assets\BroadcastifyDesktop.ico"

    icon = ROOT / "BroadcastifyCli.WinUI" / "Assets" / "BroadcastifyDesktop.ico"
    header = icon.read_bytes()[:6]
    reserved, image_type, image_count = unpack("<HHH", header)
    assert reserved == 0
    assert image_type == 1
    assert image_count >= 7
