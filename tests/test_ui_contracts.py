from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
XAML_NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"


def test_area_assignment_brief_is_collapsed_in_native_and_web() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    native = next(
        element
        for element in root.iter()
        if element.tag.endswith("Expander")
        and element.attrib.get(XAML_NAME) == "AreaSummaryExpander"
    )

    assert native.attrib["Header"] == "Generated assignment brief"
    assert native.attrib["IsExpanded"] == "False"
    assert native.attrib["Visibility"] == "Collapsed"

    web = (ROOT / "broadcastify_cli" / "web_static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert '<details class="area-narrative"><summary>Generated assignment brief</summary>' in web
    assert '<details class="area-narrative" open' not in web
