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


def test_native_and_web_expose_preview_engine_and_accuracy_upgrade() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    engine = names["DiarizationEngineComboBox"]
    engine_tags = {
        item.attrib.get("Tag")
        for item in engine
        if item.tag.endswith("ComboBoxItem")
    }
    assert engine_tags == {"community-1", "sherpa-onnx"}
    assert names["LibrarySpeakerUpgradeButton"].attrib["Visibility"] == "Collapsed"
    assert (
        names["LibrarySpeakerUpgradeButton"].attrib["Click"]
        == "LibrarySpeakerUpgrade_Click"
    )

    web_html = (
        ROOT / "broadcastify_cli" / "web_static" / "index.html"
    ).read_text(encoding="utf-8")
    web_js = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")
    assert 'id="settingDiarizationEngine"' in web_html
    assert '<option value="sherpa-onnx">Fast portable preview — CPU</option>' in web_html
    assert 'data-action="upgrade-speakers"' in web_js
    assert 'diarization_engine: "community-1"' in web_js
    assert "EnsureDiarizationSelectionCompatibility();" in (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    assert (
        "Fast portable speaker preview runs on CPU; the speaker device was reset to CPU."
        in web_js
    )
