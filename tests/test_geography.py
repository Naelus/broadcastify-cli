from __future__ import annotations

import io
import zipfile
from pathlib import Path

from broadcastify_cli.geography import ZipCentroidCatalog, haversine_miles


def _gazetteer(path: Path) -> None:
    rows = (
        "GEOID|GEOIDFQ|ALAND|AWATER|ALAND_SQMI|AWATER_SQMI|INTPTLAT|INTPTLONG\n"
        "12345|860Z200US12345|1|0|1|0|39.0000|-77.0000\n"
        "12346|860Z200US12346|1|0|1|0|39.0500|-76.9500\n"
        "12358|860Z200US12358|1|0|1|0|40.0000|-78.0000\n"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("2025_Gaz_zcta_national.txt", rows)


def test_haversine_returns_reasonable_distance() -> None:
    distance = haversine_miles(39.0000, -77.0000, 39.0500, -76.9500)
    assert 4 < distance < 5


def test_radius_catalog_orders_and_limits_cached_zips(tmp_path: Path) -> None:
    cache = tmp_path / "zcta.zip"
    _gazetteer(cache)
    catalog = ZipCentroidCatalog(cache)

    values = catalog.nearest("12345", 10, limit=2)

    assert [value.zip_code for value in values] == ["12345", "12346"]
    assert values[0].distance_miles == 0
    assert 4 < values[1].distance_miles < 5


def test_radius_catalog_rejects_unknown_zcta(tmp_path: Path) -> None:
    cache = tmp_path / "zcta.zip"
    _gazetteer(cache)
    catalog = ZipCentroidCatalog(cache)

    try:
        catalog.nearest("00000", 10)
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("Unknown ZCTAs should require an exact ZIP list.")


class _DownloadResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = {"Content-Length": str(len(body))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.body[index : index + chunk_size] for index in range(0, len(self.body), chunk_size)]


class _DownloadSession:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def get(self, *_args: object, **_kwargs: object) -> _DownloadResponse:
        self.calls += 1
        return _DownloadResponse(self.body)


def test_catalog_downloads_once_then_uses_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _gazetteer(source)
    session = _DownloadSession(source.read_bytes())
    cache = tmp_path / "cache" / "zcta.zip"

    assert ZipCentroidCatalog(cache, session=session).nearest("12345", 10)[0].zip_code == "12345"  # type: ignore[arg-type]
    assert ZipCentroidCatalog(cache, session=session).nearest("12345", 10)[0].zip_code == "12345"  # type: ignore[arg-type]
    assert session.calls == 1
