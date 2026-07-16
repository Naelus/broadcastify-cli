from __future__ import annotations

import csv
import io
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import requests


CENSUS_ZCTA_YEAR = 2025
CENSUS_ZCTA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2025_Gazetteer/2025_Gaz_zcta_national.zip"
)
MAX_ZCTA_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_ZCTA_TEXT_BYTES = 20 * 1024 * 1024
EARTH_RADIUS_MILES = 3958.7613


@dataclass(frozen=True)
class ZipCentroid:
    zip_code: str
    latitude: float
    longitude: float
    distance_miles: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "zip_code": self.zip_code,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_miles": round(self.distance_miles, 2),
        }


def default_zcta_cache_path() -> Path:
    configured = os.getenv("RADIO_ARCHIVE_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif os.name == "nt" and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "RadioArchiveIntelligence" / "cache"
    else:
        root = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        root /= "radio-archive-intelligence"
    return root / f"census-{CENSUS_ZCTA_YEAR}-zcta.zip"


def haversine_miles(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return EARTH_RADIUS_MILES * 2 * math.asin(min(1.0, math.sqrt(value)))


class ZipCentroidCatalog:
    """Small cached reader for the Census Bureau national ZCTA Gazetteer file."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        source_url: str = CENSUS_ZCTA_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else default_zcta_cache_path()
        self.source_url = source_url
        self.session = session or requests.Session()
        self.timeout = timeout
        self._centroids: dict[str, ZipCentroid] | None = None

    def nearest(
        self,
        center_zip: str,
        radius_miles: float,
        *,
        limit: int = 20,
    ) -> list[ZipCentroid]:
        center_zip = str(center_zip).strip()
        if len(center_zip) != 5 or not center_zip.isdigit():
            raise ValueError("Center ZIP must contain five digits.")
        if not 1 <= float(radius_miles) <= 100:
            raise ValueError("Coverage radius must be between 1 and 100 miles.")
        if not 1 <= int(limit) <= 20:
            raise ValueError("Nearby ZIP discovery is limited to 1 through 20 ZIPs.")

        centroids = self._load()
        center = centroids.get(center_zip)
        if center is None:
            raise ValueError(
                f"ZIP {center_zip} is not present in the {CENSUS_ZCTA_YEAR} Census ZCTA Gazetteer. "
                "Use an exact ordered ZIP list for USPS-only or special-purpose ZIPs."
            )
        candidates = []
        for value in centroids.values():
            distance = haversine_miles(
                center.latitude,
                center.longitude,
                value.latitude,
                value.longitude,
            )
            if distance <= float(radius_miles):
                candidates.append(
                    ZipCentroid(
                        zip_code=value.zip_code,
                        latitude=value.latitude,
                        longitude=value.longitude,
                        distance_miles=distance,
                    )
                )
        candidates.sort(key=lambda value: (value.distance_miles, value.zip_code))
        return candidates[: int(limit)]

    def _load(self) -> dict[str, ZipCentroid]:
        if self._centroids is not None:
            return self._centroids
        if not self.cache_path.is_file():
            self._download()
        try:
            self._centroids = self._read_archive(self.cache_path)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"The cached Census ZIP centroid file is invalid: {self.cache_path}. "
                "Delete it and retry radius discovery."
            ) from exc
        return self._centroids

    def _download(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(self.source_url, timeout=self.timeout, stream=True)
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_ZCTA_DOWNLOAD_BYTES:
            raise RuntimeError("The Census ZIP centroid download exceeded the safety limit.")

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.cache_path.parent, delete=False, suffix=".tmp"
            ) as handle:
                temporary = Path(handle.name)
                self._copy_response(response, handle)
            # Validate before replacing a previously good cache.
            self._read_archive(temporary)
            os.replace(temporary, self.cache_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _copy_response(response: requests.Response, handle: BinaryIO) -> None:
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_ZCTA_DOWNLOAD_BYTES:
                raise RuntimeError("The Census ZIP centroid download exceeded the safety limit.")
            handle.write(chunk)

    @staticmethod
    def _read_archive(path: Path) -> dict[str, ZipCentroid]:
        with zipfile.ZipFile(path) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".txt")
            ]
            if len(members) != 1 or members[0].file_size > MAX_ZCTA_TEXT_BYTES:
                raise ValueError("Unexpected Census ZCTA archive layout.")
            with archive.open(members[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text, delimiter="|")
                if not reader.fieldnames:
                    raise ValueError("The Census ZCTA file has no header.")
                fields = {str(value).strip().upper(): value for value in reader.fieldnames}
                required = {"GEOID", "INTPTLAT", "INTPTLONG"}
                if not required.issubset(fields):
                    raise ValueError("The Census ZCTA file is missing coordinate columns.")
                values: dict[str, ZipCentroid] = {}
                for row in reader:
                    zip_code = str(row.get(fields["GEOID"]) or "").strip()
                    if len(zip_code) != 5 or not zip_code.isdigit():
                        continue
                    try:
                        latitude = float(str(row.get(fields["INTPTLAT"]) or "").strip())
                        longitude = float(str(row.get(fields["INTPTLONG"]) or "").strip())
                    except ValueError:
                        continue
                    values[zip_code] = ZipCentroid(zip_code, latitude, longitude)
        if not values:
            raise ValueError("The Census ZCTA file contained no usable ZIP centroids.")
        return values
