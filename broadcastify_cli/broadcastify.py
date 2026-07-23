from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import unquote, urljoin, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from bs4 import BeautifulSoup

from .models import FeedSearchResult
from .quota import ArchiveRequestBudgetExceeded, ArchiveRequestLedger


ProgressCallback = Callable[[int, int, str], None]


class BroadcastifyError(RuntimeError):
    pass


class AuthenticationError(BroadcastifyError):
    pass


class DownloadLimitExceeded(BroadcastifyError):
    pass


class _DownloadThrottle:
    """Coordinate a shared cooldown and concurrency limit for one archive job."""

    def __init__(self, parallelism: int, min_request_interval: float = 0.0) -> None:
        self._condition = threading.Condition()
        self._limit = max(1, parallelism)
        self._min_request_interval = max(0.0, min_request_interval)
        self._active = 0
        self._cooldown_until = 0.0
        self._next_request_at = 0.0
        self._serialized = False
        self._blocked_reason: str | None = None

    @property
    def serialized(self) -> bool:
        with self._condition:
            return self._serialized

    def acquire(self) -> None:
        with self._condition:
            while True:
                if self._blocked_reason:
                    raise DownloadLimitExceeded(self._blocked_reason)
                now = time.monotonic()
                cooldown = self._cooldown_until - now
                pacing = self._next_request_at - now
                if self._active < self._limit and cooldown <= 0 and pacing <= 0:
                    self._active += 1
                    self._next_request_at = now + self._min_request_interval
                    return
                delay = max(cooldown, pacing)
                self._condition.wait(timeout=max(0.05, delay) if delay > 0 else None)

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    def block(self, reason: str) -> None:
        with self._condition:
            self._serialized = True
            self._limit = 1
            self._blocked_reason = reason
            self._condition.notify_all()

    def defer(self, seconds: float, *, serialize: bool) -> None:
        with self._condition:
            if serialize:
                # A 429 is shared state, not one unlucky request. Stop admitting
                # parallel work after the already-active requests drain.
                self._serialized = True
                self._limit = 1
            self._cooldown_until = max(
                self._cooldown_until,
                time.monotonic() + max(0.0, seconds),
            )
            self._condition.notify_all()

    def configure_parallelism(self, parallelism: int) -> None:
        with self._condition:
            if not self._serialized:
                self._limit = max(1, parallelism)
                self._condition.notify_all()


class BroadcastifyClient:
    """Client for the website endpoints used by the original CLI.

    These are intentionally kept behind one class because they are website
    endpoints rather than a stable public API.
    """

    BASE_URL = "https://www.broadcastify.com"
    LOGIN_URL = f"{BASE_URL}/login/"
    # These are private website endpoints used by Broadcastify's current
    # archives page. Keep them centralized: unlike a public API, they can move.
    ARCHIVE_LIST_URL = f"{BASE_URL}/archives/api/archives.php"
    ARCHIVE_DOWNLOAD_URL = f"{BASE_URL}/archives/download"
    FEED_SEARCH_URL = f"{BASE_URL}/listen/"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        cookie_path: str | Path = "cookies.json",
        timeout: float = 30.0,
        user_agent: str | None = None,
        download_attempts: int = 7,
        download_request_interval: float = 5.0,
        download_backoff_base: float = 2.0,
        download_backoff_max: float = 120.0,
        rate_limit_backoff_base: float = 30.0,
        rate_limit_backoff_max: float = 300.0,
        random_uniform: Callable[[float, float], float] | None = None,
        quota_ledger: ArchiveRequestLedger | None = None,
        quota_ledger_path: str | Path | None = None,
    ) -> None:
        # BROADCASTIFY_* avoids colliding with Windows' built-in USERNAME
        # environment variable. The legacy names remain supported when a
        # PASSWORD value is explicitly configured.
        legacy_password = os.getenv("PASSWORD")
        self.username = (
            username
            or os.getenv("BROADCASTIFY_USERNAME")
            or (os.getenv("USERNAME") if legacy_password else None)
        )
        self.password = password or os.getenv("BROADCASTIFY_PASSWORD") or legacy_password
        self.cookie_path = Path(cookie_path)
        self.timeout = timeout
        self.download_attempts = max(1, int(download_attempts))
        self.download_request_interval = max(0.0, float(download_request_interval))
        self.download_backoff_base = max(0.0, float(download_backoff_base))
        self.download_backoff_max = max(0.0, float(download_backoff_max))
        self.rate_limit_backoff_base = max(0.0, float(rate_limit_backoff_base))
        self.rate_limit_backoff_max = max(0.0, float(rate_limit_backoff_max))
        self._random_uniform = random_uniform or random.uniform
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._auth_lock = threading.Lock()
        self._authenticated = False
        self._download_throttle: _DownloadThrottle | None = None
        self._download_throttle_lock = threading.Lock()
        self._archive_filename_prefixes: dict[str, str] = {}
        self._archive_timezones: dict[str, ZoneInfo] = {}
        self._county_feed_cache: dict[str, list[FeedSearchResult]] = {}
        self._quota_ledger = quota_ledger
        self._quota_ledger_path = quota_ledger_path

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "BroadcastifyClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def authenticate(self, force: bool = False) -> None:
        with self._auth_lock:
            if self._authenticated and not force:
                return

            if not force and self._load_cookie():
                self._authenticated = True
                return

            if not self.username or not self.password:
                raise AuthenticationError(
                    "Sign in from the desktop app or configure "
                    "BROADCASTIFY_USERNAME and BROADCASTIFY_PASSWORD in .env."
                )

            self._clear_auth_cookie()

            response = self.session.post(
                self.LOGIN_URL,
                data={
                    "username": self.username,
                    "password": self.password,
                    "action": "auth",
                    "redirect": self.BASE_URL,
                },
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": self.BASE_URL,
                    "Referer": self.LOGIN_URL,
                },
                allow_redirects=False,
                timeout=self.timeout,
            )

            token = response.cookies.get("bcfyuser1")
            if not token:
                match = re.search(r"(?:^|;\s*)bcfyuser1=([^;]+)", response.headers.get("Set-Cookie", ""))
                token = match.group(1) if match else None

            if response.status_code not in {301, 302, 303} or not token:
                raise AuthenticationError(
                    f"Broadcastify login failed with HTTP {response.status_code}."
                )

            self._set_cookie(token)
            self.cookie_path.parent.mkdir(parents=True, exist_ok=True)
            self.cookie_path.write_text(
                json.dumps({"bcfyuser1": token}), encoding="utf-8"
            )
            self._authenticated = True

    def _load_cookie(self) -> bool:
        if not self.cookie_path.exists():
            return False
        try:
            value = json.loads(self.cookie_path.read_text(encoding="utf-8"))
            token = value.get("bcfyuser1")
        except (OSError, json.JSONDecodeError, AttributeError):
            return False
        if not token:
            return False
        self._set_cookie(str(token))
        return True

    def _set_cookie(self, token: str) -> None:
        self.session.cookies.set(
            "bcfyuser1", token, domain=".broadcastify.com", path="/"
        )

    def _clear_auth_cookie(self) -> None:
        for cookie in list(self.session.cookies):
            if cookie.name == "bcfyuser1":
                self.session.cookies.clear(cookie.domain, cookie.path, cookie.name)

    def search_feeds(self, query: str) -> list[FeedSearchResult]:
        query = query.strip()
        if len(query) < 2:
            raise ValueError("Enter at least two characters to search for feeds.")
        response = self.session.get(
            self.FEED_SEARCH_URL,
            params={"q": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        # The site's result page mixes direct feed matches with links to
        # matching county directories. A city/state or ZIP search can put the
        # desired feed only in one of those county pages, so parsing just the
        # first table makes the desktop search look much narrower than the
        # website even though both use the same endpoint.
        results = {
            result.feed_id: result
            for result in self.parse_feed_search_html(response.text)
        }
        county_paths = self.parse_zip_county_paths(response.text)
        for result in self._feeds_for_county_paths(county_paths):
            results.setdefault(result.feed_id, result)
        return self.rank_feed_search_results(query, list(results.values()))

    def search_area_feeds(
        self,
        zip_codes: Sequence[str],
        *,
        zip_distances: dict[str, float] | None = None,
    ) -> list[dict[str, object]]:
        """Search several ZIP codes and return a feed-id-deduplicated result set.

        Broadcastify's website search is the source of truth here; this deliberately
        uses the same reverse-engineered listen search as the single-feed UI instead
        of an official API or a separate geographic database.
        """
        normalized: list[str] = []
        for raw in zip_codes:
            value = str(raw).strip()
            if not re.fullmatch(r"\d{5}", value):
                raise ValueError(f"Invalid ZIP code: {value or '(blank)'}. Use five digits.")
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            raise ValueError("Enter at least one five-digit ZIP code.")
        if len(normalized) > 20:
            raise ValueError("Area searches are limited to 20 ZIP codes at a time.")

        by_feed: dict[str, dict[str, object]] = {}
        for zip_code in normalized:
            for result in self.feeds_for_zip(zip_code):
                value = by_feed.setdefault(
                    result.feed_id,
                    {**result.to_dict(), "matched_zip_codes": []},
                )
                matches = value["matched_zip_codes"]
                if isinstance(matches, list) and zip_code not in matches:
                    matches.append(zip_code)

        ordered = sorted(
            by_feed.values(),
            key=lambda value: (
                min(
                    (normalized.index(str(code)) for code in value.get("matched_zip_codes", [])),
                    default=len(normalized),
                ),
                -len(value.get("matched_zip_codes", [])),
                -int(value.get("listeners", 0)),
                str(value.get("name", "")).lower(),
            ),
        )
        distances = zip_distances or {}
        for rank, value in enumerate(ordered, start=1):
            matches = [str(code) for code in value.get("matched_zip_codes", [])]
            nearest_zip = min(
                matches,
                key=lambda code: (
                    distances.get(code, float("inf")),
                    normalized.index(code) if code in normalized else len(normalized),
                ),
                default="",
            )
            value["nearest_zip_code"] = nearest_zip
            value["distance_miles"] = (
                round(float(distances[nearest_zip]), 2)
                if nearest_zip in distances
                else None
            )
            value["priority_rank"] = rank
        if distances:
            ordered.sort(
                key=lambda value: (
                    float(value["distance_miles"])
                    if value.get("distance_miles") is not None
                    else float("inf"),
                    -len(value.get("matched_zip_codes", [])),
                    -int(value.get("listeners", 0)),
                    str(value.get("name", "")).lower(),
                )
            )
            for rank, value in enumerate(ordered, start=1):
                value["priority_rank"] = rank
        return ordered

    def feeds_for_zip(self, zip_code: str) -> list[FeedSearchResult]:
        """Follow the site's ZIP match to its county feed directory."""
        response = self.session.get(
            self.FEED_SEARCH_URL,
            params={"q": zip_code},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._feeds_for_county_paths(
            self.parse_zip_county_paths(response.text)
        )

    def _feeds_for_county_paths(
        self, county_paths: Sequence[str]
    ) -> list[FeedSearchResult]:
        results: list[FeedSearchResult] = []
        seen: set[str] = set()
        for path in county_paths:
            if path not in self._county_feed_cache:
                county_response = self.session.get(
                    urljoin(self.BASE_URL, path), timeout=self.timeout
                )
                county_response.raise_for_status()
                self._county_feed_cache[path] = self.parse_feed_search_html(
                    county_response.text
                )
            for result in self._county_feed_cache[path]:
                if result.feed_id not in seen:
                    seen.add(result.feed_id)
                    results.append(result)
        return results

    @staticmethod
    def rank_feed_search_results(
        query: str, results: Sequence[FeedSearchResult]
    ) -> list[FeedSearchResult]:
        """Put concise name matches first without city- or feed-specific rules."""

        query_words = [
            value
            for value in re.findall(r"[a-z0-9]+", query.lower())
            if len(value) >= 3
        ]
        normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        first_word = query_words[0] if query_words else ""

        def sort_key(result: FeedSearchResult) -> tuple[object, ...]:
            name = " ".join(re.findall(r"[a-z0-9]+", result.name.lower()))
            location = " ".join(
                re.findall(r"[a-z0-9]+", result.location.lower())
            )
            name_matches = sum(word in name.split() for word in query_words)
            location_matches = sum(word in location.split() for word in query_words)
            exact_phrase = bool(normalized_query and normalized_query in name)
            starts_with_query = bool(first_word and name.startswith(first_word))
            public_safety = result.genre.strip().lower() == "public safety"
            return (
                -int(exact_phrase),
                -int(starts_with_query),
                -name_matches,
                -location_matches,
                -int(public_safety),
                len(name.split()),
                -result.listeners,
                name,
                result.feed_id,
            )

        return sorted(results, key=sort_key)

    @staticmethod
    def parse_zip_county_paths(html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        paths: list[str] = []
        for link in soup.select('main a[href*="/listen/ctid/"]'):
            href = str(link.get("href") or "")
            if re.fullmatch(r"/listen/ctid/\d+", href) and href not in paths:
                paths.append(href)
        return paths

    @staticmethod
    def parse_feed_search_html(html: str) -> list[FeedSearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[FeedSearchResult] = []
        seen: set[str] = set()
        heading = soup.select_one("main h1")
        county_name = ""
        if heading is not None:
            county_name = re.sub(
                r"\s*-\s*Live Audio Feeds.*$", "", heading.get_text(" ", strip=True)
            )
        state_link = soup.select_one('a[href*="/listen/stid/"]')
        state_name = state_link.get_text(" ", strip=True) if state_link else ""
        county_location = " - ".join(value for value in (state_name, county_name) if value)

        for link in soup.select('a[href*="/listen/feed/"]'):
            href = link.get("href", "")
            match = re.search(r"/listen/feed/(\d+)", href)
            row = link.find_parent("tr")
            if not match or row is None:
                continue

            feed_id = match.group(1)
            if feed_id in seen:
                continue
            seen.add(feed_id)

            cells = row.find_all("td")
            feed_cell = link.find_parent("td")
            if feed_cell is None or feed_cell not in cells:
                continue
            feed_index = cells.index(feed_cell)
            name = link.get_text(" ", strip=True)
            location = (
                cells[feed_index - 1].get_text(" ", strip=True)
                if feed_index >= 2
                else county_location
            )
            feed_text = feed_cell.get_text(" ", strip=True)
            description = feed_text[len(name) :].strip() if feed_text.startswith(name) else feed_text
            genre = (
                cells[feed_index + 1].get_text(" ", strip=True)
                if len(cells) > feed_index + 1
                else ""
            )
            listener_text = (
                cells[feed_index + 2].get_text(" ", strip=True)
                if len(cells) > feed_index + 2
                else "0"
            )
            listener_match = re.search(r"[\d,]+", listener_text)
            listeners = int(listener_match.group(0).replace(",", "")) if listener_match else 0
            status_node = cells[0].find(attrs={"title": True}) if cells else None
            status = status_node.get("title", "") if status_node else ""

            results.append(
                FeedSearchResult(
                    feed_id=feed_id,
                    name=name,
                    location=location,
                    description=description,
                    genre=genre,
                    listeners=listeners,
                    status=status,
                )
            )

        return results

    def get_archive_ids(
        self,
        feed_id: str,
        archive_date: date,
        allow_reauthenticate: bool = True,
    ) -> list[str]:
        response = self.session.get(
            self.ARCHIVE_LIST_URL,
            params={"feedId": feed_id, "date": archive_date.strftime("%m/%d/%Y")},
            headers={"Referer": f"{self.BASE_URL}/archives/feed/{feed_id}"},
            timeout=self.timeout,
        )
        unauthorized = self._is_login_response(response)
        if unauthorized and allow_reauthenticate:
            self.authenticate(force=True)
            return self.get_archive_ids(
                feed_id, archive_date, allow_reauthenticate=False
            )
        if unauthorized:
            raise AuthenticationError("Archive listing was not authorized.")
        response.raise_for_status()
        try:
            payload = response.json()
            timezone_name = payload.get("timezone")
            if isinstance(timezone_name, str):
                try:
                    self._archive_timezones[feed_id] = ZoneInfo(timezone_name)
                except (ZoneInfoNotFoundError, ValueError):
                    pass
            self._archive_filename_prefixes.update(
                self.parse_archive_filename_prefixes(payload)
            )
            return self.parse_archive_payload(payload)
        except (ValueError, AttributeError, TypeError) as exc:
            raise BroadcastifyError("Broadcastify returned an invalid archive listing.") from exc

    @staticmethod
    def parse_archive_payload(payload: object) -> list[str]:
        if not isinstance(payload, dict):
            raise TypeError("Archive payload must be an object.")
        if "archives" not in payload:
            raise TypeError("Archive payload has no archives field.")
        rows = payload["archives"]
        if not isinstance(rows, list):
            raise TypeError("Archive list must be an array.")
        return [
            str(row["id"])
            for row in rows
            if isinstance(row, dict) and row.get("id") is not None
        ]

    @staticmethod
    def parse_archive_filename_prefixes(payload: object) -> dict[str, str]:
        if not isinstance(payload, dict) or not isinstance(payload.get("archives"), list):
            return {}
        timezone_name = payload.get("timezone")
        if not isinstance(timezone_name, str):
            return {}
        try:
            feed_timezone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return {}
        prefixes: dict[str, str] = {}
        for row in payload["archives"]:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            try:
                started = datetime.fromtimestamp(int(row["startTs"]), feed_timezone)
            except (KeyError, TypeError, ValueError, OSError, OverflowError):
                continue
            prefixes[str(row["id"])] = started.strftime("%Y%m%d%H%M")
        return prefixes

    @staticmethod
    def _is_login_response(response: requests.Response) -> bool:
        if response.status_code in {401, 403}:
            return True
        if "/login" in urlparse(response.url).path.lower():
            return True
        content_type = response.headers.get("Content-Type", "").lower()
        if not content_type.startswith("text/html"):
            return False
        body = response.text[:16_384].lower()
        return (
            'name="username"' in body
            and ('name="password"' in body or 'action="auth"' in body)
        )

    def download_day(
        self,
        feed_id: str,
        archive_date: date,
        output_dir: str | Path,
        jobs: int = 1,
        progress: ProgressCallback | None = None,
        admit_download: Callable[[], None] | None = None,
    ) -> list[Path]:
        self.authenticate()
        archive_ids = self.get_archive_ids(feed_id, archive_date)
        day_dir = Path(output_dir) / feed_id / archive_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        if not archive_ids:
            if progress:
                progress(0, 0, f"No archives found for {archive_date.isoformat()}.")
            return []

        workers = max(1, min(jobs, len(archive_ids)))
        # Broadcastify's standard archive guidance calls for one file at a
        # time. Worker threads may prepare/cache-check tasks concurrently, but
        # this shared throttle admits only one upstream media request.
        throttle = self._shared_download_throttle(1)
        successful = 0
        progress_lock = threading.Lock()

        def retry_notice(message: str) -> None:
            if progress:
                with progress_lock:
                    progress(successful, len(archive_ids), message)

        if throttle.serialized:
            retry_notice(
                "A previous Broadcastify rate limit is still active; "
                "archive requests will continue one at a time."
            )
        elif self.download_request_interval > 0:
            per_minute = 60.0 / self.download_request_interval
            retry_notice(
                f"Pacing archive requests at least {self.download_request_interval:.1f}s "
                f"apart (up to {per_minute:.0f}/minute)."
            )
        quota = self.archive_quota_status()
        retry_notice(
            f"Installation archive budget: {quota['remaining']}/"
            f"{quota['automated_limit']} automated requests available in the "
            f"rolling 24-hour window; {quota['user_reserve']} requests are reserved "
            "for manual use."
        )

        # Acquire both ends of the live player window before older backlog:
        # the archive API is newest-first, so entries 0 and 1 are the current
        # completed track and the immediately previous track. Besides making
        # current-feed runs useful quickly, the first request refreshes an
        # expired cached cookie before concurrent older requests begin.
        priority_ids = archive_ids[:2]
        downloaded: list[Path] = []
        failures: list[str] = []
        failure_keys: set[tuple[type[Exception], str]] = set()
        limit_failure: DownloadLimitExceeded | None = None

        def was_cached(archive_id: str) -> bool:
            return self._existing_archive(
                day_dir,
                feed_id,
                archive_id,
                self._archive_filename_prefixes.get(archive_id),
            ) is not None

        def ready_message(
            path: Path,
            *,
            cached: bool,
            current: int,
            total: int,
            refreshed: bool = False,
        ) -> str:
            source = "cached locally" if cached else "downloaded from Broadcastify"
            suffix = " (current-day refresh)" if refreshed else ""
            return f"Ready {current}/{total} — {source}: {path.name}{suffix}"

        for priority_index, archive_id in enumerate(priority_ids):
            cached = was_cached(archive_id)
            path = self.download_archive(
                feed_id,
                archive_date,
                archive_id,
                day_dir,
                allow_reauthenticate=priority_index == 0,
                throttle=throttle,
                notice=retry_notice,
                admit_download=admit_download,
            )
            downloaded.append(path)
            successful += 1
            if progress:
                progress(
                    successful,
                    len(archive_ids),
                    ready_message(
                        path,
                        cached=cached,
                        current=successful,
                        total=len(archive_ids),
                    ),
                )

        remaining_ids = archive_ids[len(priority_ids) :]
        if remaining_ids:
            workers = max(1, min(jobs, len(remaining_ids)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for archive_id in remaining_ids:
                    cached = was_cached(archive_id)
                    future = executor.submit(
                        self.download_archive,
                        feed_id,
                        archive_date,
                        archive_id,
                        day_dir,
                        False,
                        throttle,
                        retry_notice,
                        admit_download,
                    )
                    futures[future] = (archive_id, cached)
                for future in as_completed(futures):
                    archive_id, cached = futures[future]
                    try:
                        path = future.result()
                        downloaded.append(path)
                        with progress_lock:
                            successful += 1
                            if progress:
                                progress(
                                    successful,
                                    len(archive_ids),
                                    ready_message(
                                        path,
                                        cached=cached,
                                        current=successful,
                                        total=len(archive_ids),
                                    ),
                                )
                    except Exception as exc:  # reported after active work drains
                        if isinstance(exc, DownloadLimitExceeded):
                            limit_failure = exc
                        failure_key = (type(exc), str(exc))
                        is_new_failure = failure_key not in failure_keys
                        failure_keys.add(failure_key)
                        if is_new_failure:
                            failures.append(f"{archive_id}: {exc}")
                        if progress and is_new_failure:
                            with progress_lock:
                                progress(
                                    successful,
                                    len(archive_ids),
                                    f"Archive {archive_id} failed after retries; "
                                    f"{successful}/{len(archive_ids)} ready.",
                                )

        if limit_failure is not None:
            # Preserve the specific exception so a range job can stop issuing
            # new download requests while still processing days that were
            # completed before the account-wide quota was reached.
            raise limit_failure
        if failures:
            raise BroadcastifyError(
                "One or more archive downloads failed:\n" + "\n".join(failures)
            )

        # Today's archive list grows in 30-minute tracks. Refresh once after the
        # initial snapshot so a track finalized while an older backlog was
        # downloading is not omitted from the completed manifest.
        if self._is_current_archive_date(feed_id, archive_date):
            refreshed_ids = self.get_archive_ids(feed_id, archive_date)
            known_ids = set(archive_ids)
            new_ids = [
                archive_id
                for archive_id in refreshed_ids
                if archive_id not in known_ids
            ]
            refreshed_total = len(archive_ids) + len(new_ids)
            for archive_id in new_ids:
                cached = was_cached(archive_id)
                path = self.download_archive(
                    feed_id,
                    archive_date,
                    archive_id,
                    day_dir,
                    allow_reauthenticate=False,
                    throttle=throttle,
                    notice=retry_notice,
                    admit_download=admit_download,
                )
                downloaded.append(path)
                successful += 1
                if progress:
                    progress(
                        successful,
                        refreshed_total,
                        ready_message(
                            path,
                            cached=cached,
                            current=successful,
                            total=refreshed_total,
                            refreshed=True,
                        ),
                    )
        return sorted(downloaded)

    def _is_current_archive_date(
        self,
        feed_id: str,
        archive_date: date,
    ) -> bool:
        feed_timezone = self._archive_timezones.get(feed_id)
        today = (
            datetime.now(feed_timezone).date()
            if feed_timezone is not None
            else date.today()
        )
        return archive_date == today

    def cached_day(
        self,
        feed_id: str,
        archive_date: date,
        output_dir: str | Path,
    ) -> tuple[list[Path], int]:
        """Return a complete cached day without touching download endpoints.

        The second value is the number of archives Broadcastify currently
        lists for the day. An empty file list with a positive total means the
        cache is incomplete; ``([], 0)`` is a legitimately empty archive day.
        """

        self.authenticate()
        archive_ids = self.get_archive_ids(feed_id, archive_date)
        day_dir = Path(output_dir) / feed_id / archive_date.strftime("%Y%m%d")
        cached: list[Path] = []
        for archive_id in archive_ids:
            existing = self._existing_archive(
                day_dir,
                feed_id,
                archive_id,
                self._archive_filename_prefixes.get(archive_id),
            )
            if existing is None:
                return [], len(archive_ids)
            cached.append(existing)
        return sorted(cached), len(archive_ids)

    def download_archive(
        self,
        feed_id: str,
        archive_date: date,
        archive_id: str,
        day_dir: Path,
        allow_reauthenticate: bool = True,
        throttle: _DownloadThrottle | None = None,
        notice: Callable[[str], None] | None = None,
        admit_download: Callable[[], None] | None = None,
    ) -> Path:
        existing = self._existing_archive(
            day_dir,
            feed_id,
            archive_id,
            self._archive_filename_prefixes.get(archive_id),
        )
        if existing is not None:
            return existing
        url = f"{self.ARCHIVE_DOWNLOAD_URL}/{archive_id}"
        request_throttle = throttle or _DownloadThrottle(
            1, self.download_request_interval
        )
        reauthentication_available = allow_reauthenticate
        for attempt in range(1, self.download_attempts + 1):
            if admit_download is not None:
                admit_download()
            request_throttle.acquire()
            request_id: int | None = None
            try:
                try:
                    try:
                        request_id = self._archive_quota().reserve(
                            feed_id=feed_id,
                            archive_date=archive_date.isoformat(),
                            archive_id=archive_id,
                        )
                    except ArchiveRequestBudgetExceeded as exc:
                        message = str(exc)
                        request_throttle.block(message)
                        if notice:
                            notice(message)
                        raise DownloadLimitExceeded(message) from exc
                    with self.session.get(
                        url,
                        headers={"Referer": f"{self.BASE_URL}/archives/feed/{feed_id}"},
                        stream=True,
                        timeout=(self.timeout, 120.0),
                        allow_redirects=True,
                    ) as response:
                        assert request_id is not None
                        self._archive_quota().finish(
                            request_id,
                            outcome=f"http_{response.status_code}",
                            http_status=response.status_code,
                        )
                        request_id = None
                        if response.status_code == 429:
                            explicit_limit = self._is_download_limit_response(response)
                            message = (
                                "Broadcastify's archive request limit is exhausted."
                                if explicit_limit
                                else "Broadcastify returned HTTP 429 for an archive request."
                            )
                            message += (
                                " This installation has paused all new archive requests for "
                                "the next known rolling-window release without retrying. "
                                "Already-cached files and local processing remain available."
                            )
                            self._archive_quota().mark_rate_limited(message)
                            request_throttle.block(message)
                            if notice:
                                notice(message)
                            raise DownloadLimitExceeded(message)

                        if response.status_code in {500, 502, 503, 504}:
                            if attempt >= self.download_attempts:
                                response.raise_for_status()
                            delay = self._download_retry_delay(response, attempt)
                            request_throttle.defer(delay, serialize=False)
                            if notice:
                                notice(
                                    f"Broadcastify returned HTTP {response.status_code} for archive "
                                    f"{archive_id}; waiting {delay:.1f}s before retry "
                                    f"{attempt + 1}/{self.download_attempts}."
                                )
                            continue

                        content_type = response.headers.get("Content-Type", "").lower()
                        unauthorized = self._is_login_response(response)
                        if unauthorized and reauthentication_available:
                            reauthentication_available = False
                            self.authenticate(force=True)
                            # Retry through the loop so the current throttle slot
                            # is released before another request is admitted.
                            continue
                        if unauthorized:
                            raise AuthenticationError("Archive download was not authorized.")
                        response.raise_for_status()
                        if content_type.startswith("text/html"):
                            raise BroadcastifyError(
                                "Broadcastify returned an HTML page instead of archive audio."
                            )
                        filename = self._download_filename(response, archive_id)
                        output_path = day_dir / filename
                        if output_path.exists() and output_path.stat().st_size > 0:
                            return output_path

                        partial_path = output_path.with_suffix(output_path.suffix + ".part")
                        try:
                            with partial_path.open("wb") as handle:
                                for chunk in response.iter_content(chunk_size=1024 * 256):
                                    if chunk:
                                        handle.write(chunk)
                            partial_path.replace(output_path)
                        finally:
                            if partial_path.exists():
                                partial_path.unlink()
                        return output_path
                except (requests.ConnectionError, requests.Timeout) as exc:
                    if request_id is not None:
                        self._archive_quota().finish(
                            request_id,
                            outcome="network_error",
                        )
                        request_id = None
                    if attempt >= self.download_attempts:
                        raise
                    delay = self._exponential_backoff(attempt)
                    request_throttle.defer(delay, serialize=False)
                    if notice:
                        notice(
                            f"Temporary network error for archive {archive_id}; waiting "
                            f"{delay:.1f}s before retry {attempt + 1}/{self.download_attempts}: {exc}"
                        )
            finally:
                request_throttle.release()
        raise BroadcastifyError(
            f"Archive {archive_id} did not download after {self.download_attempts} attempts."
        )

    def _archive_quota(self) -> ArchiveRequestLedger:
        if self._quota_ledger is None:
            self._quota_ledger = ArchiveRequestLedger(self._quota_ledger_path)
        return self._quota_ledger

    def archive_quota_status(self) -> dict[str, object]:
        return self._archive_quota().status()

    def _exponential_backoff(self, attempt: int) -> float:
        base = self.download_backoff_base * (2 ** max(0, attempt - 1))
        delay = min(self.download_backoff_max, base)
        return delay + self._random_uniform(0.0, min(1.0, delay / 4))

    def _download_retry_delay(
        self, response: requests.Response, attempt: int
    ) -> float:
        retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is None:
            return self._exponential_backoff(attempt)
        # A small jitter prevents all pool workers from resuming on the same tick,
        # while still waiting at least as long as Broadcastify requested.
        return retry_after + self._random_uniform(0.0, 0.5)

    @staticmethod
    def _is_download_limit_response(response: requests.Response) -> bool:
        if response.status_code != 429:
            return False
        content_type = response.headers.get("Content-Type", "").lower()
        if "text" not in content_type and "html" not in content_type:
            return False
        return "download limit exceeded" in response.text[:1024].lower()

    def _shared_download_throttle(self, parallelism: int) -> _DownloadThrottle:
        with self._download_throttle_lock:
            if self._download_throttle is None:
                self._download_throttle = _DownloadThrottle(
                    parallelism, self.download_request_interval
                )
            else:
                self._download_throttle.configure_parallelism(parallelism)
            return self._download_throttle

    @staticmethod
    def _retry_after_seconds(
        value: str | None, now: datetime | None = None
    ) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            return max(0.0, (parsed - current).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _existing_archive(
        day_dir: Path,
        feed_id: str,
        archive_id: str,
        filename_prefix: str | None = None,
    ) -> Path | None:
        candidates = list(day_dir.glob(f"*-{archive_id}-{feed_id}.mp3"))
        candidates.extend(
            value
            for value in (day_dir / f"{archive_id}.mp3",)
            if value.exists()
        )
        # Current download URL IDs and Content-Disposition filenames contain
        # different identifiers. The archive-list payload supplies startTs and
        # the feed's IANA timezone, which together form the exact file prefix.
        if filename_prefix:
            candidates.extend(day_dir.glob(f"{filename_prefix}-*-{feed_id}.mp3"))
        return next(
            (value for value in candidates if value.is_file() and value.stat().st_size > 0),
            None,
        )

    @staticmethod
    def _download_filename(response: requests.Response, archive_id: str) -> str:
        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, re.I)
        if match:
            filename = unquote(match.group(1)).strip()
        else:
            filename = Path(urlparse(response.url).path).name
        if not filename or filename.isdigit():
            filename = f"{archive_id}.mp3"
        if not filename.lower().endswith(".mp3"):
            filename += ".mp3"
        return Path(filename).name
