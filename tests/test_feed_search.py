from pathlib import Path

from broadcastify_cli.broadcastify import BroadcastifyClient
from broadcastify_cli.models import FeedSearchResult


SEARCH_HTML = """
<main>
  <table>
    <tbody>
      <tr>
        <td><span title="Online"></span></td>
        <td class="location">TX-Dallas</td>
        <td>
          <a href="/listen/feed/5318">Dallas City Police - 1 Central</a>
          <div>Dallas Police 1 Central TG 31011</div>
        </td>
        <td>Public Safety</td>
        <td>1,234</td>
      </tr>
    </tbody>
  </table>
</main>
"""

COUNTY_HTML = """
<main>
  <nav><a href="/listen/stid/48">Texas</a></nav>
  <h1>Dallas County - Live Audio Feeds</h1>
  <table><tbody><tr>
    <td><span title="Online"></span></td>
    <td><a href="/listen/feed/46336">Dallas City Police - All Divisions Dispatch</a>
        <span>Covers all DPD Division Dispatch Channels</span></td>
    <td>Public Safety</td><td>21</td>
  </tr></tbody></table>
</main>
"""

EXAMPLE_SEARCH_HTML = """
<main>
  <a href="/listen/ctid/999">ST-Example County</a>
  <table><tbody><tr>
    <td><span title="Online"></span></td><td>ST-Example County</td>
    <td><a href="/listen/feed/90004">Example City Fire Dispatch</a></td>
    <td>Public Safety</td><td>5</td>
  </tr></tbody></table>
</main>
"""

EXAMPLE_COUNTY_HTML = """
<main>
  <nav><a href="/listen/stid/99">Example State</a></nav>
  <h1>Example County - Live Audio Feeds</h1>
  <table><tbody>
    <tr><td><span title="Online"></span></td>
      <td><a href="/listen/feed/90001">Example City Public Safety</a></td>
      <td>Public Safety</td><td>4</td></tr>
    <tr><td><span title="Online"></span></td>
      <td><a href="/listen/feed/90005">Example Area Railroads</a></td>
      <td>Rail</td><td>10</td></tr>
  </tbody></table>
</main>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_parse_feed_search_result() -> None:
    results = BroadcastifyClient.parse_feed_search_html(SEARCH_HTML)

    assert len(results) == 1
    assert results[0].feed_id == "5318"
    assert results[0].name == "Dallas City Police - 1 Central"
    assert results[0].location == "TX-Dallas"
    assert results[0].description == "Dallas Police 1 Central TG 31011"
    assert results[0].genre == "Public Safety"
    assert results[0].listeners == 1234
    assert results[0].status == "Online"


def test_parser_ignores_navigation_feed_links() -> None:
    html = '<nav><a href="/listen/feed/32602">Top feed</a></nav>'
    assert BroadcastifyClient.parse_feed_search_html(html) == []


def test_parse_county_feed_table_shape() -> None:
    result = BroadcastifyClient.parse_feed_search_html(COUNTY_HTML)[0]

    assert result.feed_id == "46336"
    assert result.location == "Texas - Dallas County"
    assert result.description == "Covers all DPD Division Dispatch Channels"
    assert result.genre == "Public Safety"
    assert result.listeners == 21


def test_parse_zip_match_county_path() -> None:
    html = """
    <main><h3>Zip Code Match</h3>
      <a href="/listen/ctid/2579">Dallas, TX</a>
      <a href="/listen/ctid/2579">View Dallas County Feeds</a>
    </main>
    """
    assert BroadcastifyClient.parse_zip_county_paths(html) == ["/listen/ctid/2579"]


def test_general_search_follows_county_result_and_ranks_concise_name_match() -> None:
    client = BroadcastifyClient()
    requested: list[str] = []

    def get(url: str, **_kwargs: object) -> FakeResponse:
        requested.append(url)
        return FakeResponse(
            EXAMPLE_COUNTY_HTML if url.endswith("/listen/ctid/999") else EXAMPLE_SEARCH_HTML
        )

    client.session.get = get  # type: ignore[method-assign]

    results = client.search_feeds("Example City")

    assert [result.feed_id for result in results] == ["90001", "90004", "90005"]
    assert requested == [client.FEED_SEARCH_URL, f"{client.BASE_URL}/listen/ctid/999"]


def test_area_search_deduplicates_feeds_and_tracks_matching_zips() -> None:
    client = BroadcastifyClient()
    responses = {
        "75201": [
            FeedSearchResult("90001", "Dallas Police", listeners=20),
            FeedSearchResult("100", "Dallas Fire", listeners=10),
        ],
        "75202": [FeedSearchResult("90001", "Dallas Police", listeners=20)],
    }
    client.feeds_for_zip = lambda query: responses[query]  # type: ignore[method-assign]

    results = client.search_area_feeds(["75201", "75202", "75201"])

    assert [value["feed_id"] for value in results] == ["90001", "100"]
    assert results[0]["matched_zip_codes"] == ["75201", "75202"]


def test_area_search_rejects_non_zip_queries() -> None:
    client = BroadcastifyClient()
    try:
        client.search_area_feeds(["Dallas"])
    except ValueError as exc:
        assert "five digits" in str(exc)
    else:
        raise AssertionError("Area searches should validate ZIP codes before network access.")


def test_parse_current_archive_payload() -> None:
    payload = {
        "archives": [
            {"id": 12345, "start": "12:00 AM", "duration": "30 min"},
            {"id": "67890", "start": "12:30 AM", "duration": "30 min"},
            {"start": "1:00 AM"},
        ]
    }

    assert BroadcastifyClient.parse_archive_payload(payload) == ["12345", "67890"]


def test_parse_archive_filename_prefix_uses_feed_timezone() -> None:
    payload = {
        "timezone": "America/Chicago",
        "archives": [{"id": "90003-1783140752", "startTs": 1783140752}],
    }

    assert BroadcastifyClient.parse_archive_filename_prefixes(payload) == {
        "90003-1783140752": "202607032352"
    }


def test_parse_archive_payload_rejects_old_shape() -> None:
    try:
        BroadcastifyClient.parse_archive_payload({"data": [[12345]]})
    except TypeError:
        pass
    else:
        raise AssertionError("The retired archive payload shape should not be accepted.")


def test_existing_archive_is_reused_without_download(tmp_path: Path) -> None:
    archive = tmp_path / "202607120000-12345-90001.mp3"
    archive.write_bytes(b"audio")
    assert BroadcastifyClient._existing_archive(tmp_path, "90001", "12345") == archive


def test_existing_archive_matches_current_epoch_url_identifier(tmp_path: Path) -> None:
    archive = tmp_path / "202607032352-672850-90003.mp3"
    archive.write_bytes(b"audio")
    current_identifier = "90003-1783140752"

    assert (
        BroadcastifyClient._existing_archive(
            tmp_path, "90003", current_identifier, "202607032352"
        )
        == archive
    )
