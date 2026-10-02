from __future__ import annotations

from unittest.mock import Mock

from f1_predict.data_sources.jolpica import JolpicaClient
from f1_predict.data_sources.openf1 import OpenF1Client


def test_jolpica_default_url() -> None:
    assert JolpicaClient().base_url == "https://api.jolpi.ca/ergast/f1"


def test_jolpica_page_item_count_results() -> None:
    races = [{"Results": [{"number": "1"}, {"number": "2"}]}, {"Results": [{"number": "3"}]}]
    assert JolpicaClient._page_item_count("2025/results/", races) == 3


def test_jolpica_page_item_count_qualifying() -> None:
    races = [{"QualifyingResults": [{"number": "1"}, {"number": "2"}]}]
    assert JolpicaClient._page_item_count("2025/qualifying/", races) == 2


def test_jolpica_retries_rate_limit() -> None:
    client = JolpicaClient(request_delay_seconds=0, max_retries=1)
    response_429 = Mock(status_code=429, headers={}, raise_for_status=Mock())
    response_ok = Mock(
        status_code=200,
        headers={},
        raise_for_status=Mock(),
        json=Mock(return_value={"MRData": {}}),
    )

    http_client = Mock()
    http_client.get.side_effect = [response_429, response_ok]

    result = client._get_json(http_client, "2025/results/", limit=100, offset=0)

    assert result == {"MRData": {}}
    assert http_client.get.call_count == 2


def test_openf1_default_url() -> None:
    assert OpenF1Client().base_url == "https://api.openf1.org/v1"
