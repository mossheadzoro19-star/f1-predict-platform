from f1_predict.data_sources.jolpica import JolpicaClient
from f1_predict.data_sources.openf1 import OpenF1Client


def test_jolpica_default_url() -> None:
    assert JolpicaClient().base_url == "https://api.jolpi.ca/ergast/f1"


def test_jolpica_requests_all_season_results() -> None:
    client = JolpicaClient()
    assert client.get_season_results.__doc__ == "Return all driver race results for a season."


def test_jolpica_requests_all_season_qualifying() -> None:
    client = JolpicaClient()
    assert client.get_season_qualifying.__doc__ == "Return all qualifying results for a season."


def test_openf1_default_url() -> None:
    assert OpenF1Client().base_url == "https://api.openf1.org/v1"
