from f1_predict.data_sources.jolpica import JolpicaClient
from f1_predict.data_sources.openf1 import OpenF1Client


def test_jolpica_default_url() -> None:
    assert JolpicaClient().base_url.startswith("https://api.jolpi.ca/")


def test_openf1_default_url() -> None:
    assert OpenF1Client().base_url == "https://api.openf1.org/v1"
