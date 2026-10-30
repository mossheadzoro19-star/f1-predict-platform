import json

import pytest

from f1_predict.ingestion.jolpica_archive import (
    JolpicaArchive,
    count_records,
    validate_cached_payload,
)


def make_payload(year=2024, round_number=1, endpoint="laps"):
    race = {
        "season": str(year),
        "round": str(round_number),
    }

    if endpoint == "laps":
        race["Laps"] = [
            {
                "number": "1",
                "Timings": [
                    {"driverId": "a", "time": "1:30.0"},
                    {"driverId": "b", "time": "1:31.0"},
                ],
            }
        ]
    else:
        race["PitStops"] = [
            {"driverId": "a", "lap": "10"},
            {"driverId": "b", "lap": "12"},
        ]

    return {"MRData": {"RaceTable": {"Races": [race]}}}


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_race_laps(self, year, round_number):
        self.calls.append(("laps", year, round_number))
        return make_payload(year, round_number, "laps")

    def get_race_pitstops(self, year, round_number):
        self.calls.append(("pitstops", year, round_number))
        return make_payload(year, round_number, "pitstops")


def test_count_records_for_laps():
    assert count_records("laps", make_payload()) == 2


def test_count_records_for_pitstops():
    payload = make_payload(endpoint="pitstops")
    assert count_records("pitstops", payload) == 2


def test_cache_requires_matching_season_and_round(tmp_path):
    path = tmp_path / "laps.json"
    path.write_text(json.dumps(make_payload(2024, 1)))

    assert validate_cached_payload(path, "laps", 2024, 1) is not None
    assert validate_cached_payload(path, "laps", 2024, 2) is None
    assert validate_cached_payload(path, "laps", 2023, 1) is None


def test_corrupt_cache_is_not_accepted(tmp_path):
    path = tmp_path / "laps.json"
    path.write_text("{not valid json")

    assert validate_cached_payload(path, "laps", 2024, 1) is None


def test_ingestion_writes_payloads_and_manifest(tmp_path):
    client = FakeClient()
    archive = JolpicaArchive(
        root=tmp_path,
        client=client,
        pause_seconds=0,
    )

    archive.ingest_race(2024, 1)

    assert (tmp_path / "raw/2024/round_01/laps.json").is_file()
    assert (tmp_path / "raw/2024/round_01/pitstops.json").is_file()

    manifest = json.loads(
        (tmp_path / "manifests/ingestion_manifest.json").read_text()
    )
    entry = manifest["races"]["2024-01"]

    assert entry["endpoints"]["laps"]["status"] == "complete"
    assert entry["endpoints"]["pitstops"]["status"] == "complete"
    assert entry["endpoints"]["laps"]["records"] == 2
    assert entry["endpoints"]["pitstops"]["records"] == 2


def test_completed_endpoints_are_reused_from_cache(tmp_path):
    client = FakeClient()
    archive = JolpicaArchive(
        root=tmp_path,
        client=client,
        pause_seconds=0,
    )

    archive.ingest_race(2024, 1)
    first_call_count = len(client.calls)

    restarted = JolpicaArchive(
        root=tmp_path,
        client=client,
        pause_seconds=0,
    )
    restarted.ingest_race(2024, 1)

    assert len(client.calls) == first_call_count


def test_failed_endpoint_is_not_marked_complete(tmp_path):
    class BrokenClient(FakeClient):
        def get_race_laps(self, year, round_number):
            raise RuntimeError("simulated network failure")

    archive = JolpicaArchive(
        root=tmp_path,
        client=BrokenClient(),
        pause_seconds=0,
    )

    with pytest.raises(RuntimeError, match="simulated network failure"):
        archive.ingest_race(2024, 1)

    manifest = json.loads(
        (tmp_path / "manifests/ingestion_manifest.json").read_text()
    )

    assert manifest["races"]["2024-01"]["endpoints"]["laps"]["status"] == "failed"
