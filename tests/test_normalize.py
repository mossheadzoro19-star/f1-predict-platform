from f1_predict.ingestion.normalize import (
    normalize_entities,
    normalize_qualifying,
    normalize_races,
    normalize_results,
)


def _payload():
    races = [
        {
            "season": "2025",
            "round": "1",
            "raceName": "Test Grand Prix",
            "Circuit": {
                "circuitId": "test",
                "circuitName": "Test Circuit",
                "Location": {"lat": "12.3", "long": "45.6", "locality": "Test City", "country": "Testland"},
            },
            "date": "2025-03-01",
            "time": "12:00:00Z",
        }
    ]
    result = {
        "number": "44",
        "position": "1",
        "positionText": "1",
        "points": "25",
        "grid": "1",
        "laps": "57",
        "status": "Finished",
        "Driver": {
            "driverId": "hamilton",
            "givenName": "Lewis",
            "familyName": "Hamilton",
            "code": "HAM",
            "permanentNumber": "44",
            "nationality": "British",
        },
        "Constructor": {"constructorId": "ferrari", "name": "Ferrari", "nationality": "Italian"},
        "Time": {"millis": "5000", "time": "1:23:45.000"},
        "FastestLap": {
            "rank": "1",
            "lap": "42",
            "Time": {"millis": "90000", "time": "1:30.000"},
            "AverageSpeed": {"units": "kph", "speed": "210.5"},
        },
    }
    qualifying = {
        "number": "44",
        "position": "1",
        "Driver": result["Driver"],
        "Constructor": result["Constructor"],
        "Q1": "1:20.000",
        "Q2": "1:19.000",
        "Q3": "1:18.000",
    }
    return (
        {"MRData": {"RaceTable": {"Races": [{**races[0], "Results": [result]}]}}},
        {"MRData": {"RaceTable": {"Races": [{**races[0], "QualifyingResults": [qualifying]}]}}},
    )


def test_normalize_races():
    df = normalize_races(_payload()[0])
    assert len(df) == 1
    assert df.iloc[0]["round"] == 1
    assert df.iloc[0]["circuit_id"] == "test"


def test_normalize_results():
    df = normalize_results(_payload()[0])
    row = df.iloc[0]
    assert len(df) == 1
    assert row["driver_id"] == "hamilton"
    assert row["position"] == 1
    assert row["fastest_lap_speed"] == 210.5


def test_normalize_qualifying():
    df = normalize_qualifying(_payload()[1])
    assert len(df) == 1
    assert df.iloc[0]["q3"] == "1:18.000"


def test_normalize_entities_uses_stable_ids():
    results, qualifying = _payload()
    drivers, constructors = normalize_entities(results, qualifying)
    assert list(drivers["driver_id"]) == ["hamilton"]
    assert list(constructors["constructor_id"]) == ["ferrari"]
