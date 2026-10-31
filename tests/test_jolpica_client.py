from f1_predict.data_sources.jolpica import JolpicaClient


def test_results_pagination_counts_driver_records():
    pages = [
        {"Results": [{"Driver": {"driverId": "a"}}]},
        {"Results": [
            {"Driver": {"driverId": "b"}},
            {"Driver": {"driverId": "c"}},
        ]},
    ]

    assert JolpicaClient._page_item_count(
        "2024/results/", pages
    ) == 3


def test_qualifying_pagination_counts_driver_records():
    pages = [
        {"QualifyingResults": [{"Driver": {"driverId": "a"}}]},
        {"QualifyingResults": [
            {"Driver": {"driverId": "b"}},
        ]},
    ]

    assert JolpicaClient._page_item_count(
        "2019/qualifying/", pages
    ) == 2


def test_lap_pagination_counts_nested_timing_records():
    pages = [
        {"Laps": [
            {"number": "1", "Timings": [{"driverId": "a"}]},
            {"number": "2", "Timings": [{"driverId": "a"}]},
        ]},
        {"Laps": [
            {"number": "2", "Timings": [{"driverId": "b"}]},
        ]},
    ]

    assert JolpicaClient._page_item_count(
        "2024/1/laps/", pages
    ) == 3


def test_pitstop_pagination_counts_nested_records():
    pages = [
        {"PitStops": [{"driverId": "a"}]},
        {"PitStops": [
            {"driverId": "b"},
            {"driverId": "c"},
        ]},
    ]

    assert JolpicaClient._page_item_count(
        "2024/1/pitstops/", pages
    ) == 3


def test_results_pages_merge_into_one_race():
    pages = [
        {
            "round": "6",
            "raceName": "Miami Grand Prix",
            "Results": [
                {"Driver": {"driverId": "norris"}}
            ],
        },
        {
            "round": "6",
            "raceName": "Miami Grand Prix",
            "Results": [
                {"Driver": {"driverId": "verstappen"}},
                {"Driver": {"driverId": "leclerc"}},
            ],
        },
    ]

    merged = JolpicaClient._merge_race_pages(
        "2024/results/", pages
    )

    assert len(merged) == 1
    assert len(merged[0]["Results"]) == 3

    drivers = {
        row["Driver"]["driverId"]
        for row in merged[0]["Results"]
    }

    assert drivers == {"norris", "verstappen", "leclerc"}


def test_lap_pages_merge_split_timing_records():
    pages = [
        {
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Laps": [
                {
                    "number": "1",
                    "Timings": [{"driverId": "a", "time": "1:30.0"}],
                }
            ],
        },
        {
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Laps": [
                {
                    "number": "1",
                    "Timings": [{"driverId": "b", "time": "1:31.0"}],
                },
                {
                    "number": "2",
                    "Timings": [{"driverId": "a", "time": "1:32.0"}],
                },
            ],
        },
    ]

    merged = JolpicaClient._merge_race_pages(
        "2024/1/laps/", pages
    )

    assert len(merged) == 1
    assert len(merged[0]["Laps"]) == 2

    lap_one = merged[0]["Laps"][0]
    assert lap_one["number"] == "1"
    assert {
        timing["driverId"] for timing in lap_one["Timings"]
    } == {"a", "b"}

    assert merged[0]["Laps"][1]["number"] == "2"
