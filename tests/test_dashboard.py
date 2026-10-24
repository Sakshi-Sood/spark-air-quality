"""Dashboard tests: pure logic, plus headless page runs with Streamlit's AppTest.

AppTest runs app.py in-process (no browser, no Spark) against the real
outputs/ folder, so these tests are skipped until the pipeline has run.
"""
import pandas as pd
import pytest

from dashboard.logic import (assign_stable_colours, correlation_strength,
                             experiment_status, filter_places, key_findings,
                             season_means)
from src.config import ANALYTICS_DIR, PROCESSED_DIR, PROJECT_ROOT

PAGES = {name: str(PROJECT_ROOT / "dashboard" / "views" / f"{name}.py")
         for name in ("overview", "city_region", "weather", "temporal", "performance")}


# ---------------------------------------------------------------- pure logic

def test_filter_places_treats_empty_lists_as_all():
    df = pd.DataFrame({"city": ["A", "B", "C"], "region": ["N", "N", "S"]})
    assert len(filter_places(df, [], [])) == 3
    assert list(filter_places(df, ["N"], [])["city"]) == ["A", "B"]
    assert list(filter_places(df, ["N"], ["B", "C"])["city"]) == ["B"]


def test_stable_colours_survive_removal_and_reuse_free_slots():
    palette = ["c1", "c2", "c3"]
    first = assign_stable_colours(["A", "B", "C"], {}, palette)
    assert first == {"A": "c1", "B": "c2", "C": "c3"}
    after_removal = assign_stable_colours(["A", "C"], first, palette)
    assert after_removal == {"A": "c1", "C": "c3"}          # C keeps its colour
    assert assign_stable_colours(["A", "C", "D"], after_removal, palette)["D"] == "c2"
    assert len(assign_stable_colours(list("ABCD"), {}, palette)) == 3   # capped


@pytest.mark.parametrize("rho, text", [(None, "not available"), (float("nan"), "not available"),
                                       (0.05, "negligible"), (-0.2, "weak negative"),
                                       (0.45, "moderate positive"), (-0.7, "strong negative")])
def test_correlation_strength_wording(rho, text):
    assert correlation_strength(rho) == text


def test_season_means_ordered_by_imd_season():
    seasonal = pd.DataFrame({"pollutant": ["p"] * 4, "year": [2023, 2024, 2023, 2023],
                             "season": ["Winter", "Winter", "Monsoon", "Pre-monsoon"],
                             "mean": [50.0, 60.0, 20.0, 30.0]})
    out = season_means(seasonal, "p")
    assert list(out["season"]) == ["Winter", "Pre-monsoon", "Monsoon"]
    assert out.iloc[0]["mean"] == 55.0 and out.iloc[0]["season_years"] == 2


def test_key_findings_are_computed_from_tables():
    cities = pd.DataFrame({"pollutant": "pm2_5_mean", "city": ["X", "Y"], "mean": [80.0, 10.0],
                           "days_above_naaqs": [50, 0], "n_days": [100, 100]})
    regions = pd.DataFrame({"pollutant": "pm2_5_mean", "region": ["R1", "R2"],
                            "mean": [40.0, 20.0], "n_cities": [3, 2]})
    empty = pd.DataFrame(columns=["pollutant", "year", "season", "mean", "scope_type",
                                  "season_group", "weather_variable", "bin_order"])
    findings = key_findings(cities, regions, empty, empty)
    assert "**X** (80.0 µg/m³)" in findings[0] and "**Y**" in findings[0]
    assert "**25.0 %**" in findings[1]
    assert "**R1**" in findings[2]
    assert len(findings) == 3           # no seasonal or bin rows -> those findings are skipped


def test_experiments_pending_until_measured():
    assert set(experiment_status(None)["status"]) == {"Pending"}
    status = experiment_status(pd.DataFrame({"experiment": ["E1", "E1", "E3"]}))
    assert dict(zip(status["experiment"], status["status"]))["E3"] == "Measured"
    assert (status["status"] == "Pending").sum() == 5


# ---------------------------------------------------------------- headless page runs

needs_outputs = pytest.mark.skipif(
    not (ANALYTICS_DIR / "weather_correlations").exists() or not (PROCESSED_DIR / "daily_joined").exists(),
    reason="pipeline outputs not generated")


@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


def _open(at, page):
    at.switch_page(PAGES[page])
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@needs_outputs
@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_renders_with_default_filters(app, page):
    at = _open(app, page)
    assert not at.error


@needs_outputs
def test_global_filters_apply_across_pages(app):
    app.sidebar.selectbox(key="pollutant").set_value("india_aqi").run()
    app.sidebar.multiselect(key="regions").set_value(["Northeast"]).run()
    for page in PAGES:
        at = _open(app, page)
        assert not at.error
    # The ozone caveat is shown wherever the all-pollutant AQI is selected.
    assert any("ozone" in w.value for w in app.sidebar.warning)


@needs_outputs
def test_region_change_drops_cities_outside_it(app):
    app.sidebar.multiselect(key="cities").set_value(["Delhi", "Chennai"]).run()
    app.sidebar.multiselect(key="regions").set_value(["South"]).run()
    assert app.sidebar.multiselect(key="cities").value == ["Chennai"]


@needs_outputs
def test_weather_page_filters(app):
    at = _open(app, "weather")
    for scope in ("region", "all", "city"):
        at.selectbox(key="w_scope_type").set_value(scope).run()
        assert not at.exception
    at.selectbox(key="w_season").set_value("Monsoon").run()
    at.selectbox(key="w_var").set_value("precipitation_sum").run()
    at.radio(key="w_metric").set_value("pearson_r").run()
    assert not at.exception and not at.error
    # A pollutant without weather results falls back to PM2.5 with a note.
    at.sidebar.selectbox(key="pollutant").set_value("so2_mean").run()
    assert any("not analysed on this page" in i.value for i in at.info)


@needs_outputs
def test_temporal_page_handles_empty_episode_filter(app):
    app.sidebar.multiselect(key="cities").set_value(["Leh"]).run()
    at = _open(app, "temporal")
    at.slider(key="t_min_len").set_value(30).run()
    assert any("No episodes match" in i.value for i in at.info)


@needs_outputs
def test_missing_output_shows_message_not_traceback(app, monkeypatch, tmp_path):
    import dashboard.data_loader as loader
    monkeypatch.setattr(loader, "ANALYTICS_DIR", tmp_path)
    loader.load_table.clear()
    at = _open(app, "overview")
    assert any("Required pipeline output is missing" in e.value for e in at.error)
    loader.load_table.clear()
