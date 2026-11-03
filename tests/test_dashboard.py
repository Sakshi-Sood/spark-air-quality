"""Dashboard tests: pure logic, plus headless page runs with Streamlit's AppTest.

AppTest runs app.py in-process (no browser, no Spark) against the real
outputs/ folder, so these tests are skipped until the pipeline has run.
"""
import pandas as pd
import pytest

from dashboard.logic import (assign_stable_colours, bin_scope, correlation_strength,
                             experiment_status, filter_places, key_findings,
                             season_means)
from src.config import ANALYTICS_DIR, PROCESSED_DIR, PROJECT_ROOT

PAGES = {name: str(PROJECT_ROOT / "dashboard" / "views" / f"{name}.py")
         for name in ("overview", "city_region", "weather", "temporal", "pm2_5_forecasting", "performance")}


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


FINDING_CITIES = pd.DataFrame({"pollutant": "pm2_5_mean", "city": ["X", "Y", "Z"],
                               "region": ["R1", "R1", "R2"], "mean": [80.0, 10.0, 30.0],
                               "days_above_naaqs": [50, 0, 10], "n_days": [100, 100, 100]})
FINDING_REGIONS = pd.DataFrame({"pollutant": "pm2_5_mean", "region": ["R1", "R2"],
                                "mean": [45.0, 30.0], "n_cities": [2, 1]})
EMPTY = pd.DataFrame(columns=["pollutant", "city", "region", "year", "season", "mean", "scope_type",
                              "scope", "season_group", "weather_variable", "bin_order"])


def test_key_findings_are_computed_from_tables():
    findings = key_findings(FINDING_CITIES, FINDING_REGIONS, EMPTY, EMPTY)
    assert "**X** (80.0 µg/m³)" in findings[0] and "**Y**" in findings[0]
    assert "**20.0 %**" in findings[1]                     # 60 of 300 city-days
    assert "**R1**" in findings[2]
    assert len(findings) == 3           # no seasonal or bin rows -> those findings are skipped


def test_key_findings_follow_the_filters():
    findings = key_findings(FINDING_CITIES, FINDING_REGIONS, EMPTY, EMPTY, regions=["R2"])
    assert findings[0].startswith("**Z**")                 # single city in view
    assert "**10.0 %**" in findings[1]
    assert not any("regional mean" in f for f in findings)  # one region: no comparison
    no_limit = FINDING_CITIES.assign(days_above_naaqs=None)
    assert not any("NAAQS" in f for f in key_findings(no_limit, FINDING_REGIONS, EMPTY, EMPTY))


@pytest.mark.parametrize("regions, cities, scope", [
    ([], [], ("all", "All cities")), (["North"], [], ("region", "North")),
    ([], ["Delhi"], ("city", "Delhi")), (["North"], ["Delhi", "Agra"], ("region", "North")),
    (["North", "South"], [], ("all", "All cities"))])
def test_bin_scope_matches_filters(regions, cities, scope):
    assert bin_scope(regions, cities) == scope


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


def test_experiment_chart_uses_median_and_range_per_variant():
    from dashboard.charts import experiment_comparison
    runs = pd.DataFrame({"experiment": ["E1"] * 6 + ["E2"],   # test input only, never displayed
                         "variant": ["8", "8", "8", "200", "200", "200", "4"],
                         "seconds": [1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 1.0]})
    bar = experiment_comparison(runs, "E1").data[0]
    assert list(bar.x) == ["8", "200"] and list(bar.y) == [2.0, 6.0]
    assert list(bar.error_y.array) == [2.0, 3.0] and list(bar.error_y.arrayminus) == [1.0, 1.0]


@needs_outputs
def test_overview_kpis_and_findings_follow_region_filter(app):
    at = _open(app, "overview")
    values = {m.label: m.value for m in at.metric}
    assert values["Cities in view"] == "50 of 50"
    at.sidebar.multiselect(key="regions").set_value(["South"]).run()
    values = {m.label: m.value for m in at.metric}
    assert values["Cities in view"] == "13 of 50"
    findings = " ".join(md.value for md in at.markdown if md.value.startswith("- "))
    assert "13 selected cities" in " ".join(c.value for c in at.caption)
    assert "Delhi" not in findings          # Delhi is not in the South region


def test_dark_theme_and_charts_agree():
    """The theme is dark, and charts/cards use the dark tokens, not light-only colours."""
    import tomllib
    from dashboard.charts import INK, ranking_bar
    from dashboard.labels import SURFACE
    from dashboard.ui import CARD_CSS
    theme = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert theme["base"] == "dark"
    assert SURFACE in CARD_CSS and "#ffffff" not in CARD_CSS
    df = pd.DataFrame({"label": ["A", "B"], "mean": [1.0, 2.0], "median": [1.0, 2.0], "p95": [1.0, 2.0]})
    fig = ranking_bar(df, "label", "x", "u")
    assert fig.layout.font.color == INK == theme["textColor"]
    assert fig.layout.hoverlabel.bgcolor != "white"
