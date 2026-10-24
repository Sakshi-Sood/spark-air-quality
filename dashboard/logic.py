"""Pure pandas logic used by the dashboard pages (no Streamlit, easy to test).

Everything here works on the small precomputed analytical tables; no Spark.
"""
import pandas as pd

from src.config import ALL_YEAR, SEASON_ORDER

EXPERIMENTS = {
    "E1": "Shuffle partitions (AQE off)",
    "E2": "Input partitioning (repartition)",
    "E3": "Caching vs no caching",
    "E4": "CSV vs Parquet, inferred vs explicit schema",
    "E5": "Broadcast vs sort-merge join",
    "E6": "Cores: local[1] / local[2] / local[4]",
    "E7": "Input size, real subsets (limited)",
}


def filter_places(df: pd.DataFrame, regions: list[str], cities: list[str]) -> pd.DataFrame:
    """Keep rows for the selected regions and cities; an empty list means 'all'."""
    out = df
    if regions and "region" in out:
        out = out[out["region"].isin(regions)]
    if cities and "city" in out:
        out = out[out["city"].isin(cities)]
    return out


def assign_stable_colours(selected: list[str], previous: dict[str, str],
                          palette: list[str]) -> dict[str, str]:
    """Give each selected city a colour that does not change when others are removed.

    Cities keep the colour they already had; new cities take the first free
    slot. At most len(palette) cities can be shown.
    """
    kept = {c: previous[c] for c in selected if c in previous}
    free = [colour for colour in palette if colour not in kept.values()]
    for city in selected:
        if city not in kept and free:
            kept[city] = free.pop(0)
    return kept


def correlation_strength(rho: float | None) -> str:
    """Plain-language size of a correlation coefficient."""
    if rho is None or pd.isna(rho):
        return "not available"
    size = abs(rho)
    word = "negligible" if size < 0.1 else "weak" if size < 0.3 else "moderate" if size < 0.5 else "strong"
    direction = "" if word == "negligible" else (" positive" if rho > 0 else " negative")
    return word + direction


def season_means(seasonal: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Average over complete season-years (the seasonal table holds only those)."""
    subset = seasonal[seasonal["pollutant"] == pollutant]
    out = (subset.groupby("season", as_index=False)
           .agg(mean=("mean", "mean"), season_years=("year", "nunique")))
    out["season"] = pd.Categorical(out["season"], SEASON_ORDER, ordered=True)
    return out.sort_values("season")


def key_findings(city_stats: pd.DataFrame, region_stats: pd.DataFrame,
                 seasonal: pd.DataFrame, bins: pd.DataFrame,
                 pollutant: str = "pm2_5_mean", unit: str = "µg/m³") -> list[str]:
    """Short findings computed from the analytical tables (never hardcoded)."""
    findings = []
    cities = city_stats[city_stats["pollutant"] == pollutant].sort_values("mean")
    if not cities.empty:
        hi, lo = cities.iloc[-1], cities.iloc[0]
        findings.append(f"Highest mean modelled PM2.5: **{hi.city}** ({hi['mean']:.1f} {unit}); "
                        f"lowest: **{lo.city}** ({lo['mean']:.1f} {unit}).")
        above, days = cities["days_above_naaqs"].sum(), cities["n_days"].sum()
        if days:
            findings.append(f"**{100 * above / days:.1f} %** of all city-days have a modelled "
                            f"PM2.5 24-h mean above the NAAQS value (not a compliance result).")
    regions = region_stats[region_stats["pollutant"] == pollutant].sort_values("mean")
    if not regions.empty:
        hi, lo = regions.iloc[-1], regions.iloc[0]
        findings.append(f"Highest regional mean: **{hi.region}** ({hi['mean']:.1f} {unit}, "
                        f"{hi.n_cities} cities); lowest: **{lo.region}** ({lo['mean']:.1f} {unit}).")
    seasons = season_means(seasonal, pollutant)
    if not seasons.empty:
        hi = seasons.loc[seasons["mean"].idxmax()]
        lo = seasons.loc[seasons["mean"].idxmin()]
        findings.append(f"Across complete seasons, **{hi.season}** has the highest mean "
                        f"({hi['mean']:.1f} {unit}) and **{lo.season}** the lowest ({lo['mean']:.1f}).")
    wind = bins[(bins["scope_type"] == "all") & (bins["season_group"] == ALL_YEAR)
                & (bins["weather_variable"] == "wind_speed_10m_max")
                & (bins["pollutant"] == pollutant)].sort_values("bin_order")
    if len(wind) >= 2:
        calm, windy = wind.iloc[0], wind.iloc[-1]
        findings.append(f"Mean PM2.5 is {calm['mean']:.1f} {unit} on days with wind {calm.bin_label} "
                        f"and {windy['mean']:.1f} on days with wind {windy.bin_label} "
                        f"(an association, not proof of cause).")
    return findings


def experiment_status(results: pd.DataFrame | None) -> pd.DataFrame:
    """One row per planned experiment: 'Measured' only if results exist for it."""
    measured = set(results["experiment"]) if results is not None and "experiment" in results else set()
    return pd.DataFrame([
        {"experiment": eid, "description": desc,
         "status": "Measured" if eid in measured else "Pending"}
        for eid, desc in EXPERIMENTS.items()
    ])
