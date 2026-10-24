"""Validate the three raw CSVs with Spark and produce a JSON report.

Severity levels used in the report:
  critical - the data cannot be processed safely; the pipeline stops.
  warning  - processing can continue, but the issue is recorded.
  info     - a documented observation (e.g. a known boundary effect).

Spark notes:
  * Each check is an *action* (count, first, collect), and every action runs
    a job. Without caching, every job would re-read and re-parse the CSV, so
    each raw DataFrame is cached for the duration of validation.
  * Many checks are computed in ONE pass by putting several aggregate
    expressions into a single select() instead of running one job per column.
"""
from datetime import datetime
from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import (AQI_BANDS, CITY_KEY, DAILY_KEY, DATASET_REPO,
                        DATASET_REVISION, DOMINANT_POLLUTANTS,
                        MIN_HOURS_AVAILABLE, NAAQS_24H, POLLUTANT_UNITS,
                        RAW_FILES, WEATHER_UNITS)
from src.schemas import CORRUPT_COLUMN, SCHEMAS, read_raw_csv
from src.spark_session import describe_environment
from src.utils import get_logger, sha256_of_file

log = get_logger(__name__)

CRITICAL, WARNING, INFO = "critical", "warning", "info"
DATASET_KEYS = {"air_quality": DAILY_KEY, "weather": DAILY_KEY, "cities": CITY_KEY}


def make_check(name: str, dataset: str, severity: str, passed: bool, details) -> dict:
    return {"check": name, "dataset": dataset, "severity": severity,
            "passed": bool(passed), "details": details}


# ---------------------------------------------------------------- file-level checks (plain Python)

def read_csv_header(path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return handle.readline().strip().split(",")


def check_header(name: str, path) -> dict:
    """Compare the CSV header with the expected schema column names."""
    expected = SCHEMAS[name].fieldNames()
    actual = read_csv_header(path)
    details = {
        "missing_columns": [c for c in expected if c not in actual],
        "unexpected_columns": [c for c in actual if c not in expected],
        "same_order": actual == expected,
    }
    return make_check("header_matches_schema", name, CRITICAL, actual == expected, details)


# ---------------------------------------------------------------- Spark checks

def count_nulls(df: DataFrame, columns: list[str]) -> dict:
    """Null count for every column, computed in a single job."""
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns]).first()
    return {c: int(row[c] or 0) for c in columns}


def count_duplicates(df: DataFrame, key: list[str]) -> dict:
    """Separate harmless exact copies from key duplicates with conflicting values.

    duplicate_rows        rows that are exact copies of another row
    duplicate_key_rows    rows whose key already appeared
    conflicting_key_rows  key duplicates whose values differ (ambiguous data)
    """
    total = df.count()
    distinct_rows = df.distinct().count()
    distinct_keys = df.select(*key).distinct().count()
    return {
        "rows": total,
        "duplicate_rows": total - distinct_rows,
        "duplicate_key_rows": total - distinct_keys,
        "conflicting_key_rows": distinct_rows - distinct_keys,
    }


def count_violations(df: DataFrame, rules: list[tuple[str, str, Column]]) -> dict:
    """Count rows violating each rule; all rules are evaluated in one pass.

    A rule whose condition is NULL (because an input is null) is not counted
    as a violation; nulls are reported separately by count_nulls.
    """
    exprs = [F.sum(F.when(cond, 1).otherwise(0)).alias(name) for name, _, cond in rules]
    row = df.select(exprs).first()
    return {name: int(row[name] or 0) for name, _, _ in rules}


def aqi_category_expr(aqi_col: str) -> Column:
    """Map an AQI value to its CPCB category using AQI_BANDS."""
    expr = None
    for upper, label in AQI_BANDS:
        cond = F.col(aqi_col) <= upper
        expr = F.when(cond, label) if expr is None else expr.when(cond, label)
    return expr


def _any(conditions: list[Column]) -> Column:
    return reduce(lambda a, b: a | b, conditions)


def range_rules(name: str) -> list[tuple[str, str, Column]]:
    """Plausibility rules per dataset: (rule name, severity, violation condition)."""
    col = F.col
    if name == "air_quality":
        return [
            ("negative_concentration", CRITICAL, _any([col(c) < 0 for c in POLLUTANT_UNITS])),
            ("pm2_5_max_below_mean", WARNING, col("pm2_5_max") < col("pm2_5_mean")),
            ("pm10_max_below_mean", WARNING, col("pm10_max") < col("pm10_mean")),
            ("pm2_5_above_pm10", WARNING, col("pm2_5_mean") > col("pm10_mean")),
            ("hours_available_out_of_range", WARNING,
             ~col("hours_available").between(MIN_HOURS_AVAILABLE, 24)),
            ("india_aqi_out_of_range", CRITICAL, ~col("india_aqi").between(0, 500)),
            ("india_aqi_pm_out_of_range", CRITICAL, ~col("india_aqi_pm").between(0, 500)),
            ("india_aqi_category_mismatch", WARNING,
             aqi_category_expr("india_aqi") != col("india_aqi_category")),
            ("india_aqi_pm_category_mismatch", WARNING,
             aqi_category_expr("india_aqi_pm") != col("india_aqi_pm_category")),
            ("unknown_dominant_pollutant", WARNING,
             ~col("dominant_pollutant").isin(DOMINANT_POLLUTANTS)),
        ]
    if name == "weather":
        temps = ["temperature_2m_max", "temperature_2m_min",
                 "temperature_2m_mean", "apparent_temperature_max"]
        non_negative = ["precipitation_sum", "wind_speed_10m_max",
                        "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
        return [
            ("humidity_out_of_range", CRITICAL,
             ~col("relative_humidity_2m_mean").between(0, 100)),
            ("wind_direction_out_of_range", CRITICAL,
             ~col("wind_direction_10m_dominant").between(0, 360)),
            ("negative_physical_quantity", CRITICAL, _any([col(c) < 0 for c in non_negative])),
            ("precipitation_hours_out_of_range", CRITICAL,
             ~col("precipitation_hours").between(0, 24)),
            ("temperature_order_violated", WARNING,
             (col("temperature_2m_min") > col("temperature_2m_mean"))
             | (col("temperature_2m_mean") > col("temperature_2m_max"))),
            ("temperature_implausible", WARNING, _any([~col(c).between(-60, 60) for c in temps])),
        ]
    if name == "cities":
        return [
            ("blank_city_name", CRITICAL, F.trim(col("city")) == ""),
            ("coordinates_outside_india", WARNING,
             ~col("latitude").between(6, 38) | ~col("longitude").between(68, 98)),
        ]
    raise ValueError(f"Unknown dataset: {name}")


def date_coverage(df: DataFrame) -> dict:
    """Per-city date coverage, summarised. Missing days = expected - observed."""
    per_city = (
        df.groupBy("city")
        .agg(F.min("date").alias("first"), F.max("date").alias("last"),
             F.countDistinct("date").alias("days"))
        .withColumn("missing", F.datediff("last", "first") + 1 - F.col("days"))
    )
    s = per_city.agg(
        F.count("*").alias("cities"),
        F.min("first").alias("first_date"), F.max("first").alias("latest_first_date"),
        F.min("last").alias("earliest_last_date"), F.max("last").alias("last_date"),
        F.min("days").alias("min_days_per_city"), F.max("days").alias("max_days_per_city"),
        F.sum("missing").alias("missing_days_total"),
        F.sum((F.col("missing") > 0).cast("int")).alias("cities_with_gaps"),
    ).first().asDict()
    s["same_range_all_cities"] = (s["first_date"] == s["latest_first_date"]
                                  and s["earliest_last_date"] == s["last_date"])
    return s


def unmatched_cities(daily: DataFrame, cities: DataFrame) -> dict:
    """City names present on one side only (left_anti join = rows with no match)."""
    daily_cities = daily.select("city").distinct()
    unknown = daily_cities.join(cities.select("city"), "city", "left_anti")
    unused = cities.select("city").join(daily_cities, "city", "left_anti")
    # Both results are at most 50 rows, so collecting them is safe.
    return {"not_in_cities_csv": sorted(r.city for r in unknown.collect()),
            "without_daily_rows": sorted(r.city for r in unused.collect())}


# ---------------------------------------------------------------- orchestration

def validate_dataset(name: str, df: DataFrame) -> tuple[dict, list[dict]]:
    """Run all single-dataset checks on one cached raw DataFrame."""
    data_cols = SCHEMAS[name].fieldNames()
    key = DATASET_KEYS[name]
    data = df.select(*data_cols)
    checks = []

    # Spark only allows filtering on the corrupt-record column of a cached
    # DataFrame, which is one more reason the raw input is cached.
    corrupt = df.filter(F.col(CORRUPT_COLUMN).isNotNull()).count()
    checks.append(make_check("no_malformed_rows", name, CRITICAL, corrupt == 0,
                             {"malformed_rows": corrupt}))

    nulls = count_nulls(data, data_cols)
    key_nulls = {c: nulls[c] for c in key}
    other_nulls = {c: n for c, n in nulls.items() if c not in key and n > 0}
    checks.append(make_check("no_null_keys", name, CRITICAL, not any(key_nulls.values()), key_nulls))
    checks.append(make_check("no_null_values", name, WARNING, not other_nulls, other_nulls))

    dups = count_duplicates(data, key)
    checks.append(make_check("no_exact_duplicate_rows", name, WARNING,
                             dups["duplicate_rows"] == 0, dups))
    checks.append(make_check("no_conflicting_key_duplicates", name, CRITICAL,
                             dups["conflicting_key_rows"] == 0, dups))

    rules = range_rules(name)
    violations = count_violations(data, rules)
    for rule, severity, _ in rules:
        checks.append(make_check(rule, name, severity, violations[rule] == 0,
                                 {"violating_rows": violations[rule]}))

    summary = {
        "rows": dups["rows"],
        "columns": [{"name": f.name, "type": f.dataType.simpleString()}
                    for f in data.schema.fields],
        "spark_input_partitions": df.rdd.getNumPartitions(),
        "malformed_rows": corrupt,
        "nulls": nulls,
        "duplicates": dups,
        "rule_violations": violations,
    }
    if "date" in data_cols:
        coverage = date_coverage(data)
        summary["date_coverage"] = coverage
        checks.append(make_check("complete_daily_coverage", name, WARNING,
                                 coverage["missing_days_total"] == 0, coverage))
        checks.append(make_check("same_date_range_all_cities", name, WARNING,
                                 coverage["same_range_all_cities"], coverage))
    return summary, checks


def cross_dataset_checks(raw: dict[str, DataFrame], summaries: dict) -> list[dict]:
    """Checks that need more than one dataset."""
    checks = []
    cities = raw["cities"]
    for name in ("air_quality", "weather"):
        result = unmatched_cities(raw[name], cities)
        checks.append(make_check("all_cities_in_metadata", name, CRITICAL,
                                 not result["not_in_cities_csv"], result))
        checks.append(make_check("all_metadata_cities_have_data", name, WARNING,
                                 not result["without_daily_rows"], result))

    aq_cov = summaries["air_quality"]["date_coverage"]
    w_cov = summaries["weather"]["date_coverage"]
    covered = (w_cov["first_date"] <= aq_cov["first_date"]
               and w_cov["last_date"] >= aq_cov["last_date"])
    checks.append(make_check("weather_covers_air_quality_dates", "air_quality+weather",
                             WARNING, covered,
                             {"air_quality": [aq_cov["first_date"], aq_cov["last_date"]],
                              "weather": [w_cov["first_date"], w_cov["last_date"]]}))

    # Documented boundary effect: CPCB breakpoints are whole numbers (PM2.5
    # 61-90, PM10 101-250), so a concentration just above a NAAQS limit can
    # still have india_aqi_pm = 100. Exceedance flags therefore use the
    # concentrations, not the AQI.
    aq = raw["air_quality"]
    above = (F.col("pm2_5_mean") > NAAQS_24H["pm2_5_mean"]) | (F.col("pm10_mean") > NAAQS_24H["pm10_mean"])
    boundary = aq.filter(above & (F.col("india_aqi_pm") <= 100)).count()
    reverse = aq.filter(~above & (F.col("india_aqi_pm") > 100)).count()
    checks.append(make_check("naaqs_vs_india_aqi_pm_consistency", "air_quality", INFO,
                             reverse == 0,
                             {"above_limit_but_aqi_pm_le_100": boundary,
                              "within_limit_but_aqi_pm_gt_100": reverse}))
    return checks


def summarise_checks(checks: list[dict]) -> dict:
    failed = [c for c in checks if not c["passed"]]
    return {
        "checks_run": len(checks),
        "passed": len(checks) - len(failed),
        "critical_failures": [c["check"] + "@" + c["dataset"] for c in failed if c["severity"] == CRITICAL],
        "warnings": [c["check"] + "@" + c["dataset"] for c in failed if c["severity"] == WARNING],
        "info": [c["check"] + "@" + c["dataset"] for c in failed if c["severity"] == INFO],
    }


def validate_all(spark: SparkSession) -> dict:
    """Validate all raw files and return the full report as a dict."""
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "source": {"repo": DATASET_REPO, "revision": DATASET_REVISION},
              "environment": describe_environment(spark),
              "units": {**POLLUTANT_UNITS, **WEATHER_UNITS},
              "datasets": {}, "checks": []}

    raw = {}
    for name, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Raw file missing: {path}. Run download_data.py first.")
        file_info = {"path": str(path.relative_to(path.parents[1])),
                     "bytes": path.stat().st_size, "sha256": sha256_of_file(path)}
        header = check_header(name, path)
        report["checks"].append(header)
        report["datasets"][name] = {"file": file_info}
        if not header["passed"]:
            log.error("Header mismatch in %s: %s", name, header["details"])
            continue
        raw[name] = read_raw_csv(spark, name).cache()

    if len(raw) == len(RAW_FILES):
        for name, df in raw.items():
            log.info("Validating %s ...", name)
            summary, checks = validate_dataset(name, df)
            report["datasets"][name].update(summary)
            report["checks"].extend(checks)
        report["checks"].extend(cross_dataset_checks(raw, report["datasets"]))

    for df in raw.values():
        df.unpersist()
    report["summary"] = summarise_checks(report["checks"])
    return report
