# Distributed Air Quality and Weather Analytics System Using Apache Spark

INT315 – Cluster Computing project by Sakshi Sood.

This project uses Apache Spark to process daily air-quality and weather data for 50 Indian cities. The pipeline validates and cleans the data, joins it, and builds analytical tables on cities, regions, seasons, weather associations and pollution episodes. A Streamlit dashboard presents the results.

> **Execution mode.** Spark runs in **local mode (`local[*]`) on one laptop**: the driver and executor threads share one JVM. This is parallel execution on a single machine, not a multi-machine cluster.

## Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the project](#running-the-project)
- [Project structure](#project-structure)
- [Pipeline overview](#pipeline-overview)
- [Spark concepts used](#spark-concepts-used)
- [Dashboard](#dashboard)
- [Tests](#tests)
- [Key results](#key-results)
- [Limitations](#limitations)
- [Project status](#project-status)

## Dataset

**India Daily Weather and Air Quality.** The project uses three tables from this dataset.

| File | Rows | Columns | Grain | Date range |
|---|---:|---:|---|---|
| `data/air_quality.csv` | 77,050 | 19 | city × day | 2022-08-05 to 2026-10-23 (1,541 days) |
| `data/weather.csv` | 124,400 | 14 | city × day | 2020-01-01 to 2026-10-23 (2,488 days) |
| `data/cities.csv` | 50 | 6 | city | — |

Every city has a complete daily series with no gaps. The tables link on `city`, and the daily tables also link on `date`.

**Sources:**
- Kaggle: <https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality/data>
- Hugging Face: <https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality>

**Pinned revision.** The source dataset is updated daily. To keep results reproducible, `download_data.py` downloads Hugging Face revision `339a4f37b76b9d2bae2f4b34cb189244019b4ed1`. The validation report records the SHA-256 checksum of each analysed file:

| File | SHA-256 |
|---|---|
| `air_quality.csv` | `5216e76081c4938fec03c93c331ee132d5230bd4f67661945895a30ae5046d6e` |
| `weather.csv` | `cca8500e7d67b59cc1156f42f89a12b36fcda85ec27d7030e5a527fbba421b99` |
| `cities.csv` | `a8e790f6c580c153a8319bcc674e07321513edec9cbfd87959a83884f141995c` |

A new download at the same revision can still differ byte for byte if the `datasets` library writes CSV files differently.

**The data is modelled, not measured.** Air-quality values come from the CAMS global air-quality model, and weather values come from ERA5/IFS reanalysis, both obtained through Open-Meteo. They are not readings from ground monitoring stations. Each value represents a grid cell of roughly 9–45 km, so the data can miss very local pollution peaks.

The CSVs are excluded from Git (see `.gitignore`). Download them with `download_data.py`.

## Requirements

- Windows, with an existing Spark installation. This project was developed with:
  - Apache Spark 3.5.9 (Scala 2.13) at `C:\Spark`, set in `SPARK_HOME`
  - Java 8 (OpenJDK 1.8.0_482), set in `JAVA_HOME`
  - `HADOOP_HOME` pointing to a folder whose `bin` contains `winutils.exe` (Spark needs this on Windows to write files)
- Python 3.11

Python packages are pinned in `requirements.txt`. Spark 3.5 does not support pandas 3 or numpy 2, so they are pinned below those versions:

| Package | Version | Used for |
|---|---|---|
| pyspark | 3.5.9 | Pipeline (must match the installed Spark) |
| pandas | 2.3.3 | Dashboard data handling |
| numpy | 1.26.4 | Dashboard charts |
| pyarrow | 25.0.1 | Reading Parquet in the dashboard |
| streamlit | 1.65.0 | Dashboard |
| plotly | 7.1.0 | Dashboard charts |
| datasets | 5.1.0 | `download_data.py` only |
| pytest | 8.4.2 | Tests |

Spark and Java are not installed from `requirements.txt`. PySpark uses the installation in `SPARK_HOME`.

## Setup

From the project folder, in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe download_data.py
```

`download_data.py` skips files that already exist. To re-download and overwrite them, use `--force`.

## Running the project

### 1. Run the Spark pipeline

```powershell
.\venv\Scripts\python.exe main.py --stage all
```

| Stage | What it does | Writes |
|---|---|---|
| `validate` | Checks the raw CSVs | `outputs/reports/validation_report.json` |
| `process` | Validates, then cleans and joins the data | `outputs/processed/`, `processing_report.json`, the saved execution plan |
| `analytics` | Builds the analytical tables from the processed data | `outputs/analytics/`, `analytics_report.json` |
| `all` | Runs validate, process and analytics in order | All of the above |

`process` and `all` always validate first. If validation finds a critical problem, the pipeline stops. Spark event logs go to `outputs/spark-events/`, so a finished run can be replayed in the Spark History Server.

### 2. Start the dashboard

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard reads only the files that the pipeline wrote. It never starts Spark, so run the pipeline first.

### 3. Run the tests

```powershell
.\venv\Scripts\python.exe -m pytest
```

## Project structure

```
spark-air-quality/
├── main.py                  # Pipeline entry point (--stage validate|process|analytics|all)
├── download_data.py         # Downloads the three CSVs at the pinned revision
├── app.py                   # Streamlit entry point and page navigation
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml   # Dashboard theme and server settings
├── src/
│   ├── config.py            # Paths, dataset facts, units, thresholds, Spark settings
│   ├── spark_session.py     # Local-mode SparkSession and environment report
│   ├── schemas.py           # Explicit schemas and the raw CSV reader
│   ├── data_validation.py   # 46 data-quality checks -> validation report
│   ├── preprocessing.py     # Cleaning, calendar/season columns, flags, weather bins
│   ├── joins.py             # Key-checked joins (air quality + weather + cities)
│   ├── analytics.py         # City/region/state, temporal, episode and RDD tables
│   ├── weather_analysis.py  # Correlations and weather-bin comparisons
│   └── utils.py             # Logging, hashing, JSON, Parquet writing, plan capture
├── dashboard/
│   ├── data_loader.py       # Cached Parquet/JSON readers (no Spark)
│   ├── filters.py           # Shared sidebar filters
│   ├── logic.py             # Pure helper logic (tested)
│   ├── charts.py            # Plotly chart builders
│   ├── labels.py            # Display names, units and caveat texts
│   ├── ui.py                # Layout helpers and page styling
│   └── views/               # overview, city_region, weather, temporal, performance
├── tests/                   # pytest suite (validation, preprocessing, joins, analytics, dashboard)
├── data/                    # Raw CSVs (not committed)
└── outputs/                 # Generated by main.py (not committed)
    ├── processed/           # Cleaned and joined Parquet tables
    ├── analytics/           # 13 analytical Parquet tables
    ├── reports/             # JSON reports and plans/daily_joined_plan.txt
    └── spark-events/        # Spark event logs
```

## Pipeline overview

1. **Read.** `schemas.py` reads each CSV with an explicit schema instead of `inferSchema`. The reader uses `PERMISSIVE` mode with a corrupt-record column and `enforceSchema=False`, so a header mismatch raises an error and malformed rows are counted.
2. **Validate.** `data_validation.py` runs 46 checks:
   - header, malformed rows, nulls, exact and conflicting duplicates
   - value ranges and AQI category consistency
   - date coverage and city matching across files

   Each check has a severity: critical, warning or info. The current data passes all 46.
3. **Preprocess.** `preprocessing.py`:
   - trims strings and removes duplicate `(city, date)` keys
   - adds year, month and season (India Meteorological Department seasons)
   - flags the last 10 days as provisional
   - adds NAAQS and WHO 2021 exceedance flags (air quality)
   - adds wind, rain, temperature and humidity bins (weather)
4. **Write and re-read Parquet.** Each cleaned table is written as one Snappy-compressed Parquet file, sorted by `(city, date)`. The next step reads from Parquet instead of parsing the CSVs again.
5. **Join.** `joins.py` checks that each key is unique and then joins:
   - air quality and weather, inner join on `(city, date)`
   - city metadata, broadcast join
   - season completeness, broadcast join

   Row counts are checked after every join. The result is `daily_joined`: 77,050 rows and 49 columns. All 77,050 air-quality rows have weather data. The 47,350 weather rows without air-quality data are the days before 5 August 2022.
6. **Analyse.** `analytics.py` and `weather_analysis.py` build 13 tables:

   | Group | Tables |
   |---|---|
   | Summary | `dataset_summary` |
   | City, region and state | `city_pollutant_stats`, `region_pollutant_stats`, `state_pollutant_stats` |
   | AQI | `aqi_category_distribution` |
   | Time | `monthly_city`, `annual_city`, `seasonal_city` |
   | Weather | `weather_correlations`, `weather_bin_stats` |
   | Episodes and extremes | `pollution_episodes`, `top_pollution_days` |
   | RDD check | `rdd_city_means` |

## Spark concepts used

| Concept | Where |
|---|---|
| Lazy evaluation; transformations vs actions | Functions build plans; `count`, `collect` and `write` trigger jobs |
| Narrow vs wide transformations, shuffles | `withColumn`/`filter` vs `dropDuplicates`, `groupBy`, joins, `repartition` |
| Broadcast joins | `F.broadcast(cities)` and the season-coverage join. The saved plan shows Spark also chose a `BroadcastHashJoin` for air quality ⋈ weather, because the Parquet inputs are small |
| Caching | Raw inputs during validation, the joined table during processing, three reused inputs during analytics |
| Partitioning | `spark.sql.shuffle.partitions = 8` (default 200). Output uses `repartition(1)` + `sortWithinPartitions` |
| Spark SQL | Region and state statistics are written in SQL over a temp view |
| Window functions | `rank`, `row_number`, `lag`, average ranks for Spearman correlation, "gaps and islands" episode detection |
| Other DataFrame operations | `unpivot`, `explode`, `percentile_approx`, `corr` |
| RDD API | `map` + `reduceByKey` city means, checked against the DataFrame result. All 50 cities match (max absolute difference 0.0) |
| Execution plans | `explain("formatted")` saved to `outputs/reports/plans/daily_joined_plan.txt` |
| Adaptive Query Execution | Enabled (Spark 3.5 default) and recorded in each report |

## Dashboard

The dashboard has five pages that share sidebar filters for indicator, region and city:

- **Overview:** headline figures, regional means, key findings
- **Cities & regions:** city ranking with distributions, map, region/state tables, AQI category shares
- **Weather & pollution:** correlation matrix, all-year vs within-season comparison, scatter plot, weather-bin means
- **Trends over time:** daily, monthly, annual and seasonal views, pollution episodes, highest-PM2.5 days
- **Spark performance:** recorded environment, data sizes, measured pipeline timings, status of the planned experiments, the saved physical plan

## Tests

The `tests/` folder contains 45 test functions. Some are parametrised, so pytest collects 60 test cases. They cover:

- header and malformed-row handling
- duplicate classification
- preprocessing flags and bins
- join grain checks
- analytics logic, such as episodes, correlations with nulls, and RDD vs DataFrame agreement
- dashboard logic and page loading

Tests use a separate small `local[2]` SparkSession and write only to temporary folders.

## Key results

All results describe **modelled** data. They are computed from `outputs/analytics/` for the full period from 5 Aug 2022 to 23 Oct 2026.

- **City extremes.** Delhi has the highest mean PM2.5, at 74.2 µg/m³. Its modelled daily mean is above the national 24-hour limit (60 µg/m³) on 56.6 % of days. Leh has the lowest mean, at 4.9 µg/m³.
- **Regions.** Mean PM2.5 is highest in the East (49.1 µg/m³, 4 cities) and the North (46.5 µg/m³, 16 cities). It is lowest in the South (22.9 µg/m³, 13 cities). Region sizes differ a lot, and 24 of the 30 states have fewer than 3 cities.
- **Seasons.** Only complete seasons are counted. Average PM2.5 is highest in Winter (53.5 µg/m³), followed by Post-monsoon (46.4), Pre-monsoon (31.7) and Monsoon (22.6).
- **Weather.** Across all cities and days, PM2.5 is lower on windier days (Spearman ρ = −0.31) and on wetter days (ρ = −0.45). Mean PM2.5 falls from 47.6 µg/m³ on days with wind below 10 km/h to 21.3 µg/m³ on days with wind of 20 km/h or more. There are exceptions:
  - The wind association is positive in 7 of 50 cities.
  - Within Winter, the humidity association turns positive (ρ = +0.35).
  - Within the Monsoon, the temperature association turns positive (ρ = +0.59).
- **Episodes.** There are 1,018 runs of at least 3 consecutive days with modelled PM2.5 above 60 µg/m³, across 39 cities. The longest is 111 days, in Kolkata (Nov 2025 – Feb 2026). Twelve episodes reach the last date in the data, so they may still be ongoing.
- **RDD check.** The RDD and DataFrame city means agree for all 50 cities.

## Limitations

- **Modelled data.** Concentrations come from a global model, not from monitoring stations, and can miss local peaks.
- **Correlation is not causation.** Season is a strong confounder, so every correlation is also computed within each complete season.
- **Indian AQI is affected by ozone.** The all-pollutant `india_aqi` is dominated by ozone on about 90 % of days, because CAMS overestimates surface ozone over India. The project therefore uses PM2.5 and the PM-only `india_aqi_pm` as primary indicators.
- **Exceedance flags are not compliance results.** They compare modelled daily means with the NAAQS and WHO 24-hour values. Official compliance uses station monitoring.
- **Only three complete years.** The data contains only three complete calendar years (2023–2025). Year-to-year comparisons are descriptive, not evidence of a long-term trend.
- **Provisional and partial periods.** The last 10 days are provisional. Monsoon 2022 and Post-monsoon 2026 are incomplete and are left out of the seasonal results.
- **Single machine.** Spark runs in local mode on one laptop (4 logical cores, 7.7 GB RAM). The dataset is about 18 MB. The results show how Spark is used, not how it scales on a cluster.

## Project status

| Phase | Status |
|---|---|
| Setup, data inspection, pinned dataset | Done |
| 3. Validation, preprocessing, joins | Done |
| 4. Analytical tables | Done |
| 5. Streamlit dashboard | Done |
| 6. Performance experiments | **Pending** |

The planned experiments for Phase 6:

| ID | Experiment |
|---|---|
| E1 | Shuffle partitions, with AQE off |
| E2 | Input partitioning |
| E3 | Caching vs no caching |
| E4 | CSV vs Parquet, and inferred vs explicit schema |
| E5 | Broadcast vs sort-merge join |
| E6 | `local[1]`, `local[2]` and `local[4]` |
| E7 | Input size |

None of these experiments has been run yet. The only timings recorded so far are single pipeline runs, not controlled measurements.
