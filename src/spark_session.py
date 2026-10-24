"""Create the project's SparkSession.

Runs in Spark *local mode*: the driver and executor threads share one JVM on
this laptop. This is local parallel execution, not a multi-machine cluster.

The session uses the existing Spark installation found through SPARK_HOME
(C:\\Spark, Spark 3.5.9) and the system Java 8. Nothing global is changed;
only PYSPARK_PYTHON is set for this process so that Spark's Python workers use
the same interpreter (the project venv) as the driver.
"""
import os
import sys

from pyspark.sql import SparkSession

from src.config import DRIVER_MEMORY, SHUFFLE_PARTITIONS, SPARK_EVENTS_DIR


def create_spark_session(app_name: str = "AirQualityAnalytics",
                         master: str = "local[*]",
                         shuffle_partitions: int = SHUFFLE_PARTITIONS,
                         event_log: bool = True,
                         extra_conf: dict | None = None) -> SparkSession:
    """Build (or reuse) a SparkSession with project-level settings only.

    shuffle_partitions: number of partitions produced by wide operations
        (groupBy, join). Spark's default of 200 creates mostly empty tasks on
        this small dataset.
    event_log: write job/stage/task events to outputs/spark-events so the run
        can be replayed in the Spark History Server after the process exits.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable

    builder = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Takes effect because the JVM is launched by this Python process.
        .config("spark.driver.memory", DRIVER_MEMORY)
        # Dates in the data are IST calendar days; keep Spark's clock in IST.
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        .config("spark.ui.showConsoleProgress", "false")
        # Our tables have ~50 columns; keep saved explain() plans untruncated.
        .config("spark.sql.debug.maxToStringFields", "100")
    )
    if event_log:
        SPARK_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        builder = (builder
                   .config("spark.eventLog.enabled", "true")
                   .config("spark.eventLog.dir", SPARK_EVENTS_DIR.as_uri()))
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def describe_environment(spark: SparkSession) -> dict:
    """Versions and settings worth recording next to every report."""
    sc = spark.sparkContext
    conf = spark.conf
    return {
        "spark_version": spark.version,
        "java_version": sc._jvm.System.getProperty("java.version"),
        "scala_version": sc._jvm.scala.util.Properties.versionNumberString(),
        "python_version": sys.version.split()[0],
        "spark_home": os.environ.get("SPARK_HOME"),
        "master": sc.master,
        "default_parallelism": sc.defaultParallelism,
        "shuffle_partitions": conf.get("spark.sql.shuffle.partitions"),
        "adaptive_query_execution": conf.get("spark.sql.adaptive.enabled"),
        "driver_memory": sc.getConf().get("spark.driver.memory", "default"),
        "spark_ui": sc.uiWebUrl,
    }
