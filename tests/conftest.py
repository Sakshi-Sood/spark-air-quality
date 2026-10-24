"""Shared pytest fixtures: one small local SparkSession for the whole test run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spark_session import create_spark_session  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = create_spark_session(
        app_name="pipeline-tests", master="local[2]", shuffle_partitions=2,
        event_log=False, extra_conf={"spark.ui.enabled": "false"},
    )
    yield session
    session.stop()
