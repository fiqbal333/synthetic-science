import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthsci import config as C  # noqa: E402
from synthsci.landscape import build_landscape  # noqa: E402
from synthsci.seeding import SeedingTables  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return C.resolve("S0")


@pytest.fixture(scope="session")
def tables(cfg):
    return SeedingTables(cfg["seeding"]["table_set"])


@pytest.fixture(scope="session")
def landscape(cfg, tables):
    return build_landscape(cfg, C.seed_layers(cfg, 1), tables)


@pytest.fixture(scope="session")
def tmp_runs(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")
