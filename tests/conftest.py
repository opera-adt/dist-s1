import os
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def change_local_dir() -> Generator[Callable[[Path], Path], None, None]:
    """Change the working directory."""
    original_dir = Path.cwd()

    def _change_dir(target_dir: Path) -> Path:
        target_dir = Path(target_dir).resolve()
        os.chdir(target_dir)
        return target_dir

    yield _change_dir

    # Restore the original directory
    os.chdir(original_dir)
    assert Path.cwd() == original_dir


@pytest.fixture
def test_dir() -> Path:
    """Fixture to provide the path to the test directory."""
    test_dir = Path(__file__).parent
    test_dir = test_dir.resolve()
    return test_dir


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture to provide the path to the test_data directory."""
    test_dir = Path(__file__)
    test_data_dir = test_dir.parent / 'test_data'
    test_data_dir = test_data_dir.resolve()
    return test_data_dir


@pytest.fixture
def test_opera_golden_dummy_dataset() -> Path:
    """Fixture to provide the path to the test_out directory."""
    test_dir = Path(__file__)
    golden_datasets_dir = test_dir.parent / 'test_data' / 'golden_datasets' / '10SGD'
    golden_dummy_dataset = (
        golden_datasets_dir / 'OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250211T141605Z_S1_30_v0.1'
    )
    return golden_dummy_dataset


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture to provide a Click test runner."""
    return CliRunner()
