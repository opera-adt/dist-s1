import os
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def change_local_dir():
    """Fixture to temporarily change the working directory"""
    original_dir = Path.cwd()

    def _change_dir(target_dir: Path):
        target_dir = Path(target_dir).resolve()
        os.chdir(target_dir)
        return target_dir

    yield _change_dir

    # Restore the original directory
    os.chdir(original_dir)
    assert Path.cwd() == original_dir


@pytest.fixture
def test_dir():
    """Fixture to provide the path to the test directory."""
    test_dir = Path(__file__).parent
    test_dir = test_dir.resolve()
    return test_dir


@pytest.fixture
def test_data_dir():
    """Fixture to provide the path to the test_data directory."""
    test_dir = Path(__file__)
    test_data_dir = test_dir.parent / 'test_data'
    test_data_dir = test_data_dir.resolve()
    return test_data_dir


@pytest.fixture
def cli_runner():
    """Fixture to provide a Click test runner."""
    return CliRunner()
