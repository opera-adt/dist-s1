from collections.abc import Callable
from pathlib import Path

from click.testing import CliRunner

from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_sas_main(
    cli_runner: CliRunner,
    test_dir: Path,
    change_local_dir: Callable[[Path], None],
    cropped_10SGD_dataset_runconfig: Path,
    test_opera_golden_dummy_dataset: Path,
) -> None:
    """Test the dist-s1 sas main function.

    This is identical to running from the test_directory:

    `dist-s1 run_sas --runconfig_yml_path test_data/cropped/sample_runconfig_10SGD_cropped.yml`

    And comparing the output product directory to the golden dummy dataset.

    Note: the hardest part is serializing the runconfig to yml and then correctly finding the generated product.
    This is because the product paths from the in-memory runconfig object are different from the ones created via yml.
    This is because the product paths have the *processing time* in them, and that is different depending on when the
    runconfig object is created.
    """
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    product_data_golden = ProductDirectoryData.from_product_path(test_opera_golden_dummy_dataset)

    # Load and modify runconfig - not the paths are relative to the test_dir
    runconfig_data = RunConfigData.from_yaml(cropped_10SGD_dataset_runconfig)
    # We have a different product_dst_dir than the dst_dir called `tmp2`
    product_dst_dir = test_dir / 'tmp2'
    assert runconfig_data.product_dst_dir.resolve() == product_dst_dir.resolve()

    tmp_runconfig_yml_path = tmp_dir / 'runconfig.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path)

    # Run the command
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--runconfig_yml_path', str(tmp_runconfig_yml_path)],
    )
    # The product_dst_dir is created - have to find it because it has a processing time
    # and will be different from the runconfig data object
    product_directories = list(product_dst_dir.glob('OPERA*'))
    # Should be one and only one product directory
    assert len(product_directories) == 1
    product_data_path = product_directories[0]
    out_product_data = ProductDirectoryData.from_product_path(product_data_path)

    # Check the product_dst_dir exists
    assert product_dst_dir.exists()
    assert result.exit_code == 0

    assert out_product_data == product_data_golden
