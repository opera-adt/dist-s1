# Regression Testings the SAS

## Getting started

There are two parts:

1. Publishing the golden dataset
2. Running a (SAS) regression test on the version

For 1., you will need write access to the appropriate s3 bucket.
For 1. and 2., you will need to install the `awscli` and `boto3` via `mamba`.


## Publishing a Golden Dataset.

Ensure you can write to the relevant s3 bucket and Run

```bash
python 0_generate_golden_dataset.py
python 1_update_config_and_check.py
python 2_delivery.py
```

You can replace the last run by running `aws s3 sync . s3://dist-s1-golden-datasets/<DIST_S1_VERSION>`.

## Regression testing

1. Determine available versions via:
    ```bash
    aws s3 ls dist-s1-golden-datasets
    ```
    and select the relevant release
    ```bash
    aws s3 sync s3://dist-s1-golden-datasets/<DIST_S1_VERSION> .
    ```
    You should have all relevant files to run the code and regression test.

2. Generate a DIST-S1 product: 
    ```bash
    dist-s1 run_sas --run_config_path runconfig.yml
    ```
    The above should generate a product in `test_product`.
3. Check equality: 
    ```bash
    dist_s1 check_equality golden_dataset/<OPERA_ID> test_dataset/<OPERA_ID>
    ```
    Best to use the latest generated products in either directory.
