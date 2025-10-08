# Generation of Input Data Workflow/Processing Tests

The DIST-S1 runconfig does a lot of validation of the input data including ensuring all the burst data provided as input is coregistered, that their dates are consistent, etc.
However, there is a nice way to trick (or maybe this is a flaw in our design) in our workflows that can be exploited into processing a smaller set of data - we can replace the input RTC-S1 with a cropped subset.
Despite all the validation we do, we do not check that the input burst data itself spans some fixed area.
Thus, we can replace the expected input file with a cropped subset, the workflow will not catch this and still run.
The workflow will run much, much faster.

We do a considerable amount of testing and since our workflow is IO heavy, we have to be careful in regards to providing paths to the workflow.

This data generation of how we can crop this data is shown here: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/S_create_test_data/Generate_Test_Data.ipynb 
The data is then transferred to `test_data/cropped`.


# Regenerating the test dataset

Suppose you need to modify the golden datasets (yes there are a lot). What is the checklist of datasets that need to be updated?

- [ ] `tests/test_data/golden_datasets/10SGD/` - 2 products
- [ ] `tests/test_data/golden_dataset/10SGD_confirmed/` - 1 products
- [ ] `tests/test_data/golden_datasets/products_with_confirmation_cropped__chile-fire_2024/` - 9 products
- [ ] `tests/test_data/products_without_confirmation_cropped__chile-fire_2024/` - 9 products

The first two golden dataset products can be generated using `test_workflows` and copying them over from the `tmp` directory. Change the `ERASE_WORKFLOW_OUTPUTS` variable to `False` in that file (`test_workflows.py`) and copy them over. The paths need to be updated in `conftest.py`.

For the last two datasets, here are the steps:

1. Run `notebooks/C__Time_series_and_confirmation.ipynb`
2. Copy over the unconfirmed datasets and use this: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/W_crop_dist_s1_prod/cropping_dist_s1_prod.ipynb
3. Copy over the unconfirmed datasets to the unconfirmed directory
4. Re-run the sequential confirmation test in `test_workflows.py`.

## Runconfig Tests

We have a runconfig test (in `tests/test_runconfig_model.py`). There are some tests that use yml files to initialize RunConfigData models and AlgoConfigData models. Modifying/adding a field is usually what's needed if these datasets change.


# Water Mask Tests

The water mask tests are in `tests/test_water_mask.py`. We test the function `water_mask_control_flow` which is in the file `src/dist_s1/water_mask.py`.
The tests are hopefully self-explanatory.
Generating the sample water masks are found here: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/S_create_test_data/Water_Mask_Generation.ipynb

We also have some tests in `test_workflows.py` in which we provide different water masks to ensure a large geotiff or VRT covering a MGRS tile can be correctly utilized.