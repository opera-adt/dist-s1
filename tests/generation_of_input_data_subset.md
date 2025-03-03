# Generation of Input Data Workflow/Processing Tests

The DIST-S1 runconfig does a lot of validation of the input data including ensuring all the burst data provided as input is coregistered, that their dates are consistent, etc.
However, there is a nice way to "trick" (or "flaw") in our workflows that can be exploited into processing a smaller set of data - we can replace the input RTC-S1 with a cropped subset georeferenced to the cropped area.
Despite all the validation we do, we do not check that the input burst data itself spans some fixed area (so if we replace the expected input file with a cropped subset, the workflow will not catch this and still run).
So we can provide inputs that are a small fraction of what is nominally provided and ensure the workflow runs correctly.

We do a considerable amount of testing and since our workflow is IO heavy, we have to be careful in regards to providing paths to the workflow.

This data generation of how we can crop this data is shown here: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/S_create_test_data/Generate_Test_Data.ipynb 
The data is then transferred to `test_data/cropped`.

If you want to update this dataset, it requires some care.

To generate the intermediate data found in `test_data/10SGD_dst` for step workflow tests, change the `ERASE_WORKFLOW_OUTPUTS` variable to `False` in the `test_workflows.py` file and run the tests and copy over the necessary diretories from `tmp` to `test_data/10SGD_dst`.

Finally, we have a runconfig test (in `tests/test_runconfig_model.py`) and using the notebook above, there is input metadata that is needed to ensure what we expect from the run config is what we actually get.

Lastly, it is important to utilize


## Related Tests

- `test_runconfig_model.py` - ensures the runcofig data is instantiated correctly from the metadata - needs correct file paths
- `test_workflows.py` - ensures the workflow runs and the outputs are correct - needs correct file paths, a golden dataset (see `conftest.py` for the path of the location provided). Also workflow steps are tested and so intermediate tests are generated.

# Water Mask Tests

The water mask tests are in `tests/test_water_mask.py`. We test the function `water_mask_control_flow` which is in the file `src/dist_s1/water_mask.py`.
The tests are hopefully self-explanatory.
Generating the sample water masks are found here: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/S_create_test_data/Water_Mask_Generation.ipynb
