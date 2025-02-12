## Generation of Input Data that is Subset

We do a lot of validation of the input data including ensuring all the burst data provided is coregistered and the dates are consistent, etc.
However, there is a nice way to mock data in that we can subset the burst data and georeference to its approrpriate subset.
That is we do not check that the input burst data itself is correct relative to some fixed area (this is one assumption we make).
So we can provide inputs that are a small fraction of what is nominally provided.

This generation is shown here: https://github.com/OPERA-Cal-Val/dist-s1-research/blob/dev/marshak/S_create_test_data/Generate_Test_Data.ipynb 

Then take the data created and put into `test_data/cropped`. Relative paths are important so careful!

To generate the intermediate data found in `test_data/10SGD_dst`, change the `ERASE_WORKFLOW_OUTPUTS` variable to `False` in the `test_workflows.py` file and run the tests and copy over the necessary diretories from `tmp` to `test_data/10SGD_dst`.