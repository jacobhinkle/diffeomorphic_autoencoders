Data cleaning, analysis, and visualization code for the manuscript at
https://openreview.net/forum?id=Hkg0j9sA1V

# Instructions

In order to reproduce the figures from the submitted manuscript, the following
steps are required:

- Create a python environment containing PyTorch, lagomorph, pandas, and jupyter
- Download the OASIS data and convert to HDF5
- Run the script `run_all.py` in the current directory
- Run the notebook `OASISPlots.ipynb`

Below are details for each of these steps.

NOTE: At this time, lagomorph only supports CUDA devices, so you must have an
nvidia CUDA gpu available when running the analysis script.

## Creating a python environment

Using either conda or python's venv module, create a new virtualenv, activate
it, and run the following:

```
pip install pytorch lagomorph pandas jupyter
```


## Downloading the data

First, sign up for data access for the OASIS-3 project and agree to the terms and conditions at
https://oasis-brains.org/

Once you have a username and password for central.xnat.org, run the following
commands in the `data` directory:

```
XNATUSER=<fill in your xnat username>
./download_oasis_skullstripped.sh ./all_freesurfers.csv skullstripped $XNATUSER
./create_oasis3_h5.py
```

## Run analysis

With the virtual environment active, run

```
python3 run_all.py
```

This will create many results files in the current directory with extensions
`.pth` and `.h5`. If you re-run an interrupted run, the analysis should find
these and load them instead of recomputing the intermediate files. Clear them
all out if you'd like to play with parameters and re-run the analysis from
scratch.

## Visualize results

Once the `pth` and `h5` files are computed by the previous step, the
visualizations can be generated. Simply load the `OASISPlots.ipynb` notebook in
jupyter and rerun it to recreate the visualizations in the manuscript.
