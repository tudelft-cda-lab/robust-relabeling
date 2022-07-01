# Adversarially Robust Decision Tree Relabeling

This repository contains the source code for reproducing 'Adversarially Robust Decision Tree Relabeling'. The main contribution is robust relabeling which is implemented under `robust_relabeling.py`. Training trees with the robust relabeling criterion can be done with our implementation extending the [GROOT framework](https://github.com/tudelft-cda-lab/GROOT) under `model.py`. The remaining scripts are used for generating tables and figures of the paper as explained below.

## Installation

To run our code a relatively new version of python is required (>= 3.6). Then, all dependencies can be installed as follows:
```
pip install -r requirements.txt
```

**We recommend doing this within a virtual environment.**

## Reproducing results

The most important results are the ones comparing the performance of different algorithms for training trees / ensembles. Running `compare_robustness.py` loads the datasets, splits them, executes all training methods, evaluates the models and saves models + results under `out/`. This can take approximately a day to run. Especially training trees with robust relabeling as criterion is slow which can be turned off for example by runnning:
```
python compare_robustness.py --ignore_relabeling_criterion
```

The latex tables can be generated using `plot_results.py`. Some information about the datasets can be printed by running `summarize_datasets.py`.

### Visualizing toy datasets

To create plots of decision trees / random forests / gradient boosting on toy data run:

```
python plot_robustification.py
```

### Timing relabeling and relabeling criterion trees

To measure the time taken for robustly relabeling different models or for training trees with the relabling criterion run:

```
python time_relabeling.py
```

### Comparing with Cost Complexity Pruning

Comparing accuracy and adversarial accuracy scores for all dataset can be done by running:

```
python compare_pruning.py
```