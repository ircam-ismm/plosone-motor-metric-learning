# Movement Metric Learning - PLOS One

This repository contains the work published as a journal in [PlosOne](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272509).

The purpose of this repository is to support reproducible research. It contains:
- the datasets
- the optimisation code
- the statistical analysis code
- the visualisation code

that were used to produce the graphs and the stastical analysis present in the manuscript.

You will need a python environement to run the code. For those using `conda`, it can be created with:

```
conda env create --file requirements.yaml
```

## dataset
The raw files are provided as zip files under the release v1.0.0. Download and extract these under data.

## optimisation code
The source code for all the computations presented in the paper is contained under `app`.
In that directory, you can execute:

```
kedro run --env [exp] --pipeline [exp]
```

where `[exp]` is one of: exp1, spatial, temporal. This can be used to reproduce the baseline correlation grid search, the spatial optimisation and the temporal optimisation, respectively.

## figure and analysis
Every figures and statistical analysis can be re-generated through a set of jupyter notebooks.
These are located under figures.

## Acknowledgements

Supported by the ELEMENT project (ANR-18-CE33-0002) and the ARCOL project (ANR-19-CE33-0001) from the French National Research Agency.
