# Reproducing ADMET-AI Models

This document provides step-by-step instructions for reproducing the ADMET-AI models in the current release. To reproduce the data, models, and results from [our paper](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030), please check out a commit from ADMET-AI v1 (e.g., v1.4.0 [here](https://github.com/swansonk14/admet_ai/releases/tag/v_1.4.0)) and follow those instructions.

- [Download TDC ADMET data](#download-tdc-admet-data)
- [Create multitask datasets for regression and classification](#create-multitask-datasets-for-regression-and-classification)
- [Create a single dataset with all TDC ADMET data](#create-a-single-dataset-with-all-tdc-admet-data)
- [Train Chemprop ADMET predictors](#train-Chemprop-admet-predictors)
- [Evaluate TDC ADMET Benchmark Group models](#evaluate-tdc-admet-benchmark-group-models)
- [Get approved drugs from DrugBank](#get-approved-drugs-from-drugbank)
- [Make predictions on DrugBank approved drugs](#make-predictions-on-drugbank-approved-drugs)

## Download TDC ADMET data

For the following download commands only, use a Python environment that contains `PyTDC==1.1.15` and `typed-argument-parser`. (Note that PyTDC is incompatible with the other requirements in ADMET-AI.)

Download the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) data (v1.1.15) for evaluating models using scaffold splits in order to compare to the TDC leaderboard.

```bash
python scripts/prepare_tdc_admet_group.py \
    --raw_data_dir data/tdc_admet_group_raw \
    --save_dir data/tdc_admet_group
```

Download all TDC [ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [Tox](https://tdcommons.ai/single_pred_tasks/tox/) datasets for training models. Skip datasets that are redundant or not needed.

```bash
python scripts/prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast
```

## Filter for valid molecules

Filter all the CSV files to ensure SMILES result in valid RDKit molecules. (Make sure to use the general ADMET-AI Python environment with an up-to-date RDKit.)

```bash
python scripts/filter_valid_molecules.py \
    --data_dir data/tdc_admet_group \
    --smiles_column Drug
```

```bash
python scripts/filter_valid_molecules.py \
    --data_dir data/tdc_admet_all \
    --smiles_column smiles
```

## Create multitask datasets for regression and classification

Create multitask datasets for regression and classification for all the TDC ADMET datasets.

```bash
python scripts/merge_tdc_admet_multitask.py \
    --data_dir data/tdc_admet_all \
    --save_dir data/tdc_admet_all_multitask
```

## Create a single dataset with all TDC ADMET data

Create a single dataset with all TDC ADMET data, primarily for the purpose of searching across the TDC data.

```bash
python scripts/merge_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_path data/tdc_admet_all.csv
```

## Train Chemprop ADMET predictors

Train Chemprop predictors on the ADMET data. Note: A GPU is used by default if available.

Train Chemprop ADMET predictors on the TDC ADMET Benchmark Group data.

```bash
python scripts/train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop
```

Train Chemprop ADMET predictors on all TDC ADMET datasets.

```bash
python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all \
    --model_type chemprop
```

Train Chemprop ADMET predictors on the TDC ADMET multitask datasets.

```bash
python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop
```

## Evaluate TDC ADMET Benchmark Group models

Evaluate Chemprop ADMET predictors trained on the TDC ADMET Benchmark Group data.

```bash
python scripts/evaluate_tdc_admet_group.py \
    --data_dir data/tdc_admet_group_raw \
    --preds_dir models/tdc_admet_group/chemprop
```

## Get approved drugs from DrugBank

Get approved drugs from [DrugBank](https://go.drugbank.com/) (v5.1.14) to create a comparison set for Chemprop ADMET predictors. This assumes that the file `drugbank.xml` has been downloaded from DrugBank (license required).

```bash
python scripts/get_drugbank_approved.py \
    --data_path data/drugbank/drugbank.xml \
    --save_path data/drugbank/drugbank_approved.csv
```

## Make predictions on DrugBank approved drugs

Make ADMET predictions on DrugBank approved drugs using Chemprop multitask predictor (and compute physicochemical properties).

```bash
admet_predict \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_path data/drugbank/drugbank_approved_physchem_admet.csv \
    --models_dir models/tdc_admet_all_multitask/chemprop \
    --smiles_column smiles
```
