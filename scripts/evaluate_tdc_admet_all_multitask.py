"""Evaluate predictions from multitask Chemprop models on all TDC ADMET datasets."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score

from tdc_constants import ADMET_ALL_SMILES_COLUMN


def evaluate_tdc_admet_all_multitask(data_dir: Path, preds_dir: Path, save_dir: Path) -> None:
    """Evaluate predictions from multitask Chemprop models on all TDC ADMET datasets.

    :param data_dir: A directory containing the multitask TDC ADMET datasets.
    :param preds_dir: A directory containing the predictions from Chemprop models trained on the multitask TDC ADMET datasets.
    :param save_dir: A directory where the evaluation results will be saved.
    """
    # Get dataset names
    names = ["admet_regression", "admet_classification"]

    # Load data
    regression_data = pd.read_csv(data_dir / "admet_regression.csv")
    classification_data = pd.read_csv(data_dir / "admet_classification.csv")

    # Load predictions for each dataset
    regression_preds = pd.concat(
        [pd.read_csv(path) for path in preds_dir.glob("admet_regression/**/test_predictions.csv")]
    )
    classification_preds = pd.concat(
        [pd.read_csv(path) for path in preds_dir.glob("admet_classification/**/test_predictions.csv")]
    )

    # Average predictions across replicates
    regression_preds = regression_preds.groupby("smiles").mean()
    classification_preds = classification_preds.groupby("smiles").mean()

    # Set index to SMILES
    regression_data.set_index(ADMET_ALL_SMILES_COLUMN, inplace=True)
    classification_data.set_index(ADMET_ALL_SMILES_COLUMN, inplace=True)

    # Subset data to only include molecules with predictions
    regression_data = regression_data.loc[regression_preds.index]
    classification_data = classification_data.loc[classification_preds.index]

    # Evaluate predictions and save results
    save_dir.mkdir(parents=True, exist_ok=True)

    classification_results = []
    for column in classification_data.columns:
        valid_mask = classification_data[column].notna()
        valid_data = classification_data[valid_mask]
        valid_preds = classification_preds[valid_mask]

        classification_results.append(
            {
                "task": column,
                "roc_auc": roc_auc_score(valid_data[column], valid_preds[column]),
                "prc_auc": average_precision_score(valid_data[column], valid_preds[column]),
            }
        )

    pd.DataFrame(classification_results).to_csv(save_dir / "admet_classification_results.csv", index=False)

    regression_results = []
    for column in regression_data.columns:
        valid_mask = regression_data[column].notna()
        valid_data = regression_data[valid_mask]
        valid_preds = regression_preds[valid_mask]

        regression_results.append(
            {
                "task": column,
                "mae": mean_absolute_error(valid_data[column], valid_preds[column]),
                "r2": r2_score(valid_data[column], valid_preds[column]),
            }
        )

    pd.DataFrame(regression_results).to_csv(save_dir / "admet_regression_results.csv", index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(evaluate_tdc_admet_all_multitask)
