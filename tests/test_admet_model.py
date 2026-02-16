"""Tests for ADMET-AI."""

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from admet_ai import admet_predict, ADMETModel
from admet_ai.constants import DEFAULT_ADMET_PATH, DEFAULT_DRUGBANK_PATH


ADMET_DATA = pd.read_csv(DEFAULT_ADMET_PATH)
DRUGBANK_DATA = pd.read_csv(DEFAULT_DRUGBANK_PATH)


def test_admet_predict_drugbank() -> None:
    """Test that predictions for a specific SMILES string remain consistent."""
    with TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "data.csv"
        preds_path = Path(temp_dir) / "preds.csv"

        drugbank_smiles = DRUGBANK_DATA[["smiles"]]
        drugbank_smiles.to_csv(data_path, index=False)

        admet_predict(
            data_path=data_path,
            save_path=preds_path,
            include_physchem=True,
            drugbank_path=None,
            atc_code=None,
            num_workers=0,
        )

        preds = pd.read_csv(preds_path)

        assert len(preds.columns[1:]) == len(ADMET_DATA)

        for column in preds.columns[1:]:
            assert np.allclose(preds[column].values, DRUGBANK_DATA[column].values), f"Column {column} does not match"
