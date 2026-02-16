"""Tests for ADMET-AI."""

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from admet_ai import admet_predict, ADMETModel
from admet_ai.constants import DEFAULT_ADMET_PATH, DEFAULT_DRUGBANK_PATH


ADMET_DATA = pd.read_csv(DEFAULT_ADMET_PATH)
DRUGBANK_DATA = pd.read_csv(DEFAULT_DRUGBANK_PATH)


# TODO: move workers
@pytest.mark.parametrize("num_workers", [0])
def test_admet_predict_drugbank(num_workers: int) -> None:
    """Test predictions on DrugBank data using admet_predict."""
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
            num_workers=num_workers,
        )

        preds = pd.read_csv(preds_path).set_index("smiles")

        assert len(preds.columns) == len(ADMET_DATA)

        for column in preds.columns:
            assert np.allclose(preds[column].values, DRUGBANK_DATA[column].values), f"Column {column} does not match"


# TODO: more workers
@pytest.mark.parametrize("num_workers", [0])
def test_admet_model_drugbank(num_workers: int) -> None:
    """Test predictions on DrugBank data using ADMETModel."""
    model = ADMETModel(
        include_physchem=True,
        drugbank_path=None,
        atc_code=None,
        num_workers=num_workers,
    )

    preds = model.predict(smiles=DRUGBANK_DATA["smiles"].tolist())

    assert len(preds.columns) == len(ADMET_DATA)

    for column in preds.columns:
        assert np.allclose(preds[column].values, DRUGBANK_DATA[column].values), f"Column {column} does not match"
