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

FIRST_DRUGBANK_ROW = DRUGBANK_DATA.iloc[0]
FIRST_DRUGBANK_SMILES = FIRST_DRUGBANK_ROW["smiles"]


class TestADMETPredict:
    def test_admet_predict_single_smiles(self) -> None:
        """Test admet_predict with the first DrugBank SMILES."""
        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            preds_path = Path(temp_dir) / "preds.csv"

            pd.DataFrame({"smiles": [FIRST_DRUGBANK_SMILES]}).to_csv(data_path, index=False)

            admet_predict(
                data_path=data_path,
                save_path=preds_path,
                include_physchem=True,
                drugbank_path=None,
                atc_code=None,
                num_workers=0,
            )

            preds = pd.read_csv(preds_path).set_index("smiles")

            assert len(preds) == 1
            assert len(preds.columns) == len(ADMET_DATA)

            for column in preds.columns:
                assert np.allclose(preds[column].values, FIRST_DRUGBANK_ROW[column]), f"Column {column} does not match"

    # TODO: move workers
    @pytest.mark.parametrize("num_workers", [0])
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_predict_drugbank(self, num_workers: int, include_physchem: bool) -> None:
        """Test predictions on DrugBank data using admet_predict."""
        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            preds_path = Path(temp_dir) / "preds.csv"

            drugbank_smiles = DRUGBANK_DATA[["smiles"]]
            drugbank_smiles.to_csv(data_path, index=False)

            admet_predict(
                data_path=data_path,
                save_path=preds_path,
                include_physchem=include_physchem,
                drugbank_path=None,
                atc_code=None,
                num_workers=num_workers,
            )

            preds = pd.read_csv(preds_path).set_index("smiles")

            expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]
            assert len(preds.columns) == len(expected)

            for column in preds.columns:
                assert np.allclose(
                    preds[column].values, DRUGBANK_DATA[column].values
                ), f"Column {column} does not match"


class TestADMETModel:
    def test_admet_model_single_smiles(self) -> None:
        """Test ADMETModel.predict with the first DrugBank SMILES."""
        model = ADMETModel(
            include_physchem=True,
            drugbank_path=None,
            atc_code=None,
            num_workers=0,
        )

        preds = model.predict(smiles=FIRST_DRUGBANK_SMILES)

        assert isinstance(preds, dict)
        assert len(preds) == len(ADMET_DATA)

        for key in preds.keys():
            assert np.allclose(preds[key], FIRST_DRUGBANK_ROW[key]), f"{key} prediction does not match"

    # TODO: more workers
    @pytest.mark.parametrize("num_workers", [0])
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_model_drugbank(self, num_workers: int, include_physchem: bool) -> None:
        """Test predictions on DrugBank data using ADMETModel."""
        model = ADMETModel(
            include_physchem=include_physchem,
            drugbank_path=None,
            atc_code=None,
            num_workers=num_workers,
        )

        preds = model.predict(smiles=DRUGBANK_DATA["smiles"].tolist())

        expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]
        assert len(preds.columns) == len(expected)

        for column in preds.columns:
            assert np.allclose(preds[column].values, DRUGBANK_DATA[column].values), f"Column {column} does not match"
