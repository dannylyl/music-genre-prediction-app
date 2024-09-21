"""Inference pipeline for the GenreLabelling Model."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import classification_report

import genrelabeller

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(os.getenv("CONFIG_PATH", "conf"))
config_name = os.getenv("CONFIG_NAME", "inference_config.yaml")


@dataclass
class InferencePipeline:

    params: DictConfig = field()
    input_data: pd.DataFrame = field(init=False)

    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    datapreparation_params: DictConfig = field(init=False)
    datapreprocessing_params: DictConfig = field(init=False)
    model_params: DictConfig = field(init=False)
    random_state: int = field(init=False)

    input_data: pd.DataFrame = field(init=False)
    track_id: pd.Series = field(init=False)
    clean_data: pd.DataFrame = field(init=False)
    test_dataloader: torch.utils.data.DataLoader = field(init=False)
    model: genrelabeller.model.model.GenrePredictionModel = field(init=False)
    y_pred: List[torch.Tensor] = field(init=False)
    inference_output: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if os.getenv("DATA_FOLDER_DIR") is not None:
            base_dir = Path(os.getenv("DATA_FOLDER_DIR"))
        else:
            base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = Path(base_dir) / "data"
        self.artifacts_dir = Path(base_dir) / "artifacts"
        self.processed_data_dir = Path(base_dir) / "processed"
        self.model_dir = self.artifacts_dir / "model"

        self.dataprepation_params = self.params["datapreparation"]
        self.datapreprocessing_params = self.params["datapreprocessing"]
        self.model_params = self.params["model"]
        self.random_state = self.params.get("random_state", 42)

        logger.info(f"Setting Random Seed to {self.random_state}.")
        torch.manual_seed(self.random_state)
        pass

    def main(self):
        """Main inference pipeline for the model."""
        logger.info("Starting the Inference Pipeline.")

        #### LOAD DATA
        logger.info("Loading the Test Dataset.")
        self.load_data()
        logger.info("Dataset Successfully Loaded.")

        #### PREPARE DATA
        logger.info("Preparing the Dataset.")
        self.prepare_data()
        logger.info("Dataset Successfully Prepared.")

        #### PREPROCESS DATA
        logger.info("Preprocessing the Dataset.")
        self.preprocess_data()
        logger.info("Dataset Successfully Preprocessed.")

        #### LOAD MODEL
        logger.info("Loading the Model.")
        self.load_model()
        logger.info("Model Successfully Loaded.")

        #### PREDICT WITH MODEL ON TEST SET
        logger.info("Predicting with the Trained Model.")
        self.model_predict()
        logger.info("Model Prediction Complete.")

        #### GET INFERENCE OUTPUT DATA AND SAVE
        logger.info("Getting the Output of Model Inference.")
        self.get_inference_output()
        logger.info(self.inference_output)
        logger.info("Successfully Saved Output of Model Inference.")

        logger.info("Inference Pipeline Complete.")

    def load_data(self) -> InferencePipeline:
        """Load the input data for inference."""
        logger.info("Loading the Input Data.")
        self.input_data = pd.read_csv(self.data_dir / "test.csv")
        self.track_id = self.input_data["trackID"]
        logger.info("Input Data Loaded.")
        return self

    def prepare_data(self) -> InferencePipeline:
        """Prepare the input and label data for inference."""
        logger.info("Commencing Data Preparation.")
        data_preparation = (
            genrelabeller.data_preprocessing.data_preparation.DataPreparation(
                input_data=self.input_data,
                params=self.dataprepation_params,
            )
        )
        data_preparation.input_data["key"] = data_preparation.input_data["key"].astype(
            float
        )
        data_preparation.input_data["time_signature"] = data_preparation.input_data[
            "time_signature"
        ].astype(float)
        (
            data_preparation.drop_na()
            .remove_stopwords_title()
            .remove_stopwords_tags()
            .get_word_embeddings_title()
            .get_word_embeddings_tags()
            .one_hot_encode_time_sig()
            .one_hot_encode_key()
        )
        self.clean_data = data_preparation.data
        logger.info("Data Preparation Complete.")
        return self

    def preprocess_data(self) -> InferencePipeline:
        """Preprocess the input for inference."""
        logger.info("Commencing Data Preprocessing.")
        data_preprocessor = (
            genrelabeller.data_preprocessing.data_preprocess.DataPreprocess(
                clean_data=self.clean_data,
                params=self.datapreprocessing_params,
            )
        )
        data_preprocessor.load_scaler_path = self.artifacts_dir / "scaler.pkl"
        (
            data_preprocessor.separate_embeddings()
            .scale_data()
            .drop_trackid_val()
            .combine_features()
            .create_datasets()
            .create_dataloaders()
        )
        self.test_dataloader = data_preprocessor.val_dataloader
        logger.info("Data Preprocessing Completed.")
        return self

    def load_model(self) -> InferencePipeline:
        """Load the trained model."""
        logger.info("Commencing Model Loading.")
        self.model = genrelabeller.model.model.GenrePredictionModel(self.model_params)
        model_state = torch.load(
            self.model_dir / "model_weights.pth", weights_only=True
        )
        self.model.load_state_dict(model_state)
        logger.info("Model Successfully Loaded.")
        return self

    def model_predict(self) -> InferencePipeline:
        logger.info("Commencing Prediction on The Test Set with Trained Model.")
        self.y_pred, self.y_true = self.model.predict(self.test_dataloader)
        logger.info("Model Prediction Completed.")
        return self

    def get_inference_output(self) -> InferencePipeline:
        """Get the inference output."""
        logger.info("Getting the Inference Output.")
        genre_mapping = self.params.get("genre_mapping")
        self.y_pred = [genre_mapping[int(i)] for i in self.y_pred]
        self.inference_output = pd.DataFrame(
            {
                "trackID": self.track_id,
                "genre": self.y_pred,
            }
        )
        self.inference_output.to_csv(self.data_dir / "prediction.csv", index=False)
        logger.debug(f"Saved Inference Output to {self.data_dir / 'prediction.csv'}.")
        return self


def main():
    """Main function to run the inference pipeline."""
    params = omegaconf.OmegaConf.load(config_path / config_name)
    inference_pipeline = InferencePipeline(params=params)
    inference_pipeline.main()


if __name__ == "__main__":
    main()
