"""Training pipeline for the Genrelabelling Model."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

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
config_name = os.getenv("CONFIG_NAME", "train_config.yaml")


@dataclass
class TrainingPipeline:

    params: DictConfig = field()
    input_data: pd.DataFrame = field(init=False)
    label_data: pd.DataFrame = field(init=False)

    base_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    datapreparation_params: DictConfig = field(init=False)
    datapreprocessing_params: DictConfig = field(init=False)
    model_params: DictConfig = field(init=False)
    random_state: int = field(init=False)

    clean_data: pd.DataFrame = field(init=False)
    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)
    model: genrelabeller.model.model.GenrePredictionModel = field(init=False)
    y_pred: list[torch.Tensor] = field(init=False)
    y_true: list[torch.Tensor] = field(init=False)

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
        """Main training pipeline for the model."""
        logger.info("Starting the Training Pipeline.")

        #### LOAD DATA
        logger.info("Loading the Datasets.")
        self.load_data()
        logger.info("Datasets Successfully Loaded.")

        #### PREPARE DATA
        logger.info("Preparing the Datasets.")
        self.prepare_data()
        logger.info("Datasets Successfully Prepared.")

        #### PREPROCESS DATA
        logger.info("Preprocessing the Datasets.")
        self.preprocess_data()
        logger.info("Datasets Successfully Preprocessed.")

        #### TRAIN MODEL
        logger.info("Training the Model.")
        self.train_model().plot_losses().save_model()
        logger.info("Model Successfully Trained and Saved.")

        #### PREDICT WITH MODEL ON VALIDATION SET
        logger.info("Predicting with the Trained Model.")
        self.model_predict()
        logger.info("Model Prediction Complete.")

        #### EVALUATE MODEL
        logger.info("Evaluating the Model.")
        self.evaluate_model()
        logger.info("Successfully Evaluated the Model.")

        logger.info("Training Pipeline Complete.")

    def load_data(self) -> TrainingPipeline:
        """Load the input and label data for training."""
        logger.info("Loading the Input Data.")
        self.input_data = pd.read_csv(self.data_dir / "features.csv")
        logger.info("Loading the Label Data.")
        self.label_data = pd.read_csv(self.data_dir / "labels.csv")
        logger.info("Input and Label Data Loaded.")
        return self

    def prepare_data(self) -> TrainingPipeline:
        """Prepare the input and label data for training."""
        logger.info("Commencing Data Preparation.")
        data_preparation = (
            genrelabeller.data_preprocessing.data_preparation.DataPreparation(
                input_data=self.input_data,
                label_data=self.label_data,
                params=self.dataprepation_params,
            )
        )
        (
            data_preparation.merge_input_and_label()
            .drop_na()
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

    def preprocess_data(self) -> TrainingPipeline:
        """Preprocess the input and label data for training."""
        logger.info("Commencing Data Preprocessing.")
        data_preprocessor = (
            genrelabeller.data_preprocessing.data_preprocess.DataPreprocess(
                clean_data=self.clean_data,
                params=self.datapreprocessing_params,
            )
        )
        data_preprocessor.scaler_path = self.artifacts_dir
        (
            data_preprocessor.train_test_split()
            .one_hot_encode_labels()
            .separate_embeddings()
            .scale_data()
            .drop_trackid_train()
            .drop_trackid_val()
            .combine_features()
            .create_datasets()
            .create_dataloaders()
        )
        self.train_dataloader = data_preprocessor.train_dataloader
        self.val_dataloader = data_preprocessor.val_dataloader
        logger.info("Data Preprocessing Completed.")
        return self

    def train_model(self) -> TrainingPipeline:
        """Train the model on the preprocessed data."""
        logger.info("Commencing Model Training.")
        self.model = genrelabeller.model.model.GenrePredictionModel(
            params=self.model_params
        )
        self.model.fit(self.train_dataloader, self.val_dataloader)
        logger.info("Model Training Completed.")
        return self

    def plot_losses(self) -> TrainingPipeline:
        """Plot the training and validation losses."""
        logger.info("Commencing Loss Plotting.")
        train_losses = self.model.epoch_losses
        val_losses = self.model.val_losses
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training and Validation Losses")
        plt.legend(loc="upper right")
        plt.savefig(self.model_dir / "losses.png")
        logger.info("Loss Plotting Completed.")
        return self

    def save_model(self) -> TrainingPipeline:
        """Save the trained model."""
        logger.info("Commencing Model Saving.")
        model_state = self.model.state_dict()
        torch.save(model_state, self.model_dir / "model_weights.pth")
        logger.info("Model Successfully Saved.")
        return self

    def model_predict(self) -> TrainingPipeline:
        logger.info("Commencing Prediction on The Validation Set with Trained Model.")
        self.y_pred, self.y_true = self.model.predict(self.val_dataloader)
        logger.info("Model Prediction Completed.")
        return self

    def evaluate_model(self) -> TrainingPipeline:
        logger.info("Commencing Model Evaluation.")
        genre_mapping = self.params.get("genre_mapping")
        cr = classification_report(
            self.y_true, self.y_pred, target_names=genre_mapping.values()
        )
        logger.info(cr)
        logger.info("Model Evaluation Completed.")
        return self


def main():
    """Main function to run the training pipeline."""
    params = omegaconf.OmegaConf.load(config_path / config_name)
    training_pipeline = TrainingPipeline(params=params)
    training_pipeline.main()


if __name__ == "__main__":
    main()
