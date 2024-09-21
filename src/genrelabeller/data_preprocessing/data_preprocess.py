"""Module to preprocess the cleaned data for model training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from genrelabeller.data_preprocessing.torch_datasets import MusicDataset
from genrelabeller.utils import load_object, save_object

logger = logging.getLogger(__name__)


@dataclass
class DataPreprocess:
    """
    Class to preprocess the cleaned data for model training.

    This class provides methods to split data, encode labels, scale features,
    separate embeddings, and prepare data for training with PyTorch.

    Attributes
    ----------
    clean_data : pd.DataFrame
        The cleaned DataFrame that contains the features and labels.
    params : DictConfig
        A configuration object containing parameters such as test size.

    Notes
    -----
    The class is designed to be used with the `@dataclass` decorator, which reduces
    the boilerplate code required to define the class attributes and methods.

    The class returns self in each method to allow method chaining on a pipeline
    level.

    If the model is in inference mode, the class will not split the data into training
    and validation sets, and will use the entire dataset for inference. For quick
    implementation, I assigned the test data as the validation data in inference mode,
    and the methods are called similarly to the training pipeline, but with some
    operations skipped.
    """

    clean_data: pd.DataFrame = field()
    params: DictConfig = field()

    test_size: float = field(init=False)
    random_state: int = field(init=False)
    batch_size: int = field(init=False)
    scaler_path: Path = field(init=False)
    load_scaler_path: Path = field(init=False)
    inference_mode: bool = field(init=False)

    train_data: pd.DataFrame = field(init=False)
    val_data: pd.DataFrame = field(init=False)
    train_input: pd.DataFrame = field(init=False)
    train_label: pd.DataFrame = field(init=False)
    val_input: pd.DataFrame = field(init=False)
    val_label: pd.DataFrame = field(init=False)
    scaler: StandardScaler = field(init=False)
    train_title_embeddings: np.ndarray = field(init=False)
    train_tags_embeddings: np.ndarray = field(init=False)
    val_title_embeddings: np.ndarray = field(init=False)
    val_tags_embeddings: np.ndarray = field(init=False)
    train_input_values: np.ndarray = field(init=False)
    val_input_values: np.ndarray = field(init=False)
    train_dataset: MusicDataset = field(init=False)
    val_dataset: MusicDataset = field(init=False)
    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        """Initialize the DataPreprocess class."""
        self.test_size = self.params.get("test_size", 0.2)
        self.random_state = self.params.get("random_state", 42)
        self.batch_size = self.params.get("batch_size", 32)
        self.scaler_path = self.params.get("scaler_path", None)
        self.load_scaler_path = self.params.get("load_scaler_path", None)
        self.inference_mode = self.params.get("inference_mode", False)

        if self.inference_mode:
            logger.debug("Inference Mode Enabled.")
            self.val_data = self.clean_data

    def train_test_split(self) -> DataPreprocess:
        """
        Split the data into training and testing sets.

        The data is split into training and validation sets using stratified sampling
        based on the 'genre' column to maintain the class distribution.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with updated training and
            validation sets.
        """
        logger.debug(
            f"Splitting the data into training and testing sets using test size: {self.test_size}"
        )
        self.train_data, self.val_data = train_test_split(
            self.clean_data,
            test_size=self.test_size,
            random_state=42,
            stratify=self.clean_data["genre"],
        )
        self.train_input = self.train_data.drop(
            columns=[
                "title",
                "tags",
                "title_tokenized",
                "tags_tokenized",
                "genre",
            ]
        )
        self.train_label = self.train_data[["trackID", "genre"]]
        self.val_input = self.val_data.drop(
            columns=[
                "title",
                "tags",
                "title_tokenized",
                "tags_tokenized",
                "genre",
            ]
        )
        self.val_label = self.val_data[["trackID", "genre"]]
        logger.debug("Successfully split the data into training and testing sets")
        return self

    def one_hot_encode_labels(self) -> DataPreprocess:
        """
        One-hot encode the labels in the training and validation sets.

        The 'genre' column in the labels is one-hot encoded to be used as the target
        variable for the model.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with one-hot encoded
            labels.
        """
        logger.debug("One-hot encoding the labels")
        self.train_label = pd.get_dummies(
            self.train_label, columns=["genre"], prefix="genre"
        )
        self.val_label = pd.get_dummies(
            self.val_label, columns=["genre"], prefix="genre"
        )
        logger.debug("Successfully one-hot encoded the labels")
        return self

    @load_object("loaded_scaler", "load_scaler_path")
    @save_object("scaler", "scaler_path", "scaler.pkl")
    def scale_data(self) -> DataPreprocess:
        """
        Scale the numerical features in the training and validation sets.

        The features are scaled using `StandardScaler`, excluding certain columns such
        as 'trackID'. The scaler is saved for future use.

        If the inference mode is enabled, the scaler is loaded from the provided path.
        And the validation data (test) is scaled using the loaded scaler.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with scaled data.
        """
        logger.debug("Scaling the Numerical Data")
        if not self.inference_mode:
            self.scaler = StandardScaler()
            columns_to_exclude = ["trackID"]
            columns_to_scale = self.train_input.columns.difference(columns_to_exclude)
            self.train_input[columns_to_scale] = self.scaler.fit_transform(
                self.train_input[columns_to_scale]
            )
            self.val_input[columns_to_scale] = self.scaler.transform(
                self.val_input[columns_to_scale]
            )

        else:
            logger.debug("Loading the scaler from the provided path")
            self.scaler = self.loaded_scaler
            if not hasattr(self.scaler, "mean_"):
                raise ValueError(
                    "Scaler has not been fitted. Please fit the scaler first."
                )
            logger.debug("Loaded the scaler from the provided path")
            columns_to_exclude = ["trackID"]
            columns_to_scale = self.val_input.columns.difference(columns_to_exclude)
            self.val_input[columns_to_scale] = self.scaler.transform(
                self.val_input[columns_to_scale]
            )

        logger.debug("Successfully Scaled the Numerical Data.")
        return self

    def separate_embeddings(self) -> DataPreprocess:
        """
        Separate the word embeddings from the input data.

        The embeddings for 'title' and 'tags' are separated from the input data, and
        the original columns are removed from the input DataFrame.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with separated embeddings.
        """
        logger.debug("Separating the embeddings from the input data")
        if not self.inference_mode:
            self.train_title_embeddings = np.vstack(
                self.train_data["title_embedding"].values
            )
            self.train_tags_embeddings = np.vstack(
                self.train_data["tags_embedding"].values
            )
            self.train_input = self.train_input.drop(
                columns=["title_embedding", "tags_embedding"]
            )
        self.val_title_embeddings = np.vstack(self.val_data["title_embedding"].values)
        self.val_tags_embeddings = np.vstack(self.val_data["tags_embedding"].values)

        if self.inference_mode:
            self.val_input = self.val_data
            self.val_input = self.val_input.drop(
                columns=[
                    "title",
                    "tags",
                    "title_tokenized",
                    "tags_tokenized",
                ]
            )
        self.val_input = self.val_input.drop(
            columns=["title_embedding", "tags_embedding"]
        )
        logger.debug("Successfully separated the embeddings from the input data")
        return self

    def drop_trackid_train(self) -> DataPreprocess:
        """
        Drops the 'trackID' column from the training input and label data.

        As 'trackID' is just a unique identifier and not a feature that would contribute
        to the model's performance, it is dropped from the input and label data.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with 'trackID' dropped
        """
        logger.debug("Dropping the 'trackID' column from the input and label data")
        self.train_input.drop(columns=["trackID"], inplace=True)
        self.train_label.drop(columns=["trackID"], inplace=True)
        return self

    def drop_trackid_val(self) -> DataPreprocess:
        """
        Drops the 'trackID' column from the validation input and label data.

        As 'trackID' is just a unique identifier and not a feature that would contribute
        to the model's performance, it is dropped from the input and label data.

        If the model is in inference mode, the 'trackID' column is not dropped from the
        labels since there are no labels.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with 'trackID' dropped
        """
        logger.debug("Dropping the 'trackID' column from the validation data")
        self.val_input.drop(columns=["trackID"], inplace=True)
        if not self.inference_mode:
            self.val_label.drop(columns=["trackID"], inplace=True)
        logger.debug(
            "Successfully dropped the 'trackID' column from the validation data"
        )
        return self

    def combine_features(self) -> DataPreprocess:
        """
        Combine the features and embeddings to create the final input for the model.

        The numerical features and embeddings are combined into a single array to be
        used as the input for the model.


        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with combined features.
        """
        logger.debug("Combining the features for the model input")
        if not self.inference_mode:
            self.train_input_values = np.hstack(
                (
                    self.train_input.values,
                    self.train_title_embeddings,
                    self.train_tags_embeddings,
                )
            )
        self.val_input_values = np.hstack(
            (
                self.val_input.values,
                self.val_title_embeddings,
                self.val_tags_embeddings,
            )
        )
        logger.debug("Successfully combined the features for the model input")
        return self

    def create_datasets(self) -> DataPreprocess:
        """
        Create PyTorch datasets for the training and validation data.

        The combined features and one-hot encoded labels are used to create
        `MusicDataset` instances for both the training and validation sets.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with created datasets.
        """
        logger.debug("Creating the torch datasets for model training")
        if not self.inference_mode:
            self.train_dataset = MusicDataset(
                input_data=self.train_input_values,
                target_data=self.train_label.values,
            )
            self.val_dataset = MusicDataset(
                input_data=self.val_input_values,
                target_data=self.val_label.values,
            )
        else:
            self.val_dataset = MusicDataset(
                input_data=self.val_input_values,
                target_data=None,
            )
        logger.debug("Successfully created the torch datasets for model training")
        return self

    def create_dataloaders(self) -> DataPreprocess:
        """
        Create PyTorch dataloaders for the training and validation datasets.

        The datasets are loaded into PyTorch dataloaders to be used during model
        training. The training dataloader shuffles the data, while the validation
        dataloader does not.

        Returns
        -------
        DataPreprocess
            The current instance of the DataPreprocess class with created dataloaders.
        """
        logger.debug("Creating the torch dataloaders for model training")
        if not self.inference_mode:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._custom_collate_fn,
            )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._custom_collate_fn,
        )
        logger.debug("Successfully created the torch dataloaders for model training")
        return self

    def _custom_collate_fn(self, batch):
        """Custom collate function to handle None labels."""
        inputs = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        if all(label is None for label in labels):
            labels = None
        else:
            labels = torch.stack([item[1] for item in batch])
        return inputs, labels
