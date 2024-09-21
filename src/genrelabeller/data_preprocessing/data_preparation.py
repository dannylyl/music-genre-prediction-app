"""Module to clean the raw data for the downstream processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class DataPreparation:
    """
    A class to clean and prepare raw data for downstream processing.

    This class provides methods to merge input and label data, handle missing values,
    remove stopwords from text columns, generate word embeddings, and one-hot encode
    categorical variables.

    Attributes
    ----------
    input_data : pd.DataFrame
        The input DataFrame containing raw data features.
    label_data : pd.DataFrame
        The label DataFrame containing target variables.
    params : DictConfig
        A configuration object for various parameters.
    data : pd.DataFrame
        The combined DataFrame after processing input and label data.

    Notes
    -----
    The class is designed to be used with the `@dataclass` decorator, which reduces the
    boilerplate code required to define the class attributes and methods.

    The class returns self in each method to allow method chaining on a pipeline level.
    """

    input_data: pd.DataFrame = field()
    params: DictConfig = field()
    label_data: pd.DataFrame = field(default=None)

    data: pd.DataFrame = field(init=False)
    word_embedding_size: int = field(default=100)
    embedding_window: int = field(default=5)
    embedding_min_count: int = field(default=1)
    skipgram: int = field(default=0)
    save_path: Path = field(init=False)
    inference_mode: bool = field(default=False)

    def __post_init__(self):
        """
        Initialise the DataPreparation class.

        Class attributes are initialised with values from the configuration object, or
        with default values if not provided.

        nltk.download("stopwords") is called to download the stopwords corpus on class
        instantiation.
        """
        self.word_embedding_size = self.params.get("word_embedding_size", 100)
        self.embedding_window = self.params.get("embedding_window", 5)
        self.embedding_min_count = self.params.get("embedding_min_count", 1)
        self.skipgram = self.params.get("skipgram", 0)
        self.save_path = self.params.get("save_path", None)
        self.inference_mode = self.params.get("inference_mode", False)

        if self.inference_mode:
            self.data = self.input_data

        nltk.download("stopwords")
        pass

    def merge_input_and_label(self) -> DataPreparation:
        """
        Concatenate the input and label data on 'trackID'.

        Raises
        ------
        ValueError
            If the number of rows in input and label data are not equal.
        ValueError
            If the trackIDs in the input and label data do not match.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with merged data.
        """
        logger.debug("Merging the input and label data on the 'trackID' column")
        if self.input_data.shape[0] != self.label_data.shape[0]:
            raise ValueError(
                "The number of rows in the input and label data are not equal"
            )
        if not (self.input_data["trackID"].isin(self.label_data["trackID"]).all()):
            raise ValueError("The trackIDs in the input and label data do not match")
        self.data = pd.merge(self.input_data, self.label_data, on="trackID", how="left")
        logger.debug("Successfully merged the input and label data")
        return self

    def drop_na(self) -> DataPreparation:
        """
        Drop rows with missing values from the dataset.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with missing values
            dropped.
        """
        self.data = self.data.dropna()
        self.data = self.data
        return self

    def remove_stopwords_title(self) -> DataPreparation:
        """
        Remove stopwords from the 'title' column.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with stopwords removed
            from the 'title' column.
        """
        logger.debug("Removing the stopwords from the column 'title'")
        self.data["title"] = self.data["title"].apply(self._remove_stopwords_colwise)
        logger.debug("Successfully removed the stopwords from the 'title' column")
        return self

    def remove_stopwords_tags(self) -> DataPreparation:
        """
        Remove stopwords from the 'tags' column.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with stopwords removed
            from the 'tags' column.
        """
        logger.debug("Removing the stopwords from the 'tags' column")
        self.data["tags"] = self.data["tags"].apply(
            self._remove_stopwords_colwise, comma_separated=True
        )
        logger.debug("Successfully removed the stopwords from the 'tags' column")
        return self

    def get_word_embeddings_title(self) -> DataPreparation:
        """
        Generate word embeddings for the 'title' column.

        Uses the Word2Vec model to generate word embeddings based on the preprocessed
        title text.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with word embeddings
            generated for the 'title' column.
        """
        logger.debug("Getting the word embeddings for the 'title' column")
        self.data["title_tokenized"] = self.data["title"].apply(simple_preprocess)
        self.title_model = Word2Vec(
            sentences=self.data["title_tokenized"],
            vector_size=self.word_embedding_size,
            window=self.embedding_window,
            min_count=self.embedding_min_count,
            workers=4,
            sg=self.skipgram,
        )
        self.data["title_embedding"] = self.data["title_tokenized"].apply(
            lambda x: self._get_word_embeddings_colwise(x, self.title_model)
        )

        logger.debug("Successfully got the word embeddings for the 'title' column")
        return self

    def get_word_embeddings_tags(self) -> DataPreparation:
        """
        Generate word embeddings for the 'tags' column.

        Uses the Word2Vec model to generate word embeddings based on the preprocessed
        tags text.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with word embeddings
            generated for the 'tags' column.
        """
        logger.debug("Getting the word embeddings for the 'tags' column")
        self.data["tags_tokenized"] = self.data["tags"].apply(simple_preprocess)
        self.tags_model = Word2Vec(
            sentences=self.data["tags_tokenized"],
            vector_size=self.word_embedding_size,
            window=self.embedding_window,
            min_count=self.embedding_min_count,
            workers=4,
            sg=self.skipgram,
        )
        self.data["tags_embedding"] = self.data["tags_tokenized"].apply(
            lambda x: self._get_word_embeddings_colwise(x, self.tags_model)
        )

        logger.debug("Successfully got the word embeddings for the 'tags' column")
        return self

    def one_hot_encode_time_sig(self) -> DataPreparation:
        """
        One-hot encode the 'time_signature' column.

        Uses `pd.get_dummies` to convert the 'time_signature' column into one-hot
        encoded format. I use pd.get_dummies purely for convenience, but I could have
        used `sklearn.preprocessing.OneHotEncoder`.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with one-hot encoded
            'time_signature' column.
        """
        logger.debug("One hot encoding the 'time_signature' column")
        self.data = pd.get_dummies(
            self.data, columns=["time_signature"], prefix="time_sig"
        )
        logger.debug("Successfully one hot encoded the 'time_signature' column")
        return self

    def one_hot_encode_key(self) -> DataPreparation:
        """
        One-hot encode the 'key' column.

        Uses `pd.get_dummies` to convert the 'key' column into one-hot encoded format. I
        use pd.get_dummies purely for convenience, but I could have used
        `sklearn.preprocessing.OneHotEncoder`.

        Returns
        -------
        DataPreparation
            The current instance of the DataPreparation class with one-hot encoded 'key'
            column.
        """
        logger.debug("One hot encoding the 'key' column")
        self.data = pd.get_dummies(self.data, columns=["key"], prefix="key")
        logger.debug("Successfully one hot encoded the 'key' column")
        return self

    def _remove_stopwords_colwise(self, text, comma_separated=False) -> str:
        """
        Remove stopwords from the given text.

        This private method removes stopwords from a text string. If the text contains
        comma-separated values, it splits the text on commas; otherwise, it splits on
        whitespace.

        Parameters
        ----------
        text : str
            The text string from which stopwords will be removed.
        comma_separated : bool, default=False
            If True, the text is split by commas, if False, by whitespace.

        Returns
        -------
        str
            The text with stopwords removed.
        """
        stop_words = set(nltk.corpus.stopwords.words("english"))
        if comma_separated:
            words = text.split(",")
        else:
            words = text.split()
        filtered_words = [
            word.strip() for word in words if word.strip().lower() not in stop_words
        ]
        return " ".join(filtered_words)

    def _get_word_embeddings_colwise(self, text, model) -> np.ndarray:
        """
        Generate word embeddings for the given text.

        This private method computes the mean word embedding for a list of words using a
        pre-trained Word2Vec model.

        Parameters
        ----------
        text : list of str
            A list of words for which embeddings will be generated.
        model : Word2Vec
            A pre-trained Word2Vec model.

        Returns
        -------
        np.ndarray
            The mean word embedding vector for the given text.
        """
        valid_words = [word for word in text if word in model.wv.key_to_index]
        if not valid_words:
            logger.warning("No valid words found in the text")
            return np.zeros(model.vector_size)
        return np.mean([model.wv[word] for word in valid_words], axis=0)
