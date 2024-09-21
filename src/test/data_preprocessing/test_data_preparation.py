import numpy as np
import pandas as pd
import nltk
import pytest
from gensim.models import Word2Vec
from omegaconf import OmegaConf

from genrelabeller.data_preprocessing.data_preparation import DataPreparation


@pytest.fixture
def sample_data():
    input_data = pd.DataFrame(
        {
            "trackID": [1, 2, 3],
            "title": ["Song A", "Song B", "Song C"],
            "tags": ["happy,upbeat", "sad,melancholy", "energetic"],
        }
    )

    label_data = pd.DataFrame(
        {"trackID": [1, 2, 3], "time_signature": [4, 3, 4], "key": [1, 2, 3]}
    )

    params = OmegaConf.create(
        {
            "word_embedding_size": 10,
            "embedding_window": 3,
            "embedding_min_count": 1,
            "skipgram": 0,
            "save_path": None,
            "inference_mode": False,
        }
    )

    return input_data, label_data, params


def test_merge_input_and_label(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)

    dp.merge_input_and_label()

    assert not dp.data.empty
    assert len(dp.data.columns) == len(input_data.columns) + len(label_data.columns) - 1
    assert "trackID" in dp.data.columns


def test_merge_input_and_label_raises_value_error_on_mismatch_rows(sample_data):
    input_data, label_data, params = sample_data
    input_data = input_data.drop(index=2)

    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)

    with pytest.raises(
        ValueError, match="The number of rows in the input and label data are not equal"
    ):
        dp.merge_input_and_label()


def test_drop_na(sample_data):
    input_data, label_data, params = sample_data
    input_data.loc[1, "title"] = np.nan

    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().drop_na()

    assert dp.data.shape[0] == 2


def test_remove_stopwords_title(sample_data):
    input_data, label_data, params = sample_data
    nltk.download("stopwords")
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    dp.merge_input_and_label().remove_stopwords_title()

    assert stop_words not in dp.data["title"].str.lower().values


def test_remove_stopwords_tags(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().remove_stopwords_tags()

    assert "stopwords" not in dp.data["tags"].str.lower().values


def test_get_word_embeddings_title(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().get_word_embeddings_title()

    assert "title_embedding" in dp.data.columns
    assert dp.data["title_embedding"].apply(lambda x: isinstance(x, np.ndarray)).all()


def test_get_word_embeddings_tags(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().get_word_embeddings_tags()

    assert "tags_embedding" in dp.data.columns
    assert dp.data["tags_embedding"].apply(lambda x: isinstance(x, np.ndarray)).all()


def test_one_hot_encode_time_sig(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().one_hot_encode_time_sig()

    assert "time_sig_4" in dp.data.columns
    assert "time_sig_3" in dp.data.columns


def test_one_hot_encode_key(sample_data):
    input_data, label_data, params = sample_data
    dp = DataPreparation(input_data=input_data, label_data=label_data, params=params)
    dp.merge_input_and_label().one_hot_encode_key()

    assert "key_1" in dp.data.columns
    assert "key_2" in dp.data.columns
    assert "key_3" in dp.data.columns
