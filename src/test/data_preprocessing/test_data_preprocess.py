import pytest
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
from genrelabeller.data_preprocessing.data_preprocess import DataPreprocess


@pytest.fixture
def sample_clean_data():
    return pd.DataFrame(
        {
            "trackID": [1, 2, 3, 4],
            "title": ["Song A", "Song B", "Song C", "Song D"],
            "tags": ["happy,upbeat", "sad,melancholy", "energetic", "rock"],
            "title_tokenized": [
                ["song", "a"],
                ["song", "b"],
                ["song", "c"],
                ["song", "d"],
            ],
            "tags_tokenized": [
                ["happy", "upbeat"],
                ["sad", "melancholy"],
                ["energetic"],
                ["rock"],
            ],
            "title_embedding": [np.random.rand(10) for _ in range(4)],
            "tags_embedding": [np.random.rand(10) for _ in range(4)],
            "genre": ["rock", "rock", "jazz", "jazz"],
            "vect1": [0.1, 0.2, 0.3, 0.4],
            "vect2": [1.1, 1.2, 1.3, 1.4],
        }
    )


@pytest.fixture
def sample_params():
    return OmegaConf.create(
        {
            "test_size": 0.5,
            "random_state": 42,
            "batch_size": 32,
            "scaler_path": None,
            "load_scaler_path": None,
            "inference_mode": False,
        }
    )


def test_initialisation(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)

    assert dp.clean_data.equals(sample_clean_data)
    assert dp.params == sample_params
    assert dp.test_size == 0.5
    assert dp.random_state == 42
    assert dp.batch_size == 32
    assert dp.inference_mode is False


def test_train_test_split(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split()

    assert len(dp.train_input) == 2
    assert len(dp.train_label) == 2
    assert len(dp.val_input) == 2
    assert len(dp.val_label) == 2
    assert "genre" not in dp.train_input.columns
    assert "genre" not in dp.val_input.columns


def test_one_hot_encode_labels(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().one_hot_encode_labels()

    assert "genre_rock" in dp.train_label.columns
    assert "genre_jazz" in dp.train_label.columns
    assert "genre_rock" in dp.val_label.columns
    assert "genre_jazz" in dp.val_label.columns


def test_separate_embeddings(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().separate_embeddings()

    assert dp.train_title_embeddings.shape == (2, 10)
    assert dp.val_tags_embeddings.shape == (2, 10)


def test_drop_trackid_train(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().drop_trackid_train()

    assert "trackID" not in dp.train_input.columns
    assert "trackID" not in dp.train_label.columns


def test_drop_trackid_val(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().drop_trackid_val()

    assert "trackID" not in dp.val_input.columns
    assert "trackID" not in dp.val_label.columns


def test_combine_features(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().separate_embeddings().combine_features()

    assert dp.train_input_values.shape[1] == dp.train_input.shape[1] + 20
    assert dp.val_input_values.shape[1] == dp.val_input.shape[1] + 20


def test_create_datasets(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().separate_embeddings().combine_features().create_datasets()

    assert isinstance(dp.train_dataset, torch.utils.data.Dataset)
    assert isinstance(dp.val_dataset, torch.utils.data.Dataset)


def test_create_dataloaders(sample_clean_data, sample_params):
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.train_test_split().separate_embeddings().combine_features().create_datasets().create_dataloaders()

    assert isinstance(dp.train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(dp.val_dataloader, torch.utils.data.DataLoader)
    assert dp.train_dataloader.batch_size == 32


def test_inference_mode(sample_clean_data, sample_params):
    sample_params.inference_mode = True
    dp = DataPreprocess(clean_data=sample_clean_data, params=sample_params)
    dp.__post_init__()

    assert dp.inference_mode is True
    assert dp.val_data.equals(dp.clean_data)
