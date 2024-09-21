import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from genrelabeller.data_preprocessing.torch_datasets import MusicDataset
from genrelabeller.model.model import GenrePredictionModel


@pytest.fixture
def model_params():
    return OmegaConf.create(
        {
            "input_size": 370,
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 8,
            "epochs": 1,
            "lr": 0.005,
            "num_fc_layers": 1,
        }
    )


@pytest.fixture
def sample_data():
    X_train = torch.randn(100, 370)
    y_train = torch.nn.functional.one_hot(
        torch.randint(0, 8, (100,)), num_classes=8
    ).float()
    X_val = torch.randn(20, 370)
    y_val = torch.nn.functional.one_hot(
        torch.randint(0, 8, (20,)), num_classes=8
    ).float()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    return train_dataloader, val_dataloader


def test_model_initialization(model_params):
    model = GenrePredictionModel(model_params)
    assert model.input_size == 370
    assert model.hidden_size == 64
    assert model.output_size == 8
    assert model.num_fc_layers == 1
    assert isinstance(model.input_layer, torch.nn.Linear)
    assert isinstance(model.fc, torch.nn.Sequential)
    assert isinstance(model.output_layer, torch.nn.Linear)


def test_forward_pass(model_params):
    model = GenrePredictionModel(model_params)
    x = torch.randn(10, 370)
    output = model(x)
    assert output.shape == (10, 8)


def test_training_loop(model_params, sample_data):
    train_dataloader, val_dataloader = sample_data
    model = GenrePredictionModel(model_params)

    model.fit(train_dataloader, val_dataloader)

    assert len(model.epoch_losses) == model_params.epochs
    assert len(model.val_losses) == model_params.epochs
    assert model.best_model_state is not None
    assert isinstance(model.best_model_state, dict)


def test_predict(model_params, sample_data):
    train_dataloader, val_dataloader = sample_data
    model = GenrePredictionModel(model_params)

    model.fit(train_dataloader, val_dataloader)
    predictions, labels = model.predict(val_dataloader)

    assert predictions.shape[0] == len(val_dataloader.dataset)
    assert labels.shape[0] == len(val_dataloader.dataset)
    assert torch.max(predictions) <= 7
    assert torch.min(predictions) >= 0


def test_predict_no_labels(model_params):
    X_test = torch.randn(20, 370)
    test_dataset = MusicDataset(X_test, None)
    test_dataloader = DataLoader(
        test_dataset, batch_size=5, shuffle=False, collate_fn=_custom_collate_fn
    )

    model = GenrePredictionModel(model_params)
    predictions, labels = model.predict(test_dataloader)

    assert predictions.shape[0] == len(test_dataloader.dataset)
    assert labels is None
    assert torch.max(predictions) <= 7
    assert torch.min(predictions) >= 0


def _custom_collate_fn(batch):
    """Custom collate function to handle None labels."""
    inputs = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    if all(label is None for label in labels):
        labels = None
    else:
        labels = torch.stack([item[1] for item in batch])
    return inputs, labels
