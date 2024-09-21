"""Module with the PyTorch Neural Network Model For Section B of the Assignment."""

import logging
from copy import deepcopy
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GenrePredictionModel(torch.nn.Module):
    """
    PyTorch Neural Network Model For Section B of the DBS ML Engineer Assignment.

    The neural network model built is a simple feedforward neural network with a
    configurable number of hidden layers and units. The model is instantiated with a
    omegaconf DictConfig object that contains the hyperparameters for the model.

    As the problem is a multiclass classification problem, the output layer of the model
    has 8 output units, each corresponding to a genre. The model is trained using the
    Pytorch Cross Entropy Loss function, which expects logits as input, which is why the
    softmax activation function is not applied to the output layer of the model in the
    training loop.
    """

    def __init__(self, params: DictConfig) -> None:
        """Initialise the GenrePredictionModel."""
        logger.debug("Initialising GenrePredictionModel.")
        super(GenrePredictionModel, self).__init__()
        self.input_size: int = params.get("input_size", 370)
        self.hidden_size: int = params.get("hidden_size", 64)
        self.output_size: int = params.get("output_size", 8)
        self.epochs: int = params.get("epochs", 10)
        self.lr: float = params.get("lr", 0.005)
        self.num_fc_layers: int = params.get("num_fc_layers", 1)

        self.device = torch.device("cpu")
        self.to(self.device)

        # Build the model
        self.build_model()
        logger.debug("GenrePredictionModel Initialised.")

    def build_model(self) -> None:
        """
        Build the GenrePredictionModel.

        The model is built by defining the input layer, hidden layers, and output layer.
        The model is a simple feedforward neural network with a configurable number of
        hidden layers and units. The hidden layers are followed by ReLU activation. The
        output layer does not have an activation function applied to it as the Cross
        Entopy Loss function expects logits as input.
        """
        logger.debug("Building the GenrePredictionModel.")
        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        fc_layers = []
        for i in range(self.num_fc_layers):
            fc_layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            fc_layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*fc_layers)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
        logger.debug("GenrePredictionModel Built.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GenrePredictionModel.

        The forward pass of the model is simple since the model is a feedforward neural
        network. The input tensor is passed through the input layer, followed by the
        hidden layers, and finally the output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensors to the model.

        Returns
        -------
        torch.Tensor
            Output tensors from the neural network model.
        """
        out = self.input_layer(x)
        out = self.fc(out)
        out = self.output_layer(out)
        return out

    def fit(
        self,
        train_dataloader,
        val_dataloader,
    ) -> None:
        """
        Fit method of the GenrePredictionModel.

        The fit method trains the model on the training data and validates it on the
        validation data. The model is trained using the Adam optimiser and the Cross
        Entropy Loss function. The model is trained for a configurable number of epochs
        and the best model state is saved based on the validation loss.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader object for the training data, containing the input and label
            tensors. The DataLoader object is created using the MusicDataset class from
            the torch_datasets module.
        val_dataloader : torch.utils.data.DataLoader
            DataLoader object for the validation data, containing the input and label
            tensors. The DataLoader object is created using the MusicDataset class from
            the torch_datasets module.

        Notes
        -----
        The model is trained on the CPU by default. I didn't include GPU training as I
        am working on my personal laptop which is a CPU-only machine, and I didn't
        want to risk running into any possibilities of the code failing on a machine
        that has a GPU since I can't test it.
        """
        logger.debug("Fit method called on the GenrePredictionModel.")
        logger.debug(f"Using device: {self.device}")
        device = self.device
        criterion = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
        epochs = self.epochs
        epoch_loss = 0
        self.epoch_losses = []
        self.val_losses = []
        best_val_loss = float("inf")
        self.best_model_state = None

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for i, batch in tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
            ):

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimiser.zero_grad()
                predictions = self(inputs)
                # Convert one-hot to class index, torch.CrossEntropyLoss expects can
                # work with both one-hot and class indexes, but it's easier to work with
                # class indexes for downstream processing.
                labels = torch.argmax(labels, dim=1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_dataloader)
            self.epoch_losses.append(epoch_loss)
            logger.debug("Epoch {}: Training Loss: {}".format(epoch + 1, epoch_loss))

            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    predictions = self(inputs)
                    # Same thing here, convert one-hot to class index.
                    labels = torch.argmax(labels, dim=1)
                    loss = criterion(predictions, labels)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_dataloader)
            self.val_losses.append(epoch_val_loss)
            logger.debug("Validation Loss: {}".format(epoch_val_loss))

            # Update the best model state based on the validation loss to load at the
            # end of training and save as well.
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                self.best_model_state = deepcopy(self.state_dict())
                logger.debug("Best Model State Updated.")

        self.load_state_dict(self.best_model_state)
        logger.debug("Best Model State Loaded At End Of Training.")
        logger.debug("GenrePredictionModel Training Complete.")
        pass

    def predict(self, test_dataloader) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict method of the GenrePredictionModel.

        The predict method predicts the labels for the test data using the trained
        model. If the test data contains labels, the method also returns the true labels
        for easier evaluation of the model.

        Parameters
        ----------
        test_dataloader : torch.utils.data.DataLoader
            DataLoader object for the test data, containing the input tensors.
            The DataLoader object is created using the MusicDataset class from the
            torch_datasets module. If the test data contains labels, the DataLoader will
            also contain the label tensors.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            A tuple containing the predicted labels and the true labels. If the test
            data does not contain labels, the true labels are None.
        """
        logger.debug("Predict method called on the GenrePredictionModel.")
        device = torch.device("cpu")
        logger.debug(f"Using device: {device}")
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                if labels is not None:
                    labels = labels.to(device)
                outputs = self(inputs)
                # Apply softmax to get the probabilities for each class.
                outputs = torch.softmax(outputs, dim=1)
                predictions = outputs
                all_predictions.extend(predictions.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                else:
                    all_labels = None
            all_predictions = torch.argmax(torch.tensor(all_predictions), dim=1)
            if all_labels is not None:
                all_labels = torch.argmax(torch.tensor(all_labels), dim=1)
        logger.debug("Prediction Complete")

        return all_predictions, all_labels
