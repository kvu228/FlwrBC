import argparse
import os
import hashlib
import numpy as np
import random
import tensorflow as tf
import flwr as fl

from blockchain_service import *
from verify import *

blockchainService = BlockchainService()


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test,client_id, client_address):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = client_id
        self.client_address = client_address

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        # """Get parameters of the local model."""
        # raise Exception("Not implemented (server-side parameter initialization)")
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        session: int = config["session"]
        round: int = config["round"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
            "client_address": self.client_address,            
        }

        # Save training weights in the created directory
        if not (os.path.exists(f'../Client/Local-weights')):
            os.mkdir(f"../Client/Local-weights")

        if not (os.path.exists(f'../Client/Local-weights/Client-{self.client_id}')):
            os.mkdir(f"../Client/Local-weights/Client-{self.client_id}")

        if not (os.path.exists(f'../Client/Local-weights/Client-{self.client_id}/Session-{session}')):
            os.mkdir(f"../Client/Local-weights/Client-{self.client_id}/Session-{session}")       

        filename = f'../Client/Local-weights/Client-{self.client_id}/Session-{session}/Round-{round}-training-weights.npy'
        np.save(filename, parameters_prime)
        with open(filename,"rb") as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest() #hash the file
            print(readable_hash)

        bcResult = blockchainService.addWeight(_session=session,_round_num=round, _dataSize=num_examples_train, _filePath = filename, _fileHash = readable_hash, client_id=self.client_id)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)
        session: int = config["session"]
        round: int = config["round"]

        # Get config values
        steps: int = config["val_steps"]

        # Get global weights
        global_rnd_model = self.model.get_weights()

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)

        # Create directory for global weights
        try:
            if not (os.path.exists(f'../Client/Global-weights')):
                os.mkdir(f"../Client/Global-weights")

            if not (os.path.exists(f'../Client/Global-weights/Session-{session}')):
                os.mkdir(f"../Client/Global-weights/Session-{session}")

            filename = f'../Client/Global-weights/Session-{session}/Round-{round}-Global-weights.npy'
            if not (os.path.exists(filename)):
                np.save(filename, global_rnd_model)
        except NameError:
            print(NameError)

        return loss, num_examples_test, {"accuracy": accuracy}
