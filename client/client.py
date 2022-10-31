import argparse
from fileinput import filename
import os
from pathlib import Path
import hashlib
import tensorflow as tf
import numpy as np
import flwr as fl


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test,client_id):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = client_id

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

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
        }

        # Save training weights in the created directory
        if not (os.path.exists(f'./client/Local-weights')):
            os.mkdir(f"./client/Local-weights")

        if not (os.path.exists(f'./client/Local-weights/Client-{self.client_id}')):
            os.mkdir(f"./client/Local-weights/Client-{self.client_id}")

        if not (os.path.exists(f'./client/Local-weights/Client-{self.client_id}/Session-{session}')):
            os.mkdir(f"./client/Local-weights/Client-{self.client_id}/Session-{session}")       

        filename = f'./client/Local-weights/Client-{self.client_id}/Session-{session}/Round-{round}-training-weights.npy'
        np.save(filename, parameters_prime)
        with open(filename,"rb") as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest() #hash the file
            print(readable_hash)

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
        if not (os.path.exists(f'./client/Global-weights')):
            os.mkdir(f"./client/Global-weights")
        if not (os.path.exists(f'./client/Global-weights/Session-{session}')):
            os.mkdir(f"./client/Global-weights/Session-{session}")
        # if not (os.path.exists(f'./client/Global-weights/Client-{self.client_id}')):
        #     os.mkdir(f"./client/Global-weights/Client-{self.client_id}")
        # if not (os.path.exists(f'./client/Global-weights/Client-{self.client_id}/Session-{session}')):
        #     os.mkdir(f"./client/Global-weights/Client-{self.client_id}/Session-{session}")

        filename = f'./client/Global-weights/Session-{session}/Round-{round}-Global-weights.npy'
        if not (os.path.exists(filename)):
            np.save(filename, global_rnd_model)

        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 5), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.applications.EfficientNetB0(
        # Cifar10 is a dataset of 32x32 RGB color training images, labeled over 10 categories
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test,args.partition)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


def load_partition(client_id: int):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert client_id in range(5)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[client_id * 5000 : (client_id + 1) * 5000],
        y_train[client_id * 5000 : (client_id + 1) * 5000],
    ), (
        x_test[client_id * 1000 : (client_id + 1) * 1000],
        y_test[client_id * 1000 : (client_id + 1) * 1000],
    )


if __name__ == "__main__":
    main()