from client import CifarClient
from blockchain_service import *

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import flwr as fl
import json
import argparse


# Parse command line argument `partition`
# parser = argparse.ArgumentParser(description="Flower")
# parser.add_argument("--partition", type=int, choices=range(0, 5), required=True)
# args = parser.parse_args()
# client_id = args.partition

blockchainService = BlockchainService()

app=FastAPI()


class FLlaunch:
    def start(self):
        listen_and_participate()


def load_dataset(client_id:int):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert client_id in range(5)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[client_id * 10000 : (client_id + 1) * 10000],
        y_train[client_id * 10000 : (client_id + 1) * 10000],
    ), (
        x_test[client_id * 1000 : (client_id + 1) * 1000],
        y_test[client_id * 1000 : (client_id + 1) * 1000],
    )

def handle_launch_FL_session(model,x_train, y_train, x_test, y_test, client_id, client_address):
    """
    handles smart contract's addStrategy event by starting flwr client
    """
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client = CifarClient(model, x_train, y_train, x_test, y_test, client_id, client_address), 
        grpc_max_message_length = 1024*1024*1024)



def load_model():
    model = tf.keras.applications.EfficientNetB0(
        # Cifar10 is a dataset of 32x32 RGB color training images, labeled over 10 categories
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@app.post("/participateFL")
def listen_and_participate(client_id:int):
    client_address = blockchainService.getAddress(client_id)
    model = load_model()
    (x_train, y_train), (x_test, y_test) = load_dataset(client_id)   
    handle_launch_FL_session(model,x_train, y_train, x_test, y_test, client_id, client_address)



@app.get("/getContributions")
def getContributions(client_id):
    client_address = client_address = blockchainService.getAddress(client_id)
    contributions = BlockchainService.getContributions(client_address)
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(contributions)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/")
def testFAST(client_id):
    client_address = blockchainService.getAddress(client_id)
    return("Hello from client add: ", client_address)