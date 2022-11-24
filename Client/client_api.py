from client import CifarClient
from blockchain_service import *
import verify

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import flwr as fl
import json
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np



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
        y_train[client_id * 10000 : (client_id + 1) * 10000].reshape(-1),
    ), (
        x_test[client_id * 1000 : (client_id + 1) * 1000],
        y_test[client_id * 1000 : (client_id + 1) * 1000].reshape(-1),
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
    # model = tf.keras.applications.EfficientNetB0(
    #     # Cifar10 is a dataset of 32x32 RGB color training images, labeled over 10 categories
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,kernel_initializer='he_normal', activation='relu',input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64,3,kernel_initializer='he_normal', activation='relu',strides=1),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,3,kernel_initializer='he_normal', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D((4,4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,kernel_initializer='he_normal',activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10,kernel_initializer='glorot_uniform', activation='softmax')
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@app.post("/participateFL")
def listen_and_participate(client_id:int):
    client_address = blockchainService.getAddress(client_id)
    model = load_model()
    (x_train, y_train), (x_test, y_test) = load_dataset(client_id)   
    handle_launch_FL_session(model,x_train, y_train, x_test, y_test, client_id, client_address)



@app.get("/getContributions")
def getContributions(client_id:int):
    client_address = client_address = blockchainService.getAddress(client_id)
    contributions = BlockchainService.getContributions(client_address)
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(contributions)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/")
def testFAST(client_id:int):
    client_address = blockchainService.getAddress(client_id)
    return("Hello from client add: ", client_address)

@app.get("/getResult")
def getResult():
    model = load_model()
    last_weights = verify.load_last_global_model_weights(model,f'./Global-weights')
    model.set_weights(last_weights)
    (_,_),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.reshape(-1)
    predictions = model.predict(x_test)


    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        verify.plot_image(i, predictions[i], y_test, x_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        verify.plot_value_array(i, predictions[i], y_test)
        plt.tight_layout()
    plt.show()


    return()

    