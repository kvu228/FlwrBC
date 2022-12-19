import os
import json
import time
import numpy as np

import flwr as fl
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt

from FLstrategy import *
from blockchain_service import BlockchainService

server=FastAPI()
blockchainService = BlockchainService()


def load_model():
    with open('model_architecture.json','r') as file:
        json_data = file.read()
    model_architecture = json.loads(json_data)
    model = tf.keras.models.model_from_json(model_architecture)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@server.get('/getContributions')
def getContributions():
    contributions = blockchainService.getContributions()
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(contributions)
    return JSONResponse(content=json_compatible_item_data)


@server.get('/getTrainingSessions')
def getTrainingSessions():
    trainingSessions = blockchainService.getTrainingSessions()
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(trainingSessions)
    return JSONResponse(content=json_compatible_item_data)

@server.get('/getModel')
def getModel():
    with open('config_training.json', 'r') as config_training:
        config=config_training.read()
        data = json.loads(config)
        session = data['session']
        num_round = data['num_rounds']
    model = blockchainService.getModel(session,num_round)
    # Conver Python list to JSON
    json_compatible_item_data = jsonable_encoder(model)
    return JSONResponse(content=json_compatible_item_data)

@server.post("/launchFL")
def launch_fl_session(num_rounds:int, is_resume:bool, budget: float):
    """Start server and trigger update_strategy then connect to clients to perform fl session"""
    session = int(time.time())
    model = load_model()
    with open('config_training.json', 'w+') as config_training:
        config=config_training.read()        
        try :
            data = json.loads(config)
            data['num_rounds']=num_rounds
            data['is_resume']=is_resume
            data['session']= session
            json.dump(data,config_training)

        except json.JSONDecodeError:
            data={}
            data['num_rounds']=num_rounds
            data['is_resume']=is_resume
            data['session']= session
            json.dump(data,config_training)
    
    # Load last session parameters if they exist
    if not (os.path.exists('../Server/fl_sessions')):
    # create fl_sessions directory if first time
        os.mkdir('../Server/fl_sessions')

    # initialise sessions list and initial parameters
    sessions = []
    initial_params = None

    # loop through fl_sessions sub-folders and get the list of directories containing the weights 
    for root, dirs, files in os.walk("../Server/fl_sessions", topdown = False):
        for name in dirs:
            if name.find('Session')!=-1:
                hist_session = name.strip('Session-')
                sessions.append(hist_session)
               

    if (is_resume and len(sessions)!=0):
        # test if we will start training from the last session weights and
        # if we have at least a session directory
        if os.path.exists(f'../Server/fl_sessions/Session-{sessions[-1]}/global_session_{sessions[-1]}_model.npy'):
            # if the latest session directory contains the global model parameters
            initial_parameters = np.load(f"../Server/fl_sessions/Session-{sessions[-1]}/global_session_{sessions[-1]}_model.npy", allow_pickle=True)
            # load latest session's global model parameters
            initial_params = initial_parameters[0]
            # model.set_weights(initial_params)

    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters = initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Add strategy to the blockchain
    strat_added_BC = blockchainService.addStrategy(session,'FedAvg',num_rounds,strategy.__getattribute__('min_available_clients'))

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    for client in strategy.contribution.keys():
        if client != 'total_data_size':
            blockchainService.addContribution(
                _rNo = strategy.contribution[client]['num_rounds_participated'],
                _dataSize= strategy.contribution[client]['data_size'],
                _client_address = strategy.contribution[client]['client_address'],
                _totalDataSize = strategy.contribution['total_data_size'],
                _totalBudget = budget,
                number_of_rounds= num_rounds
            )

    with open(f'../report/temp/FL_result.json','w') as result:
        json.dump(strategy.result, result)
 
@server.get('/')
def testFAST():
    return("Hello from server!")
