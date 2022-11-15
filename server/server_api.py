import os
import json
import time
import numpy as np

import flwr as fl
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from FLstrategy import *
from blockchain_service import BlockchainService

server=FastAPI()
blockchainService = BlockchainService()


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


@server.post("/launchFL")
def launch_fl_session(num_rounds:int, is_resume:bool):
    """Start server and trigger update_strategy then connect to clients to perform fl session"""
    session = int(time.time())
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
                sessions.append(name)
               

    if (is_resume and len(sessions)!=0):
        # test if we will start training from the last session weights and
        # if we have at least a session directory
        if os.path.exists(f'../Server/fl_sessions/{sessions[-1]}/global_session_model.npy'):
            # if the latest session directory contains the global model parameters
            initial_parameters = np.load(f"../Server/fl_sessions/{sessions[-1]}/global_session_model.npy", allow_pickle=True)
            # load latest session's global model parameters
            initial_params = initial_parameters[0]

    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters = initial_params,
    )

    strat_added_BC = blockchainService.addStrategy(session,'FedAvg',num_rounds,strategy.__getattribute__('min_available_clients'))
    print(strat_added_BC)

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    for client in strategy.contribution.keys():
        blockchainService.addContribution(
            strategy.contribution[client]['num_rounds_participated'],
            strategy.contribution[client]['data_size'],
            strategy.contribution[client]['client_address']
        )


@server.get('/')
def testFAST():
    return("Hello from server!")