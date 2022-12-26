import os
import numpy as np
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
import tensorflow as tf
from blockchain_service import *


from pinatapy import PinataPy

with open('../api_key.json', 'r') as api:
    keys=api.read()
    data = json.loads(keys)
    api_key=data['api_key']
    secret_key=data['secret_key']
pinata = PinataPy(api_key, secret_key)

blockchainService = BlockchainService()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
        *,
        fraction_fit=0.3, # Sample 30% of available clients for training
        fraction_evaluate=0.2, # Sample 20% of available clients for evaluation
        min_fit_clients=1, # Never sample less than 1 clients for training
        min_evaluate_clients=1, # Never sample less than 1 clients for evaluation
        min_available_clients=1, # Wait until all 1 clients are available
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters:fl.common.Parameters = None
        ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.contribution={
            'total_data_size': 0
        }
        self.result={
            'aggregated_loss':{
                0:0
            },
            'aggregated_accuracy':{
                0:0
            }
        }


    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> fl.common.Parameters:
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            # get num_rounds from config_training json file to be use to verify
            # if the current round is the first round
            with open('config_training.json', 'r') as config_training:
                config=config_training.read()
                data = json.loads(config)
                num_rounds=data['num_rounds']
                session=data['session']

            if not os.path.exists(f"../Server/fl_sessions/Session-{session}"):
                os.makedirs(f"../Server/fl_sessions/Session-{session}")
                
            if  server_round < num_rounds:
                np.save(f"../Server/fl_sessions/Session-{session}/round-{server_round}-weights.npy", aggregated_weights)
            elif server_round==num_rounds:
                np.save(f"../Server/fl_sessions/Session-{session}/global_session_{session}_model.npy", aggregated_weights)
                file_path = f'../Server/fl_sessions/Session-{session}/global_session_{session}_model.npy'
                with open(file_path,"rb") as f:
                    bytes = f.read() # read entire file as bytes
                    readable_hash = hashlib.sha256(bytes).hexdigest() #hash the file
                    print(readable_hash)
                global_model_BC = blockchainService.addModel(session,server_round,file_path,readable_hash)
                pinata.pin_file_to_ipfs(
                    path_to_file= file_path,
                    ipfs_destination_path = '',
                    save_absolute_paths = False,
                )


        # loop through the results and update contribution (pairs of key, value) where
        # the key is the client id and the value is a dict of data size, sent size
        # and num_rounds_participated: updated value
        # total_data_size = 0
        for res in results:
            # results: List[Tuple[ClientProxy, FitRes]]
            # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
            print("data size = ", res[1].num_examples)
            print("client id = ",res[1].metrics["client_id"])
            print("client address = ",res[1].metrics['client_address'])
            
            if res[1].metrics['client_id'] not in self.contribution.keys():
                self.contribution[res[1].metrics["client_id"]]={
                    "data_size":res[1].num_examples,
                    "num_rounds_participated":1,
                    "client_address":res[1].metrics['client_address']
                }
                self.contribution['total_data_size'] = self.contribution['total_data_size']+res[1].num_examples
            else:
                self.contribution[res[1].metrics["client_id"]]["num_rounds_participated"]+=1
        # if total_data_size !=0:
        #     self.contribution['total_data_size'] = total_data_size
        return aggregated_weights

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        """Aggregate evaluation accuracy using weighted average."""
        if not results:
            return None, {}
        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        self.result['aggregated_loss'][server_round]=aggregated_loss
        self.result['aggregated_accuracy'][server_round]=aggregated_accuracy

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize data
    x_test = x_test/255.

    # # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate



def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, 1 local epochs.
    """

    def fit_config(server_round: int) -> Dict[str, str]:
        with open('config_training.json', 'r') as config_training:
            config=config_training.read()
            data = json.loads(config)
            session=data['session']
        config = {
            "batch_size": 32,
            "local_epochs": 2,
            "round": server_round,
            "session": session,
        }
        return config
        
    return fit_config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    with open('config_training.json', 'r') as config_training:
        config=config_training.read()
        data = json.loads(config)
        session=data['session']
    return {"val_steps": val_steps, "round": server_round, "session":session}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}