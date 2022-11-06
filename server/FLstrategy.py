import os
import numpy as np
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

import flwr as fl
import tensorflow as tf

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
        *,
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=2,
        evaluate_fn=None,
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
            initial_parameters=initial_parameters
        )
        self.contribution={}


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

            if not os.path.exists(f"./Server/fl_sessions/Session-{session}"):
                os.makedirs(f"./Server/fl_sessions/Session-{session}")
                if  server_round < num_rounds:
                    np.save(f"./Server/fl_sessions/Session-{session}/round-{server_round}-weights.npy", aggregated_weights)
                elif server_round==num_rounds:
                    np.save(f"./Server/fl_sessions/Session-{session}/global_session_model.npy", aggregated_weights)
            else:
                if  server_round < num_rounds:
                    np.save(f"./Server/fl_sessions/Session-{session}/round-{server_round}-weights.npy", aggregated_weights)
                elif server_round==num_rounds:
                    np.save(f"./Server/fl_sessions/Session-{session}/global_session_model.npy", aggregated_weights)


        # loop through the results and update contribution (pairs of key, value) where
        # the key is the client id and the value is a dict of data size, sent size
        # and num_rounds_participated: updated value
        for res in results:
            # results: List[Tuple[ClientProxy, FitRes]]
            # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
            print("data size = ", res[1].num_examples)
            print ("client id = ",res[1].metrics["client_id"])
            
            if res[1].metrics['client_id'] not in self.contribution.keys():
                self.contribution[res[1].metrics["client_id"]]={"data_size":res[1].num_examples ,"num_rounds_participated":1}
            else:
                self.contribution[res[1].metrics["client_id"]]["num_rounds_participated"]+=1
            
        return aggregated_weights



        #     if not (os.path.exists(f'./Server/Global-weights')):
        #         os.mkdir(f"./Server/Global-weights")
        #     filename = f"./Server/Global-weights/round-{server_round}-weights.npy"
        #     # Save weights
        #     print(f"Saving round {server_round} weights...")
        #     np.save(filename, aggregate_weights)
        #     with open(filename,"rb") as f:
        #         bytes = f.read() # read entire file as bytes
        #         readable_hash = hashlib.sha256(bytes).hexdigest() # hash the file
        #         print(readable_hash)
        # return aggregate_weights


def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        # Use the last 5k training examples as a validation set
        x_val, y_val = x_train[45000:50000], y_train[45000:50000]

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_val, y_val)
            return loss, {"accuracy": accuracy}

        return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    with open('config_training.json', 'r') as config_training:
        config=config_training.read()
        data = json.loads(config)
        session=data['session']

    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
        "round": server_round,
        "session": session,
    }
    return config


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

    