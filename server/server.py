from http import server
import time
import json

from FLstrategy import *

def launch_fl_session(num_rounds: int, resume: bool):
    """Start server and trigger update_strategy then connect to clients to perform fl session"""
    session = int(time.time())
    with open('config_training.json', 'w+') as config_training:
        config=config_training.read()        
        try :
            data = json.loads(config)
            data['num_rounds']=num_rounds
            data['resume']=resume
            data['session']= session
            json.dump(data,config_training)

        except json.JSONDecodeError:
            data={}
            data['num_rounds']=num_rounds
            data['resume']=resume
            data['session']= session
            json.dump(data,config_training)
    
    # Load last session parameters if they exist
    if not (os.path.exists('./Server/fl_sessions')):
    # create fl_sessions directory if first time
        os.mkdir('./Server/fl_sessions')

    # initialise sessions list and initial parameters
    sessions = []
    initial_params = None

    # loop through fl_sessions sub-folders and get the list of directories containing the weights 
    for root, dirs, files in os.walk("./Server/fl_sessions", topdown = False):
        for name in dirs:
            if name.find('Session')!=-1:
                sessions.append(name)
               

    if (resume and len(sessions)!=0):
        # test if we will start training from the last session weights and
        # if we have at least a session directory
        if os.path.exists(f'./Server/fl_sessions/{sessions[-1]}/global_session_model.npy'):
            # if the latest session directory contains the global model parameters
            initial_parameters = np.load(f"./Server/fl_sessions/{sessions[-1]}/global_session_model.npy", allow_pickle=True)
            # load latest session's global model parameters
            initial_params = initial_parameters[0]

        # Create strategy
        strategy = SaveModelStrategy(
            fraction_fit=0.3,
            fraction_evaluate=0.2,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            on_fit_config_fn=get_on_fit_config_fn(),
            on_evaluate_config_fn=evaluate_config,
            initial_parameters = initial_params,
        )

        # Start Flower server
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    else:
        model = tf.keras.applications.EfficientNetB0(
            input_shape=(32, 32, 3), weights=None, classes=10
        )
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        # Create strategy
        strategy = SaveModelStrategy(
            fraction_fit=0.3,
            fraction_evaluate=0.2,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            on_fit_config_fn=get_on_fit_config_fn(),
            evaluate_fn=get_evaluate_fn(model),
            on_evaluate_config_fn=evaluate_config,
            initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights()),
        )

        # Start Flower server of federated learning
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

def main() -> None:

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    launch_fl_session(2, False)
    

if __name__ == "__main__":
    main()