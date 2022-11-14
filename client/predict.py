import numpy as np
import os

def load_last_global_model_weights(model, weights_dir):
    """
    Loads the last received weights in the directory in which the global model weights are saved
    """

    # files list will contain the paths to the npy files in the directory
    files_list=[]

    # check every file under the root directory to have .npy extension
    for root, dirs, files in os.walk(weights_dir, topdown = False):
        for file in files:
            if file.endswith(".npy"):
                files_list.append(os.path.join(root,file))
    # get the latest file
    latest_weights_file = max(files_list, key=os.path.getmtime)
    # load the weights from the file
    weights=np.load(latest_weights_file,allow_pickle=True)
    
    return weights


