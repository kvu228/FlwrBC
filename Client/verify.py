import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import json
import requests

import tensorflow as tf
import flwr as fl
from pinatapy import PinataPy

with open('../api_key.json', 'r') as api:
    keys=api.read()
    data = json.loads(keys)
    api_key=data['api_key']
    secret_key=data['secret_key']
pinata = PinataPy(api_key, secret_key)


def load_last_global_model_weights_from_localDB(weights_directory): 
    """
    loads the last received weights in the directory
    in which the global model weights are saved
    """
    # files list will contain the paths to the npy files in the directory
    files_list=[]
    # check every file under the root directory to have .npy extension
    for root, dirs, files in os.walk(weights_directory, topdown = False):
        for file in files:
            if file.endswith(".npy"):
                files_list.append(os.path.join(root,file))
    # get the latest file
    latest_weights_file = max(files_list, key=os.path.getmtime)
    # load the weights from the file
    weights=np.load(latest_weights_file,allow_pickle=True)
    return weights


def load_last_global_model_weights_from_IPFS():
    list_files = pinata.pin_list()
    URL = 'https://gateway.pinata.cloud/ipfs/'+ list_files['rows'][0]['ipfs_pin_hash']
    r = requests.get(URL,allow_redirects=True)
    if not os.path.exists('./temp'):
        os.makedirs('./temp/')
    open('./temp/lastParameters.npy', 'wb').write(r.content)
    latest_parameters = np.load('./temp/lastParameters.npy',allow_pickle=True)
    latest_weights = fl.common.parameters_to_ndarrays(latest_parameters[0])
    folder = './temp'
    shutil.rmtree(folder)
    return latest_weights


def plot_image(i, predictions_array, true_label, img):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'hourse', 'ship', 'truck']

    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_confussion_matrix(model):
    (_,_),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.reshape(-1)
    x_test = x_test/255.
    predictions = model.predict(x_test)
    model_preds = predictions.argmax(axis=1)

    confusion  = tf.math.confusion_matrix(
        labels=y_test,
        predictions=model_preds,
        num_classes=10    
    )
    conf_matrix = np.array(confusion)
    # print(conf_matrix)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha = 1)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j]/10, va='center', ha='center', size='x-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()        
    plt.show()