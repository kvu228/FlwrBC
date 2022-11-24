import matplotlib.pyplot as plt
import numpy as np
import os


def load_last_global_model_weights(model,weights_directory): 
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