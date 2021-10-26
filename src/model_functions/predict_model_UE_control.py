import json
import sys
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2

from src.utils.model_utility import *
from src.utils.generators import *


def predict_model(model_configuration: str,
                dataset_configuration: str,
                computer_configuration: str):
    # loads name, image width/ height and l2_reg data
    # model_parameters = load_model_parameters(model)
    with open('src/configuration/model/{}.json'
                      .format(model_configuration)) as json_file:
        model_parameters = json.load(json_file)

    # loads data, number of gpus
    with open('src/configuration/computer/{}.json'
                      .format(computer_configuration)) as json_file:
        computer_parameters = json.load(json_file)

    # loads n_classes, labels, class names, etc.
    # dataset_parameters = load_dataset_parameters(dataset_name, path)
    with open('src/configuration/dataset/{}.json'
                      .format(dataset_configuration)) as json_file:
        dataset_parameters = json.load(json_file)

    model = tf.keras.models.load_model(model_parameters['weights'])

    # get basic shape images
    data_path = computer_parameters["dataset_path"] + "UE4_control"
    file_list = sorted(os.listdir(data_path))
    print("file_list", file_list)

    data = np.zeros((len(file_list), 224, 224, 3))
    for f, file in enumerate(file_list):
        im = cv2.imread(os.path.join(data_path, file))
        print("shape im", np.shape(im))
        im = im[:, 80:560, :]  # crop image
        print("shape im", np.shape(im))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))

        data[f] = im
    print("shape data", np.shape(data))
    print("min max data", np.amin(data), np.amax(data))
    print("first pixel", data[0, 0, 0, :])
    data = resnet_v2.preprocess_input(data)
    print("min max data", np.amin(data), np.amax(data))
    print("prerpocess", data[0, 0, 0, :])

    # predict images
    predictions = model.predict(data, workers=1, verbose=1)

    print("predictions", np.shape(predictions))
    print(np.argmax(predictions, axis=1))
    for i in range(len(file_list)):
        arg = np.argmax(predictions[i])
        print("prediction", file_list[i], dataset_parameters['class_names'][arg])
        print(predictions[i])


if __name__ == '__main__':
    """
    Run the model to predict images set in a folder
    
    run: python -m src.model_functions.predict_model_basic_shape -m resnet50v2 -d affectnet -c blue
    run: python -m src.model_functions.predict_model_UE_control -m resnet50v2 -d UE4_control -c michael_win
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="select your model")
    parser.add_argument("-d", "--dataset",
                        help="select your dataset")
    parser.add_argument("-c", "--computer",
                        help="select your dataset")

    args = parser.parse_args()
    model_configuration_name = args.model
    dataset_configuration_name = args.dataset
    computer_configuration_name = args.computer

    predict_model(model_configuration_name,
                dataset_configuration_name,
                computer_configuration_name)
