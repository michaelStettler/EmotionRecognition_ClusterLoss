import json
import sys
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

sys.path.insert(0, '../utils')
from model_utility import *
from generators import *


def train_model(model_configuration: str,
                dataset_configuration: str,
                computer_configuration: str):
    # loads name, image width/ height and l2_reg data
    # model_parameters = load_model_parameters(model)
    with open('../configuration/model/{}.json'
                      .format(model_configuration)) as json_file:
        model_parameters = json.load(json_file)

    # loads data, number of gpus
    with open('../configuration/computer/{}.json'
                      .format(computer_configuration)) as json_file:
        computer_parameters = json.load(json_file)

    # loads n_classes, labels, class names, etc.
    # dataset_parameters = load_dataset_parameters(dataset_name, path)
    with open('../configuration/dataset/{}.json'
                      .format(dataset_configuration)) as json_file:
        dataset_parameters = json.load(json_file)

    model = tf.keras.models.load_model(model_parameters['model_path'])

    # create the training and validation data
    empty_training_data, validation_data = get_generator(dataset_parameters,
                                                         model_parameters,
                                                         computer_parameters,
                                                         True)

    predictions = model.predict(validation_data,
                                workers=12,
                                verbose=1)

    print("shape prediction", np.shape(predictions))
    print('** classes: **', validation_data.classes)
    print('** classes indices: **', validation_data.class_indices)
    if 'AffectNet' in dataset_parameters['dataset_name']:
        print(classification_report(validation_data.classes,
                                    predictions.argmax(axis=1),
                                    target_names=dataset_parameters[
                                        'class_names']))
    else:
        print(classification_report(validation_data.classes,
                                    predictions.argmax(axis=1)))

    np.save('../metrics/{}/'.format(dataset_parameters['dataset_name'])
            + 'predictions', predictions)


if __name__ == '__main__':
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

    train_model(model_configuration_name,
                dataset_configuration_name,
                computer_configuration_name)
