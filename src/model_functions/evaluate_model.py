import json
import sys
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

sys.path.insert(0, '../utils')
from src.utils.model_utility_multi import load_model
from src.utils.generators import get_generator


def evaluate_model(model_configuration: str,
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

    # model template serves to save the model even with multi GPU training
    # model, model_template = load_model(model_parameters,
    #                                    dataset_parameters)

    model = tf.keras.models.load_model(model_parameters['weights'])

    # load optimizer with custom learning rate
    if model_parameters['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers. \
            SGD(lr=model_parameters['learning_rate'],
                momentum=0.9,
                nesterov=False)
    elif model_parameters['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            lr=model_parameters['learning_rate'])

    # compile the model
    model.compile(loss=model_parameters['loss'],
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])

    # create the training and validation data
    empty_training_data, validation_data = get_generator(dataset_parameters,
                                                         model_parameters,
                                                         computer_parameters,
                                                         True)

    # evaluate the model
    print("evaluate model")
    evaluation = model.evaluate_generator(validation_data,
                                          workers=12,
                                          verbose=1)
    print("evaluation", evaluation)
    print("evaluation", model.metrics_names)


if __name__ == '__main__':
    """
    run: python -m src.model_functions.evaluate_model -m resnet50v2 -d affectnet -c blue
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

    evaluate_model(model_configuration_name,
                dataset_configuration_name,
                computer_configuration_name)
