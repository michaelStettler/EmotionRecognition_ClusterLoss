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
    # data_path = computer_parameters["dataset_path"] + "basic_shape_test"
    # data_path = "/app/Dataset/basic_shape"  # test on full BFS data
    data_path = "/app/Dataset/basic_shape_validation"
    file_list = sorted(os.listdir(data_path))
    print("file_list", file_list)

    data = np.zeros((len(file_list), 224, 224, 3))
    for f, file in enumerate(file_list):
        im = cv2.imread(os.path.join(data_path, file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))

        data[f] = im
    print("shape data", np.shape(data))
    print("min max data", np.amin(data), np.amax(data))
    if 'vgg' in model_parameters['model_name']:
        print("VGG pre_process")
        data = vgg16.preprocess_input(data)
    else:
        print("ResNet pre_process")
        data = resnet_v2.preprocess_input(data)
    print("min max data", np.amin(data), np.amax(data))

    # predict images
    predictions = model.predict(data, workers=1, verbose=1)

    print("predictions", np.shape(predictions))
    print(np.argmax(predictions, axis=1))
    n_correct = 0
    hum_correct = 0
    monk_correct = 0
    mery_correct = 0
    num_human = 0
    num_monk = 0
    num_mery = 0
    for i in range(len(file_list)):
        arg = np.argmax(predictions[i])
        print("prediction", file_list[i], dataset_parameters['class_names'][arg])
        is_correct = False
        if 'Sad' in file_list[i] and dataset_parameters['class_names'][arg] == 'Sad':
            is_correct = True
        elif 'Angry' in file_list[i] and dataset_parameters['class_names'][arg] == 'Anger':
            is_correct = True
        elif 'Disgust' in file_list[i] and dataset_parameters['class_names'][arg] == 'Disgust':
            is_correct = True
        elif 'Fear' in file_list[i] and dataset_parameters['class_names'][arg] == 'Fear':
            is_correct = True
        elif 'Happy' in file_list[i] and dataset_parameters['class_names'][arg] == 'Happy':
            is_correct = True
        elif 'Neutral' in file_list[i] and dataset_parameters['class_names'][arg] == 'Neutral':
            is_correct = True
        elif 'Surprise' in file_list[i] and dataset_parameters['class_names'][arg] == 'Surprise':
            is_correct = True

        if is_correct:
            n_correct += 1

            if 'louise' in file_list[i]:
                hum_correct += 1
            elif 'Mery' in file_list[i]:
                mery_correct += 1
            elif 'Monkey' in file_list[i]:
                monk_correct += 1

        if 'louise' in file_list[i]:
            num_human += 1
        elif 'Mery' in file_list[i]:
            num_monk += 1
        elif 'Monkey' in file_list[i]:
            num_mery += 1

    print("len fil list", len(file_list), num_human, num_monk, num_mery)
    print('Total accuracy:', n_correct/len(file_list), n_correct)
    print('Total human accuracy:', hum_correct/num_human, hum_correct)
    print('Total monkey accuracy:', monk_correct/num_monk, monk_correct)
    print('Total mery accuracy:', mery_correct/num_mery, mery_correct)


if __name__ == '__main__':
    """
    Run the model to predict images set in a folder
    
    run: python -m src.model_functions.predict_model_basic_shape -m resnet50v2 -d affectnet -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m resnet50v2 -d basic_shape_resnet -c blue
    
    run: python -m src.model_functions.predict_model_basic_shape -m vgg19_m0006 -d affectnet_vgg_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m vgg19_m0006_bfs -d affectnet_vgg_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m resnet50v2 -d affectnet_resnetv2_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m resnet50v2_bfs -d affectnet_resnetv2_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m CORnet_S_m0003 -d affectnet_resnetv2_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape -m CORnet_S_m0003_bfs -d affectnet_resnetv2_bfs -c blue

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
