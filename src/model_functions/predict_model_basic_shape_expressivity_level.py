import json
import os
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
    with open('src/configuration/model/{}.json'.format(model_configuration)) as json_file:
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

    # load and add softmax layer for probabilities
    model = tf.keras.models.load_model(model_parameters['weights'])
    probabilities = tf.keras.layers.Softmax()(model.output)
    model = tf.keras.models.Model(model.input, probabilities)

    # get basic shape expressivity level images
    data_path = "/Users/michaelstettler/PycharmProjects/BVS/data/BFS_expressivity_level"
    file_list = sorted(os.listdir(data_path))
    print("file_list", len(file_list))
    print(file_list)

    data = []
    label = []
    for f, file in enumerate(file_list):
        if '.DS_Store' not in file:
            im = cv2.imread(os.path.join(data_path, file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (224, 224))

            data.append(im)
            label.append(file)
    data = np.array(data).astype(np.float32)
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
    hum_file = []
    hum_pred_value = []
    hum_neutral_values = []
    for i in range(len(label)):
        arg = np.argmax(predictions[i])
        print("prediction", label[i], dataset_parameters['class_names'][arg])
        is_correct = False
        if 'sad' in label[i] and dataset_parameters['class_names'][arg] == 'Sad':
            is_correct = True
        elif 'anger' in label[i] and dataset_parameters['class_names'][arg] == 'Anger':
            is_correct = True
        elif 'disgust' in label[i] and dataset_parameters['class_names'][arg] == 'Disgust':
            is_correct = True
        elif 'fear' in label[i] and dataset_parameters['class_names'][arg] == 'Fear':
            is_correct = True
        elif 'happy' in label[i] and dataset_parameters['class_names'][arg] == 'Happy':
            is_correct = True
        elif 'neutral' in label[i] and dataset_parameters['class_names'][arg] == 'Neutral':
            is_correct = True
        elif 'surprise' in label[i] and dataset_parameters['class_names'][arg] == 'Surprise':
            is_correct = True

        if is_correct:
            n_correct += 1

            if 'louise' in label[i]:
                hum_correct += 1
            elif 'Mery' in label[i]:
                mery_correct += 1
            elif 'Monkey' in label[i]:
                monk_correct += 1

        if 'louise' in label[i]:
            num_human += 1

            # get predictions
            hum_file.append(label[i])
            for exp in range(len(dataset_parameters['class_names'])):
                if dataset_parameters['class_names'][exp].lower() in label[i]:
                    print("true label: {} {} ({})".format(dataset_parameters['class_names'][exp], label[i], exp))
                    hum_pred_value.append(predictions[i, exp])
            if "neutral" in label[i]:
                hum_neutral_values = predictions[i]

        elif 'Mery' in label[i]:
            num_monk += 1
        elif 'Monkey' in label[i]:
            num_mery += 1

    print("len fil list", len(label), num_human, num_monk, num_mery)
    print('Total accuracy:', n_correct/len(label), n_correct)
    print('Total human accuracy:', hum_correct/num_human, hum_correct)
    print('Total monkey accuracy:', monk_correct/num_monk, monk_correct)
    print('Total mery accuracy:', mery_correct/num_mery, mery_correct)
    print()
    print("Human prediction values", np.shape(hum_pred_value))
    print("len(hum_file)", len(hum_file))
    for i in range(len(hum_file)):
        print("{}: {}".format(hum_file[i], hum_pred_value[i]))

    # transform hum_pred to matrix
    hum_predictions = np.zeros((6, 5))

    # set neutral values
    print("hum_neutral_values", hum_neutral_values)
    for i, matched in enumerate([4, 0, 6, 7, 3, 2]):  # matching between AffectNet and BVS
        hum_predictions[i, 0] = hum_neutral_values[matched]

    # set values for each expressions
    for e, expression in enumerate(["happy", "anger", "sad", "surprise", "fear", "disgust"]):
        for l, level in enumerate(["0.25", "0.50", "0.75", "1.00"]):
            for i, label in enumerate(hum_file):
                if expression + '_' + level in label:
                    hum_predictions[e, l + 1] = hum_pred_value[i]
    print("hum_predictions")
    print(hum_predictions)
    np.save("louise_CORnet_pred_expressivity_level", hum_predictions)



if __name__ == '__main__':
    """
    Run the model to predict images set in a folder
    
    run: python -m src.model_functions.predict_model_basic_shape_expressivity_level -m CORnet_S_m0003 -d affectnet_resnetv2_bfs -c blue
    run: python -m src.model_functions.predict_model_basic_shape_expressivity_level -m CORnet_S_m0003 -d affectnet_resnetv2_bfs -c mac

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
