from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from joblib import load
import sys, os
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

sys.path.insert(0, '../../models/RBF')
from RBF import *


def load_data(file_path):
    # get all files within the folder
    list_files = os.listdir(file_path)

    # count the num of images with the folders
    nb_images = len(list_files)

    # for each image store it within the numpy array
    data = np.zeros((nb_images, 224, 224, 3))

    for i, file_name in enumerate(list_files):
        img = image.load_img(file_path + file_name, target_size=(224, 224))
        data[i] = image.img_to_array(img)

    return data


def predict_DNN_SVM(model_weights_path, data_path, dataset_name, show=False):
    # load the model
    print("------ load model ------")
    model = ResNet50(weights='imagenet')
    # model = VGG16(weights='imagenet')
    # layer_name = 'activation_49'
    layer_name = 'activation_23'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    # pca_params = np.load(model_weights_path + 'pca_params_01.npy')
    # print("pca_params")
    # print(pca_params)
    # pca = PCA()
    # pca.set_params(copy=pca_params['copy'],
    #                n_components=pca_params['n_components'],
    #                svd_solver=pca_params['svd_solver'],
    #                random_state=pca_params['random_state'],
    #                tol=pca_params['tol'],
    #                whiten=pca_params['whiten'],
    #                iterated_power=pca_params['iterated_power'])

    preproc_params = np.load(model_weights_path + 'preproc_params_01.npy')
    pca = load('../../models/RBF/weights/pca_01.joblib')

    rbf_weights = np.load(model_weights_path + 'rbf_weights_01.npy')
    rbf_centers = np.load(model_weights_path + 'rbf_centers_01.npy')
    print("model loaded")
    print()

    # load the data
    print("------ load data ------")
    data = load_data(data_path + dataset_name + '/')
    print("data", np.shape(data))
    print()

    # run the model
    print("--------------  start prediction  -------------")
    # DNN
    test_data = preprocess_input(data)
    activations = intermediate_layer_model.predict(test_data, batch_size=10, verbose=1)
    activations = np.reshape(activations, (activations.shape[0], -1))
    # reduction and standardize
    data_red = activations[:, preproc_params[0]]
    data_stand = (data_red - preproc_params[1]) / preproc_params[2]
    print("data_stand", np.shape(data_stand))
    x_test = pca.transform(data_stand)
    print("pca shape", np.shape(x_test))

    num_category = np.shape(rbf_centers)[0]
    predictions = []
    for l in range(num_category):
        # rbf = RBF(np.shape(x_test)[1], rbf_centers[l], 1, sigma=2.6)
        rbf = RBF(np.shape(x_test)[1], rbf_centers[l], 1, sigma=4)
        rbf.set_weights(rbf_weights[l])
        predict = rbf.predict(x_test)
        # predict = np.round(predict, 0)
        # predict = np.clip(predict, 0, 1)

        predictions.append(predict)

    # print results
    print("predictions", np.shape(predictions))
    np.save('../../models/RBF/predictions/predictions_' + dataset_name, predictions)

    if show:
        print("shape predictions[3, :, 0]", np.shape(predictions[3]))
        plt.plot(predictions[3])
        plt.show()


if __name__ == '__main__':
    print("coucou :-)")
    # dataset_name = 'Fear_1.0'
    dataset_name = ''
    model_weights_path = '../../models/RBF/weights/'
    # data_path = '../../../../Downloads/MonkeyHeadEmotion/FearGrin/1.0/'
    # data_path = '../../../monkey_emotions/validation_tresh/3/'  # computer m
    # data_path = '../../../monkey_emotions/train_tresh/3/'  # computer m
    # data_path = '../../../monkey_emotions/sequences/'  # computer m
    # data_path = '../../../monkey_emotions/sequences/Fear_1.0/'  # computer m
    data_path = '../../../monkey_emotions/sequences/02_FearGrin_1.0_120fps/'  # computer m

    predict_DNN_SVM(model_weights_path, data_path, dataset_name, show=True)
