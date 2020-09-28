from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
from joblib import dump
import tqdm
import sys, os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '../../models/RBF')
from RBF import *

np.set_printoptions(precision=2, linewidth=200, suppress=True)


def plot_confusion_matrix(y_true, y_prediction):
    conf_mat = confusion_matrix(y_true, y_prediction)
    print(conf_mat)


def get_accuracy(y_true, y_predict):
    count = 0
    print("shape y_true y_predict", np.shape(y_true), np.shape(y_predict))
    for i, y in enumerate(y_true):
        if y == y_predict[i]:
            count += 1
    return count / np.shape(y_true)[0]


def _PCA(data, dims_rescaled_data=2):
    """
    returns: data_processing transformed in 2 dims/columns + regenerated original data_processing
    pass in: data_processing as 2D NumPy array
    https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data_processing
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data_processing array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data_processing using eigenvectors
    # and return the re-scaled data_processing, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs


def load_data_from_folder(file_path, data_type):

    # train
    if data_type == 'tresh':
        train_folder = '/train_tresh'
        val_folder = '/validation_tresh'
    elif data_type == 'time':
        train_folder = '/train_time'
        val_folder = '/validation_time'
    else:
        print("Please specify which type (condition)!")

    list_folder = os.listdir(file_path + train_folder)
    num_category = len(list_folder)
    label = []

    # count the num of images with the folders
    nb_images = 0
    for sub_folder in list_folder:
        # just to remove the .DS_Store file
        dir = os.path.join(file_path + train_folder, sub_folder)
        if os.path.isdir(dir):
            nb_images += len(os.listdir(dir))

    # for each image store it within the numpy array
    data = np.zeros((nb_images, 224, 224, 3))
    i = 0
    idx_label = 0
    for sub_folder in list_folder:
        # just to remove the .DS_Store file
        dir = os.path.join(file_path + train_folder, sub_folder)
        if os.path.isdir(dir):
            sub_files = os.listdir(dir)
            for file_name in sub_files:
                # print("file_name", file_name)
                # print("i", i)
                # print()
                img = image.load_img(dir + '/' + file_name, target_size=(224, 224))
                data[i] = image.img_to_array(img)
                label.append(idx_label)
                i += 1
            idx_label += 1

    # test
    list_folder = os.listdir(file_path + val_folder)
    test_label = []

    # count the num of images with the folders
    nb_images = 0
    for sub_folder in list_folder:
        # just to remove the .DS_Store file
        dir = os.path.join(file_path + val_folder, sub_folder)
        if os.path.isdir(dir):
            nb_images += len(os.listdir(dir))

    # for each image store it within the numpy array
    test_data = np.zeros((nb_images, 224, 224, 3))
    i = 0
    idx_label = 0
    for sub_folder in list_folder:
        # just to remove the .DS_Store file
        dir = os.path.join(file_path + val_folder, sub_folder)
        if os.path.isdir(dir):
            sub_files = os.listdir(dir)
            for file_name in sub_files:
                img = image.load_img(dir + '/' + file_name, target_size=(224, 224))
                test_data[i] = image.img_to_array(img)
                test_label.append(idx_label)
                i += 1
            idx_label += 1

    return data, label, test_data, test_label, num_category


def load_csv_data(file_path):
    df = pd.read_csv(file_path)

    data = np.zeros((df.shape[0], 224, 224, 3))
    for i, line in enumerate(df.iterrows()):
        # dir, img = im_path.split('/')
        img_name = df.loc[i, 'subDirectory_filePath']
        img_path = path + dataset + '/' + img_name

        img = image.load_img(img_path, target_size=(224, 224))
        data[i] = image.img_to_array(img)
    # pca_params = np.load(mo
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

    return data


def run(file_path, csv, data_type, show=False):
    # Load model
    model = ResNet50(weights='imagenet')
    # model = VGG16(weights='imagenet')
    model.summary()

    # Predict
    # layer_name = 'activation_49'
    # layer_name = 'activation_34'
    layer_name = 'activation_23'
    # layer_name = 'activation_11'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    # load the data_processing and put them into a numpy array
    if csv:
        data = load_csv_data(file_path + '.csv')
    else:
        data, label, test_data, test_label, num_category = load_data_from_folder(file_path, data_type)
        y = label
        y_test = test_label

    print("shape data_processing")
    print(data.shape)

    # predict all images
    data = preprocess_input(data)
    activations = intermediate_layer_model.predict(data, batch_size=16, verbose=1)
    print("intermediate output")
    print(np.shape(activations))

    # reshape the output to reduce the matrix dimensionality
    activations = np.reshape(activations, (activations.shape[0], -1))
    print("reshape (flatten)", activations.shape)

    print("---------------- STD -------------------")
    # reduce the dimensionality features of the activations
    # calculate the variance
    std = np.std(activations, axis=0)
    # print("var shape", std.shape)
    print("max value", np.max(std))
    thresh = False
    if thresh:
        # threshold the variance and keep only the interesting values
        threshold = 2
        print(np.shape(std[std > threshold]))
        idx_red = np.reshape(np.argwhere(std > threshold), -1)
        print("idx_thresh", np.shape(idx_red))
        x_red = activations[:, idx_red]
    else:
        # keep the n max values
        n = 500
        idx_red = np.argsort(std)[-n:]
        x_red = activations[:, idx_red]
        print("idx_n_max", np.shape(idx_red))
    print("x_red shape", x_red.shape)

    # print(val_thresh)
    # print()

    # standardize the data_processing
    mean_x_red = np.mean(x_red)
    std_x_red = np.std(x_red)
    x_stand = (x_red - mean_x_red) / std_x_red

    print("mean_x_red, std_x_red", mean_x_red, std_x_red)
    print()
    # PCA
    print("---------------- PCA -------------------")
    # pca, evals, evecs = _PCA(val_thresh, dims_rescaled_data=2)
    # print("PCA", np.shape(pca))
    # print(pca)
    # print()
    # pca = PCA(n_components=np.shape(val_thresh)[0])  # n_components = n_samples != n_features
    # pca = PCA(n_components=8)  # n_components = n_samples != n_features
    pca = PCA(.95)  # means we want the minimum of components to retain 95% of the variability
    pca.fit(x_stand)
    # print(pca.components_)
    # print(pca.explained_variance_)
    print("PCA explained variance")
    print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))
    # print(pca.singular_values_)
    x = pca.transform(x_stand)
    print("PCA x", np.shape(x))
    # print(x)
    print()

    print("---------------- RBF -------------------")
    print("y", np.shape(y))
    y = np.array(y)

    # sigma = 2
    # sigma = 2.6
    sigma = 4  # better with activation 34
    # sigma = .2
    # one against others training
    rbf_weights = []
    rbf_centers = []
    for l in range(num_category):
        # create a new label array -> 1 means the label we want to train on
        new_y_train = np.zeros(np.shape(y)[0])
        new_y_train[y == l] = 1
        print("shape new_y_train", np.shape(new_y_train))
        # print("new_y_train", np.shape(new_y_train))
        # print(new_y_train)

        # train the RBF
        centers = np.zeros((np.shape(x)[1]))
        # print("centers", np.shape(centers))
        rbf = RBF(np.shape(x)[1], centers, 1, sigma=sigma)
        # rbf.set_unsup_centers(x, k=40)
        rbf.set_unsup_centers(x, k=148)
        rbf.fit(x, new_y_train)
        # print("RBF trained")
        np.set_printoptions(precision=0, linewidth=200, suppress=True)
        y_predict = rbf.predict(x)
        # round it as the label are integers
        y_predict = np.round(y_predict, 0)
        # also clip the array as it is a binary task
        y_predict = np.clip(y_predict, 0, 1)

        # print("y_predict", y_predict)
        rbf_weights.append(rbf.get_weights())
        rbf_centers.append(rbf.get_centers())

        print("Class: ", l, " Accuracy:", get_accuracy(new_y_train, y_predict))
        print(plot_confusion_matrix(new_y_train, y_predict))
        print()

    # saving the models for futur use
    np.save('../../models/RBF/weights/rbf_weights_01', rbf_weights)
    np.save('../../models/RBF/weights/rbf_centers_01', rbf_centers)
    dump(pca, '../../models/RBF/weights/pca_01.joblib')
    np.save("../../models/RBF/weights/preproc_params_01", [idx_red, mean_x_red, std_x_red])
    print("model saved")
    print()

    print("--------------  testing  -------------")
    #evaluate on test set
    # DNN
    test_data = preprocess_input(test_data)
    activations = intermediate_layer_model.predict(test_data, batch_size=16, verbose=1)
    activations = np.reshape(activations, (activations.shape[0], -1))
    # reduction and standardize
    val_red = activations[:, idx_red]
    val_stand = (val_red - mean_x_red) / std_x_red
    x_test = pca.transform(val_stand)

    y_test = np.array(y_test)
    predictions = []
    for l in range(num_category):
        new_y_test = np.zeros(np.shape(y_test)[0])
        new_y_test[y_test == l] = 1
        # print(new_y_test.astype(int))

        rbf = RBF(np.shape(x_test)[1], rbf_centers[l], 1, sigma=sigma)
        rbf.set_weights(rbf_weights[l])
        y_predict = rbf.predict(x_test)
        # round it as the label are integers
        y_predict = np.round(y_predict, 0)
        # also clip the array as it is a binary task
        y_predict = np.clip(y_predict, 0, 1)
        # print(np.squeeze(y_predict).astype(int))
        predictions.append(y_predict)

        print("Class: ", l, " Accuracy:", get_accuracy(new_y_test, y_predict))
        print(plot_confusion_matrix(new_y_test, y_predict))
        print()

    np.save('../../models/RBF/predictions/test_predictions', predictions)

    if show:
        print("shape predictions[3, :, 0]", np.shape(predictions[3]))
        plt.plot(predictions[3])
        plt.show()


if __name__ == '__main__':
    # get the data_processing
    # path = '../../../../Downloads/AffectNet/'  # computer a
    # path = '../../../../Downloads/MonkeyHeadEmotion/'  # computer a
    path = '../../../'  # computer m
    # dataset = 'training_one_batch'
    # dataset = 'train_sub5_500'
    # dataset = 'categorical_training_one_batch'
    dataset = 'monkey_emotions'
    csv = False
    data_type = 'tresh'  # choose between time or tresh -> depending on which dataset we want to use

    run(path + dataset, csv, data_type, show=True)
