import json
from argparse import ArgumentParser
from tensorflow import keras
import cv2
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '../utils')
from model_utility_multi import *

def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        faces.append(image[y: y + h, x: x + w])

    return faces


def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm


def normalize_faces(image, faces_coord):
    faces = cut_faces(image, faces_coord)
    faces = resize(faces)

    return faces


def webcam_detection(model_configuration: str,
                     dataset_configuration: str):
    cap = cv2.VideoCapture(0)

    # web cam params
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_Of_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear',
             'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'No-Face']

    # loads name, image width/ height and l2_reg data
    with open('../configuration/model/{}.json'
                      .format(model_configuration)) as json_file:
        model_parameters = json.load(json_file)

    # loads n_classes, labels, class names, etc.
    with open('../configuration/dataset/{}.json'
                      .format(dataset_configuration)) as json_file:
        dataset_parameters = json.load(json_file)

    model = keras.models.load_model(model_parameters['weights'])
    print(model.summary())

    # launch web cam
    video_capture = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')

    exit = False
    while not exit:
        _, frame = video_capture.read()

        faces = classifier.detectMultiScale(
            frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE)

        if faces == ():
            pass
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            faces_only = normalize_faces(frame, faces)
            for face in faces_only:
                image = Image.fromarray(face, 'RGB')
                image_array = np.array(image)
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model(image_array)
                print(prediction)
            predicted_name = names[np.argmax(prediction)]

            # add text on the image
            cv2.putText(frame, predicted_name,
                        bottom_left_corner_Of_text,
                        font,
                        font_scale,
                        font_color,
                        line_type)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # to snap a picture
            # out = cv2.imwrite('capture.jpg', frame)
            # break
            exit = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="select your model")
    parser.add_argument("-d", "--dataset",
                        help="select your dataset")

    args = parser.parse_args()
    model_configuration_name = args.model
    dataset_configuration_name = args.dataset

    webcam_detection(model_configuration_name,
                     dataset_configuration_name)
