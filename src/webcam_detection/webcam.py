import json
from argparse import ArgumentParser
import tensorflow as tf
import cv2
import sys
import numpy as np
from PIL import Image


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


def webcam_detection(model_path: str):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(
        physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    cap = cv2.VideoCapture(0)

    # web cam params
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_Of_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear',
             'Disgust', 'Anger', 'Contempt']
    names = sorted(names)

    model = tf.keras.models.load_model(model_path)

    # launch web cam
    video_capture = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')

    exit = False
    while not exit:
        _, frame = video_capture.read()

        faces = classifier.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE)

        predicted_name = 'unknown'
        if faces == ():
            pass
        else:
            for (x, y, w, h) in faces:
                x -= 30
                y -= 30
                w += 60
                h += 60
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            faces_only = normalize_faces(frame, faces)
            for face in faces_only:
                image = Image.fromarray(face, 'RGB')
                image_array = np.array(image, dtype=np.float32)
                image_array /= 127.5
                image_array -= 1.
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model(image_array)

        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #x = cv2.resize(img, (224, 224))
        #x = np.array(x, dtype=np.float32)
        #x /= 127.5
        #x -= 1.
        #x = np.expand_dims(x, axis=0)
        #prediction = model(x)

            for i, item in enumerate(prediction[0]):
                print(f'{names[i]}: {float(item)}')
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

    args = parser.parse_args()
    model_configuration_name = args.model

    webcam_detection(model_configuration_name)
