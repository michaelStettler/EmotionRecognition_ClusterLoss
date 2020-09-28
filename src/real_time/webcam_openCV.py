from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_absolute_face_position(cap, detector, predictor):
    ret, frame = cap.read()
    image = imutils.resize(frame, width=500)

    # flip img
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('frame', image)

    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    return [x + w / 2, y + h / 2]


def get_relative_face_position(cap, detector, predictor):
    ret, frame = cap.read()

    # normal size (a bit slow..) // todo: create some variable to control the size?
    # height, width = np.shape(frame)[0], np.shape(frame)[1]
    # # flip img
    # image = cv2.flip(frame, 1)

    image = imutils.resize(frame, width=500)
    height, width = np.shape(image)[0], np.shape(image)[1]
    image = cv2.flip(image, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('frame', image)

    if len(rects) > 0:
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        result = [x + w / 2 - width / 2, -(y + h / 2 - height / 2)]
    else:
        (x, y, w, h) = (None, None, None, None)
        result = [x, y]

    return result


def test_webcam():
    cap = cv2.VideoCapture(0)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor

    shape_predictor_path = '../../data/external/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # launch web cam
    exit = False
    while not exit:
        position = get_relative_face_position(cap, detector, predictor)
        print("position", np.shape(position), position)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # to snap a picture
            # out = cv2.imwrite('capture.jpg', frame)
            # break
            exit = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #                 help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to input image")
    # args = vars(ap.parse_args())

    test_webcam()
