import cv2
import sys
import numpy as np

sys.path.insert(0, '../utils')
from parameters import *

cap = cv2.VideoCapture(0)

# web cam params
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

# models params
model_name = 'resnet50'
version = '6'
dataset = 'affectnet'
computer = 'm'
# weights_path = '../../models/ResNet/keras/weights/resnet18_classification_affectnet_one_batch_da-2_v-1_01.h5'
weights_path = '../../models/ResNet/keras/weights/resnet50_classification_affectnet-sub8_tl--75_w-2-1-5-8-10-15-5-15_da-1_v-6_01.h5'
names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain',
         'No-Face']

# load model
model_params = load_model_params(model_name, version)
computer = load_computer_params(computer, model_params)
data = load_dataset_params(dataset, model_params, computer)
model, model_template = load_model(True, weights_path, model_params, data)
# launch web cam
exit = False
while not exit:
    ret, frame = cap.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # flip img
    img = cv2.flip(img, 1)

    # # expand dimensions to be able to send the img to the network
    x = cv2.resize(img, (model_params['img_width'], model_params['img_width']))
    x = np.expand_dims(x, axis=0)
    # send img through the network
    predictions = model.predict(x)
    # print(str(np.argmax(predictions)))

    # add text on the image
    cv2.putText(img, names[np.argmax(predictions)],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # to snap a picture
        # out = cv2.imwrite('capture.jpg', frame)
        # break
        exit = True


cap.release()
cv2.destroyAllWindows()