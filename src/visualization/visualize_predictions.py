from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image

sys.path.insert(0, '../utils/')
from utils import*

np.set_printoptions(precision=3, linewidth=200, suppress=True)

# build the network with ImageNet weights
model = ResNet50(weights='imagenet', include_top=True)

# load the images
images = []
raw_img = []
path_folder = import_img_name_from_files("../../data_processing/processed/Maya/Face/face_invariant")
prediction = []
for path_img in sorted(path_folder):
    x = image.load_img(path_img, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    images.append(x)

    # somehow needs to reload the pictures.. strange
    y = image.load_img(path_img, target_size=(224, 224))
    y = image.img_to_array(y)
    raw_img.append(y)

print("np.shape(images)", np.shape(images))
print("np.shape(raw_img)", np.shape(raw_img))

num_row = 2
num_col = np.shape(images)[0]
plt.figure(figsize=(3.5 * num_col, 2 * num_row))
plt.suptitle('Predictions')

# loop over each images
for img_idx in range(num_col):
    # Predict
    preds = model.predict(images[img_idx])
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    decode_pred = decode_predictions(preds, top=3)[0]

    pred_class = [decode_pred[idx][1] for idx in range(np.shape(decode_pred)[0])]
    pred_val = [decode_pred[idx][2] for idx in range(np.shape(decode_pred)[0])]

    plt.subplot(num_row, num_col, img_idx + 1)
    plt.imshow(raw_img[img_idx] / 255)
    plt.subplot(num_row, num_col, (img_idx + 1) + num_col)
    plt.bar([0, 1, 2], pred_val)
    plt.xticks([0, 1, 2], pred_class)
    plt.ylim(ymax=1)
    # plt.title("Image "+str(img_idx))

plt.show()
