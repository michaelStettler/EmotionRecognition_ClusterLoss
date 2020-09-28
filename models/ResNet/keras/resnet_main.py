from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

resnet50 = ResNet50(weights='imagenet')
print(resnet50.summary())

# img_path = '../../../data/processed/Monkey/MonkeyHead.jpg'
# img_path = 'output_filter_373.png'
img_path = '../../../data/processed/Maya/Face/face.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# Predict
preds = resnet50.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])


# Extract features
# model = Model(inputs=resnet50.input, outputs=resnet50.get_layer('res5c_branch2a').output)
# res5c_branch2a_features = model.predict(x)
# print(np.shape(res5c_branch2a_features))
# print(np.shape(res5c_branch2a_features[0, :, :, 0]))
# print(res5c_branch2a_features[0, :, :, 0])
