import os
from PIL import Image
import numpy as np
from tqdm import tqdm

path = '../../../../Downloads/AffectNet/training/'
list_img = os.listdir(path)
# print(os.listdir(path))
print(len(list_img))
means = []  # R G B

for img_name in tqdm(list_img):
    img = Image.open(path + img_name)
    img.load()
    data = np.asarray(img, dtype='int32')
    data = data/255.
    # print(np.shape(data))

    mean_R = np.mean(data[:, :, 0])
    mean_G = np.mean(data[:, :, 1])
    mean_B = np.mean(data[:, :, 2])

    # print('mean_RGB', [mean_R, mean_G, mean_B])
    means.append([mean_R, mean_G, mean_B])

# print("means", means)
# print("stds", stds)
mean_RGB = np.mean(means, axis=0)
std_RGB = [np.std(mean_R), np.std(mean_G), np.std(mean_B)]
print("mean", mean_RGB)
print("std", std_RGB)
