import csv
import os
import scipy.ndimage
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import numpy as np

YELLOW = [255, 255, 0]
PURPLE = [128, 0, 128]
CHARTREUSE = [127, 255, 0]


def draw_landmarks(img, img_size, labels):
    start_x = img_size[0]
    end_x = start_x + img_size[2]
    start_y = img_size[1]
    end_y = start_y + img_size[3]

    # horizontal lines
    img[start_y:end_y, start_x, :] = YELLOW
    img[start_y:end_y, end_x, :] = YELLOW

    # vertical
    img[start_y, start_x:end_x, :] = YELLOW
    img[end_y, start_x:end_x, :] = YELLOW

    for i in range(len(labels)//2):
        x = int(float(labels[i*2]))
        y = int(float(labels[i*2+1]))
        if 0 <= x <= np.shape(img)[0] and 0 <= y <= np.shape(img)[1]:
            img[y, x, :] = CHARTREUSE

        # draw = ImageDraw.Draw(img)
        # draw.text((y, x), str(i), (255, 255, 255))

    img = Image.fromarray(img, 'RGB')

    draw = ImageDraw.Draw(img)
    for i in range(len(labels)//2):
        x = int(float(labels[i*2]))
        y = int(float(labels[i*2+1]))
        draw.text((x, y), str(i), (127, 255, 0))

    return img


if __name__ == '__main__':
    path = '../../../../Downloads/AffectNet/'
    file_name = 'training_small.csv'
    # file_name = 'output_1.csv'

    # labels = []
    # # print(os.listdir('../../../../Desktop/'))
    # img = np.zeros((1200, 1200, 3), dtype=np.uint8)
    # print(np.shape(img))

    img_num = 8
    # weird/false: 6 8
    df = pd.read_csv(path + file_name)
    img = scipy.ndimage.imread(path + 'Manually_Annotated_Images/' + df.loc[img_num, 'subDirectory_filePath'])
    img_size = [df.loc[img_num, 'face_x'], df.loc[img_num, 'face_y'], df.loc[img_num, 'face_width'], df.loc[img_num, 'face_height']]
    labels = df.loc[img_num, 'facial_landmarks'].split(';')

    # with open(path + file_name, 'r') as csvfile:
    #     labels_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     next(labels_reader)  # discard first row
    #     row2 = next(labels_reader)
    #     print("img name", row2[0])
    #     # img = scipy.ndimage.imread('../../../../Desktop/2f305fd32c585f3c8f0a6c00a7f76d6aa1d8821453b0464572684ef0.jpg')
    #     # img = scipy.ndimage.imread(path + 'Manually_Annotated_Images/' + row2[0])
    #     # print(row2)
    #     # print(row2[6])
    #     # labels.append(row2[6])
    #     # img = scipy.ndimage.imread(path + row1[0])
    #     # print(np.shape(img))
    #
    #     # labels.append(row2[1:5] + row2[5].split(';') + row2[6:])

    img = draw_landmarks(img, img_size, labels)
    img.show()
