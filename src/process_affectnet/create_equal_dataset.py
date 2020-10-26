import os
import shutil
from argparse import ArgumentParser

import pandas as pd


def create_equal_dataset(path: str,
                         number_of_images: int,
                         train: bool):
    if train:
        file_name = 'training_modified_renamed.csv'
        directory = 'training'
    else:
        file_name = 'validation_modified_renamed.csv'
        directory = 'validation'

    # read the csv and create a new dataframe
    dataframe = pd.read_csv(path + file_name)
    new_dataframe = pd.DataFrame()

    # create new directory for the equal affectnet
    directory_new = path + directory + '_equal' + number_of_images
    if not os.path.exists(directory_new):
        os.mkdir(directory_new)

    image_counter = {"Neutral": 0, "Happy": 0, "Sad": 0, "Surprise": 0,
                     "Fear": 0, "Disgust": 0, "Anger": 0, "Contempt": 0,
                     "None": 0, "Uncertain": 0, "Non-Face": 0}

    # copy the limited amount of images from the old into the new directory
    for index, row in enumerate(dataframe.iterrows()):
        counter = 0
        expression = dataframe.loc[index, 'expression']
        print(image_counter[expression])
        if image_counter[expression] <= number_of_images:
            image_name = dataframe.loc[index, 'subDirectory_filePath']
            shutil.copyfile(path + directory + '/' + image_name,
                            directory_new + '/' + image_name)
            image_counter[expression] = image_counter[expression] + 1
            new_dataframe.append(row)

        for x, y in image_counter.items():
            counter = counter + y
        print('** {} **'.format(counter), end="\r", flush=True)

        if counter == number_of_images * 11:
            print('finished')
            break

        elif index % 1000 == 0:
            print('processed rows: {}'.format(index))

    if train:
        dataframe.to_csv(path + "training_equal" + number_of_images +
                         ".csv", index=False)
    else:
        dataframe.to_csv(path + "validation_equal" + number_of_images +
                         ".csv", index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="path to the affectnet directory")
    parser.add_argument("-n", "--number",
                        help="number of images")
    parser.add_argument("-t", "--train", type=bool,
                        help="include when creating a training directory")

    args = parser.parse_args()
    affectnet_path = args.path
    subset_size = args.number
    training_bool = args.train

    create_equal_dataset(affectnet_path, subset_size, training_bool)