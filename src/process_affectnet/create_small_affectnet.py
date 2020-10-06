import os
import shutil
import pandas as pd


def create_small_affectnet(path: str,
                           train: bool,
                           name: str,
                           number_of_rows: int):

    if train:
        file_name = 'training_modified.csv'
        directory = 'training'
    else:
        file_name = 'validation_modified.csv'
        directory = 'validation'

    # read the csv and create a new dataframe with a limited number of rows
    dataframe = pd.read_csv(path+file_name, nrows=number_of_rows)
    small_dataframe = pd.DataFrame(dataframe)

    # create new directory for the small affectnet
    directory_small = path + dir + '_' + name
    if not os.path.exists(directory_small):
        os.mkdir(directory_small)

    # copy the limited amount of images from the old into the new directory
    for index, row in enumerate(small_dataframe.iterrows()):
        image_name = small_dataframe.loc[index, 'subDirectory_filePath']
        shutil.copyfile(path + directory + '/' + image_name,
                        directory_small + '/' + image_name)

    # create a new csv file for the limited amount of images
    if train:
        dataframe.to_csv(path + "training_" + name + ".csv", index=False)
    else:
        dataframe.to_csv(path + "validation_" + name + ".csv", index=False)