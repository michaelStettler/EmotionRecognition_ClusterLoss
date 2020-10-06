import shutil
import os
import pandas as pd


def loop_over(path: str,
              old_path: str,
              name: str,
              size_name: str,
              dataframe):

    # columns in the csv file
    image_name = 'subDirectory_filePath'
    label_name = 'expression'

    # loop over the dataframe and copy the images into the label folder
    for index, row in enumerate(dataframe.iterrows()):

        img_name = dataframe.loc[index, image_name]
        label = dataframe.loc[index, label_name]

        new_path = path + 'categorical_' + name + size_name + '/' + str(label)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        shutil.copyfile(old_path + '/' + img_name,
                        new_path + '/' + image_name)


def sort_small_affectnet(name: str, path: str):

    training_csv_name = path + 'training_'+name+'.csv'
    training_directory = path + 'training_'+name
    validation_csv_name = path + 'validation_'+name+'.csv'
    validation_directory = path + 'validation_'+name

    training_dataframe = pd.read_csv(training_csv_name)
    validation_dataframe = pd.read_csv(validation_csv_name)

    loop_over(path, training_directory, 'training_', name, training_dataframe)
    loop_over(path, validation_directory, 'validation_', name,
              validation_dataframe)
