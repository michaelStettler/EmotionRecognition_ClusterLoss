"""
This file is probably not required/ outdated.
It moves ALL images into the correct label sub-folders.
"""
import os
import csv
import pandas as pd

def create_folder(path):

    # create the train, validation and test folder if needed
    if not os.path.isdir(path+'training'):
        os.mkdir(path+'training')
    if not os.path.isdir(path+'validation'):
        os.mkdir(path+'validation')
    if not os.path.isdir(path+'testing'):
        os.mkdir(path+'testing')

    # create 11 folder for the 11 classes/ labels in affectnet
    for i in range(1, 11):
    if not os.path.isdir(folder_path+'train/'+str(i)):
        os.mkdir(folder_path+'train/'+str(i))

    # move the labeled images to the correct training sub-folder
    image_name = 'subDirectory_filePath'
    label_name = 'expression'
    training_dataframe = pd.read_csv(path+'training.csv')
    for index, row in enumerate(training_dataframe.iterrows()):
        training_folder = path+'training/'+training_dataframe.loc[index, image_name]
        image_path = path+'Manually_Annotated_Images/'+training_dataframe.loc[index, label_name]
        if os.path.isfile(image_path):
            os.rename(image_path, training_folder)

""" reading the csv without pandas

    with open (path+'training.csv') as file:
        csv_file = csv.reader(file, delimiter=',')
        for row in file:
            training_folder = path+'training/'+row[6]
            image_path = path+'Manually_Annotated_Images/'+row[0]
            if os.path.isfile(img_path):
                os.rename(image_path, training_folder)

"""
