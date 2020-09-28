"""
Partition the dataset folder categories into train, validation and test folder
"""
import os
import shutil
import numpy as np


def sort_folder(path, partition, cat, images, shuffle_arr):
    # create the category folder
    if not os.path.isdir(path + partition + cat):
        os.mkdir(path + partition + cat)

    # move the images to their new folder according to the shuffle array
    for i in shuffle_arr:
        source_path = path+cat+'/'+images[i]
        dest_path = path+partition+cat+'/'+images[i]
        os.rename(source_path, dest_path)


def create_data_partition(folder_path, train_ratio, val_ratio, verbose=True):
    # get all categories folder within the dataset
    categories_names = os.listdir(folder_path)

    # create the train, validation and test folder if needed
    if not os.path.isdir(folder_path+'train'):
        os.mkdir(folder_path+'train')
    if not os.path.isdir(folder_path+'validation'):
        os.mkdir(folder_path+'validation')
    if not os.path.isdir(folder_path+'test'):
        os.mkdir(folder_path+'test')

    # for each categories
    for cat in categories_names:
        # get all images within the folder
        images_name = os.listdir(folder_path+cat)
        # get the number of images
        num_img = np.size(images_name)
        # create an array for sorting the images
        positions = np.arange(num_img)
        np.random.shuffle(positions)

        # get the idx of the position for each partition
        train_idx = int(num_img*train_ratio)
        val_idx = train_idx + int(num_img*val_ratio)

        # partition the positions
        train_pos = positions[:train_idx]
        val_pos = positions[train_idx:val_idx]
        test_pos = positions[val_idx:]

        # sort the files in the new partition according to the shuffled positions
        sort_folder(folder_path, 'train/', cat, images_name, train_pos)
        sort_folder(folder_path, 'validation/', cat, images_name, val_pos)
        sort_folder(folder_path, 'test/', cat, images_name, test_pos)

        # delete old categorical folders
        shutil.rmtree(folder_path+cat)

        if verbose:
            print('Summary for category:', cat)
            print('Num train images:', np.shape(train_pos))
            print('Num val images:', np.shape(val_pos))
            print('Num test images:', np.shape(test_pos))
            print()


if __name__ == '__main__':
    # folder_path = '../../data/processed/Monkey/Monkey1/'
    # folder_path = '../../../../Downloads/Test/'
    folder_path = '../../../../Downloads/Monkey/Monkey_2/'
    # print(os.listdir(folder_path))
    train_ratio = 0.8
    val_ratio = 0.2
    # test_ratio = 1 - train_ratio - val_ratio

    create_data_partition(folder_path, train_ratio, val_ratio)
