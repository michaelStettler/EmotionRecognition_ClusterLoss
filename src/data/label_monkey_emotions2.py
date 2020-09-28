import numpy as np
import sys, os
import shutil
import matplotlib.pyplot as plt

sys.path.insert(0, '../utils/')

np.set_printoptions(precision=3, linewidth=200, suppress=True)


# def sort_per_folder(img_dir, output_folder, dist_norm, boundaries):
def sort_per_folder(jnt_dists, labels, maximum, boundaries, img_path, output_dir, tresh=True):
    print("folder labels", labels)
    img_names = os.listdir(img_path)

    # create the label folders
    if tresh:
        train_dir = output_dir + 'train_tresh/'
        print("train dir", train_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        val_dir = output_dir + 'validation_tresh/'
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
    else:  # timing sorting
        train_dir = output_dir + 'train_time/'
        print("train dir", train_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        val_dir = output_dir + 'validation_time/'
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)

    for label in labels:
        if not os.path.exists(train_dir + str(label)):
            os.mkdir(train_dir + str(label))
        if not os.path.exists(val_dir + str(label)):
            os.mkdir(val_dir + str(label))

    # loop over the files
    for i, joint_dist in enumerate(jnt_dists):

        is_sorted = False
        for b, boundary in enumerate(boundaries):
            if not is_sorted and joint_dist <= boundary:
                is_sorted = True
                # select if the image is set to be sorted in the training or in the validation folder
                # if i % 2 == 0 and i <= maximum:
                if i <= maximum:
                    save_dir = train_dir + str(labels[b])
                else:
                    save_dir = val_dir + str(labels[b])

        # get the img name and copy past is to the new folder
        shutil.copyfile(img_path + '/' + img_names[i], save_dir + '/' + img_names[i])


def label_monkey_emotions_treshold(files, img_paths, output_dir, plot=False):
    print("file name", files)

    print("np shape files", np.shape(files))
    data = []
    for file in files:
        loaded_data = np.load(file)
        data.append(loaded_data[:240])
    print("data", np.shape(data))

    boundaries = [.2, .3, .4, .5, .6, .7, .8, .9, 1]

    maximums = np.argmax(data, axis=1)
    print("maximum", maximums)

    labels = np.arange(int(np.shape(data)[0] * len(boundaries)))
    print("labels", labels)
    for d, jnt_dists in enumerate(data):
        # get the labels per condition
        start = int(d * len(boundaries) + 1)
        end = start + len(boundaries) - 1
        emotion_labels = np.concatenate(([0], labels[start:end]))
        # sort the images to their respective folders
        sort_per_folder(jnt_dists,
                        emotion_labels,
                        maximums[d],
                        boundaries,
                        img_paths[d],
                        output_dir)

    if plot:
        for d in data:
            plt.figure()
            plt.plot(d)
            for i in range(np.shape(boundaries)[0]):
                plt.plot(np.ones(np.shape(d)[0]) * boundaries[i], 'g')

        plt.show()


def label_monkey_emotions_timing(files, img_paths, output_dir, plot=False):
    print("file name", files)

    print("np shape files", np.shape(files))
    data = []
    for file in files:
        loaded_data = np.load(file)
        data.append(loaded_data[:240])
    print("data", np.shape(data))

    maximums = np.argmax(data, axis=1)
    print("maximum", maximums)
    print(data[0][108])
    min = 50
    maximums = [108, 95]  # 108 is the first peak of the open mouth treat condition

    # set the number of condition per category
    num_labels_per_cat = 18
    labels = np.arange(int(np.shape(data)[0] * num_labels_per_cat) + 1)
    print("labels", labels)
    for d, jnt_dists in enumerate(data):
        step_size = int((maximums[d] - min) / num_labels_per_cat)
        print("step_size", step_size)

        # get the labels per condition
        start = int(d * num_labels_per_cat + 1)
        end = start + num_labels_per_cat - 1
        print("start-end", start, end)
        boundaries = []
        for i in range(num_labels_per_cat):
            idx = min + i * step_size
            boundaries.append(jnt_dists[idx])

        emotion_labels = np.concatenate(([0], labels[start:end]))
        # sort the images to their respective folders
        sort_per_folder(jnt_dists,
                        emotion_labels,
                        maximums[d],
                        boundaries,
                        img_paths[d],
                        output_dir,
                        tresh=False)

    if plot:
        for d in data:
            plt.figure()
            plt.plot(d)
            for i in range(np.shape(boundaries)[0]):
                plt.plot(np.ones(np.shape(d)[0]) * boundaries[i], 'g')

        plt.show()


if __name__ == '__main__':
    computer = 'm'

    if computer == 'a':
        data_path = '../../data/processed/MayaAnimation/'
        output_path = '../../../../Downloads/MonkeyHeadEmotion/'
        img_path = ''
    elif computer == 'm':
        data_path = '../../../monkey_emotions/metrics/'
        output_path = '../../../'
        img_path = '/Volumes/Samsung_T5/MonkeyPredictions_v2/images/Predictions/'

    # print(os.listdir(img_path))

    dataset_name = 'monkey_emotions/'
    output_dir = output_path + dataset_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # files = os.listdir(data_path + "*all_joints_norm*")
    files = [data_path + 'Angry_v2_all_joints_norm.npy',
             data_path + 'Fear_v2_all_joints_norm.npy']
    img_paths = [img_path + '02_OpenMouthThreat_1.0_120fps/',
                 img_path + '02_FearGrin_1.0_120fps/']
    label_monkey_emotions_treshold(files, img_paths, output_dir, plot=True)
    # label_monkey_emotions_timing(files, img_paths, output_dir, plot=True)
