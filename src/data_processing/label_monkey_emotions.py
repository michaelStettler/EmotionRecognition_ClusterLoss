import numpy as np
import sys, os
import shutil
import matplotlib.pyplot as plt

sys.path.insert(0, '../utils/')

np.set_printoptions(precision=3, linewidth=200, suppress=True)


# def sort_per_folder(img_dir, output_folder, dist_norm, boundaries):
def sort_per_folder(emotion, joint_dist, labels, label_counters, maximum, boundaries, input_file, output_dir):
    print("folder labels", labels)

    # create the label folders
    train_dir = output_dir + 'train/'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    val_dir = output_dir + 'validation/'
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    for label in labels:
        if not os.path.exists(train_dir + str(label)):
            os.mkdir(train_dir + str(label))
        if not os.path.exists(val_dir + str(label)):
            os.mkdir(val_dir + str(label))

    # loop over the img and sort them
    # for i, img in enumerate(sorted(os.listdir(input_file))):
    for i, dist in enumerate(joint_dist):
        out_dir = output_dir
        train = False

        # select if it's train or validation
        if i < maximum + 1:  # +1 to ensure to take the maximum
            # if it's smaller than the max we can use it for train
            if i % 2 == 0:
                # use only even numbers for training
                out_dir += 'train/'
                train = True
            else:
                out_dir += 'validation/'

        else:
            # sort the img in the validation directory
            out_dir += 'validation/'

        img_name = 'Sequence.' + str('{0:04d}'.format(i)) + '.jpeg'
        img_path = input_file + img_name
        if os.path.exists(img_path):
            label = 0
            # select the label folder
            is_sorted = False
            for b, boundary in enumerate(boundaries):
                if not is_sorted and dist < boundary:
                    # out_dir
                    directory = out_dir + '/' + str(labels[b]) + '/'
                    # count the number of images per category
                    label = labels[b]
                    is_sorted = True

            #  if it's not sorted yet it means it's the last label
            if not is_sorted:
                directory = out_dir + '/' + str(labels[-1]) + '/'
                label = labels[-1]

            if train:
                label_counters[0, label] += 1
            else:
                label_counters[1, label] += 1

            # get the img name and copy past is to the new folder
            shutil.copyfile(img_path, directory + str(emotion) + '_' + img_name)
        else:
            print(img_path, "does not exists!")

    print("label_counters")
    print(label_counters)


def label_monkey_emotions(joint_dist_files, input_files, output_dir, plot=False):
    print("file name", joint_dist_files)

    joint_dists = []
    for file in joint_dist_files:
        joint_dists.append(np.load(file))
    print("joint_dists", np.shape(joint_dists))

    boundaries = [.4, .6, .8, .9]

    maximums = np.argmax(joint_dists, axis=1)
    print(maximums)

    labels = np.arange(int(np.shape(joint_dists)[0] * len(boundaries) + 1))
    label_counters = np.zeros((2, np.shape(labels)[0]))  # 0: train counters, 1: val counters
    print("labels", labels)
    # for d, joint_dist in enumerate(joint_dists):
    #     # get the labels per condition
    #     start = int(d * len(boundaries) + 1)
    #     end = start + len(boundaries)
    #     emotion_labels = np.concatenate(([0], labels[start:end]))
    #     # sort the images to their respective folders
    #     sort_per_folder(d,
    #                     joint_dist,
    #                     emotion_labels,
    #                     label_counters,
    #                     maximums[d],
    #                     boundaries,
    #                     input_files[d],
    #                     output_dir)

    if plot:
        for d in joint_dists:
            plt.figure()
            plt.plot(d)
            plt.plot(np.ones(np.shape(d)[0]) * boundaries[0], 'g')
            plt.plot(np.ones(np.shape(d)[0]) * boundaries[1], 'g')
            plt.plot(np.ones(np.shape(d)[0]) * boundaries[2], 'g')
            plt.plot(np.ones(np.shape(d)[0]) * boundaries[3], 'g')

        plt.show()


if __name__ == '__main__':
    data_path = '../../data_processing/processed/MayaAnimation/'
    # input_path = '../../../../../../michael/A634A95D34A930EB/Users/Michael.DESKTOP-11J049G/Documents/maya/projects/MonkeyHead_MayaProject/images/Fur_FullDetail/'
    input_path = '../../../maya/MonkeyHead_MayaProject/images/Fur_FullDetail/'
    print("list input path:", os.listdir(input_path))
    output_path = '../../../../Downloads/MonkeyHeadEmotion/'
    dataset_name = 'monkey_emotions/'
    output_dir = output_path + dataset_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # files = os.listdir(data_path + "*all_joints_norm*")
    joint_dist_files = [data_path + 'Attention_all_joints_norm.npy',
                        data_path + 'Fear_all_joints_norm.npy', #
                        data_path + 'Angry_all_joints_norm.npy',]
    input_files = [input_path + 'Attention/1.0/StereoCameraLeft/',
                   input_path + 'FearGrin/1.0/StereoCameraLeft/']
    label_monkey_emotions(joint_dist_files, input_files, output_dir, plot=True)
