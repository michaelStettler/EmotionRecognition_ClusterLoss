import os

folder_path = '../../../../Downloads/MonkeyHeadEmotion/'  # computer a
dataset = 'test_svm/'

num_category = len(os.listdir(folder_path + dataset + 'labelled/'))
print("num_category founded:", num_category)

# create the train, validation and test folder if needed
if not os.path.isdir(folder_path+dataset+'train'):
    os.mkdir(folder_path+dataset+'train')
# if not os.path.isdir(folder_path+dataset+'validation'):
#     os.mkdir(folder_path+dataset+'validation')
if not os.path.isdir(folder_path+dataset+'test'):
    os.mkdir(folder_path+dataset+'test')

for i in range(num_category):
    # create the category folders if they do not exists
    if not os.path.isdir(folder_path+dataset+'train/'+str(i)):
        os.mkdir(folder_path+dataset+'train/'+str(i))
    # if not os.path.isdir(folder_path+dataset+'validation/'+str(i)):
    #     os.mkdir(folder_path+dataset+'validation/'+str(i))
    if not os.path.isdir(folder_path + dataset + 'test/' + str(i)):
        os.mkdir(folder_path + dataset + 'test/' + str(i))

    print("category", i)
    for img in os.listdir(folder_path + dataset + 'labelled/' + str(i)):
        # control to avoid any issues with hidden file or other stuff
        if '.jpeg' in img:
            # get the image number out of the file name
            img_numb = int(img[9:13])
            img_path = folder_path + dataset + 'labelled/' + str(i) + '/' + img
            train_folder = folder_path + dataset + 'train/' + str(i) + '/' + img
            test_folder = folder_path + dataset + 'test/' + str(i) + '/' + img
            # print("img", img_numb, img)

            # sort the picture depending if they are odd or even
            if img_numb % 2 == 0:
                # print("even")
                os.rename(img_path, train_folder)
            else:
                # print("odd")
                os.rename(img_path, test_folder)
