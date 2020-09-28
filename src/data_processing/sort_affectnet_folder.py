import os
import csv

# Python 2!!!!

folder_path='../../../../Downloads/AffectNet/'
os.listdir(folder_path)

# create the train, validation and test folder if needed
if not os.path.isdir(folder_path+'train'):
    os.mkdir(folder_path+'train')
if not os.path.isdir(folder_path + 'validation'):
    os.mkdir(folder_path + 'validation')
if not os.path.isdir(folder_path + 'test'):
    os.mkdir(folder_path + 'test')

for i in range(1, 11):
    if not os.path.isdir(folder_path+'train/'+str(i)):
        os.mkdir(folder_path+'train/'+str(i))

with open(folder_path+'training.csv', 'rb') as csvfile:
    # file = csv.reader(csvfile, delimiter=',')
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        print row[0], row[6]

        dest_folder = folder_path + 'train/'+row[6]

        img_path = folder_path+'Manually_Annotated_Images/'+row[0]
        if os.path.isfile(img_path):
            os.rename(img_path, dest_folder)



# while IFS=, read -r col1 col2 col3 col4
# do
#      echo $col1 $col2 $col3 $col4
# done < $folder_path'training.csv'

