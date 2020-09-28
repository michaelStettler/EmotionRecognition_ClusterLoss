import os

def load_model_params(model, version, class_weights=None):
    model_dict = {'name': model, 'n_fix_layers': -1}
    # n_fix_layers : needed only for transfer learning

    if 'RESNET' in model.upper():
        model_dict['img_width'] = 224
        model_dict['img_height'] = 224
        model_dict['model_path'] = '../../models/ResNet/keras/'
        model_dict['n_fix_layers'] = -75
    elif 'DENSENET' in model.upper():
        model_dict['img_width'] = 224
        model_dict['img_height'] = 224
        model_dict['model_path'] = '../../models/DenseNet/keras/'
    elif 'SIMPLE' in model.upper():
        model_dict['img_width'] = 224
        model_dict['img_height'] = 224
        model_dict['model_path'] = '../../models/Simple/'
    elif 'VGG' in model.upper():
        model_dict['img_width'] = 224
        model_dict['img_height'] = 224
        model_dict['model_path'] = '../../models/VGG/keras/'

    # simple small exists just to try out cifar-10
    if 'SMALL' in model.upper():
        model_dict['img_width'] = 32
        model_dict['img_height'] = 32

    model_dict['optimizer'] = 'sgd'
    if version == '00':
        model_dict['lr'] = [0.01]
        model_dict['num_epochs'] = [3]
        model_dict['l2_reg'] = 0.00001 / 2
    elif version == '0':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.00001 / 2
    elif version == '1':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.00001 / 2
    elif version == '2':
        model_dict['lr'] = [0.001, 0.0001, 0.00001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.00001 / 2
    elif version == '3':
        model_dict['lr'] = [0.2, 0.1, 0.01]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.00001 / 2
    elif version == '4':
        model_dict['lr'] = [0.1, 0.01, 0.001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.000001
    elif version == '5':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.000001
    elif version == '6':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.00001
    elif version == '7':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.0001
    elif version == '8':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.001
    elif version == '9':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.01
    elif version == '10':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30, 60]
        model_dict['l2_reg'] = 0.1
    elif version == '11':
        model_dict['lr'] = [0.01, 0.001, 0.0001, 0.00001]
        model_dict['num_epochs'] = [5, 5, 5, 5]
        model_dict['l2_reg'] = 0.00001
    elif version == '1a':
        model_dict['lr'] = [0.01]
        model_dict['num_epochs'] = [60]
        model_dict['l2_reg'] = 0.0001/2
    elif version == '1b':
        model_dict['lr'] = [0.1]
        model_dict['num_epochs'] = [60]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '1c':
        model_dict['lr'] = [0.001]
        model_dict['num_epochs'] = [60]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '1d':
        model_dict['lr'] = [0.0001]
        model_dict['num_epochs'] = [60]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '2a':
        model_dict['lr'] = [0.01, 0.001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.0001
    elif version == '2b':
        model_dict['lr'] = [0.01, 0.001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '2c':
        model_dict['lr'] = [0.01, 0.001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.001
    elif version == '2d':
        model_dict['lr'] = [0.001, 0.0001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '2e':
        model_dict['lr'] = [0.001, 0.0001, 0.00001]
        model_dict['num_epochs'] = [30, 30, 30]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '3a':
        model_dict['lr'] = [0.1, 0.01, 0.001]
        model_dict['num_epochs'] = [30, 30, 30]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '3b':
        model_dict['lr'] = [0.01, 0.001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.0001
    elif version == '3c':
        model_dict['lr'] = [0.01, 0.001]
        model_dict['num_epochs'] = [30, 30]
        model_dict['l2_reg'] = 0.001 / 2
    elif version == '4a':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [20, 20, 30]
        model_dict['l2_reg'] = 0.0001 / 2
    elif version == '4b':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [20, 20, 30]
        model_dict['l2_reg'] = 0.0001
    elif version == '4c':
        model_dict['lr'] = [0.01, 0.001, 0.0001]
        model_dict['num_epochs'] = [20, 20, 30]
        model_dict['l2_reg'] = 0.001 / 2
    elif version == '5a':
        model_dict['lr'] = [0.000024]
        model_dict['num_epochs'] = [20]
        model_dict['l2_reg'] = 0.0001 / 2
        model_dict['optimizer'] = 'adam'


    if class_weights == '0':
        model_dict['class_weights'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif class_weights == '1':
        model_dict['class_weights'] = [2.0, 1.0, 5.0, 8.0, 10.0, 15.0, 5.0, 15.0]
    elif class_weights == '2':
        model_dict['class_weights'] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0]
    else:
        model_dict['class_weights'] = class_weights
    # model_dict['batch_size'] = 256
    # model_dict['batch_size'] = 192
    # model_dict['batch_size'] = 128
    model_dict['batch_size'] = 64
    # model_dict['batch_size'] = 5

    return model_dict


def load_computer_params(computer, model):
    computer_dict = {'name': computer}
    # first computer (980ti)
    if computer == 'a':
        computer_dict['data_path'] = '../../../../Downloads/'
        model['batch_size'] = 32  # memory issues
        print("Careful the batch size is changed to %.0f!!!!!!!!!!!!!!" % model['batch_size'])
    # blue computer (2x1080 ti)
    elif computer == 'b':
        computer_dict['data_path'] = '../../../../media/data_processing/'
    # cluster
    elif computer == 'c':
        computer_dict['data_path'] = '../../../../../../beegfs/work/knaxq01/michael/data_processing/'
    # personal MacBook
    elif computer == 'm':
        # computer_dict['data_path'] = '../../../'
        computer_dict['data_path'] = '/Volumes/Samsung_T5/'

    else:
        raise ValueError('Wrong or no computer selected')

    return computer_dict


def load_dataset_params(dataset, model, computer):
    data_dict = {'dataset': dataset}
    # imageNet mean and std
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
    data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    af_val = [75374, 134915, 25959, 14590, 6878, 4303, 25382, 4250, 33588, 12145, 82915]

    # define which dataset we want to use in order to train the model
    if dataset == 'cifar10':
        data_dict['weights_path'] = model['model_path']+'weights/'
        data_dict['metrics_path'] = model['model_path']+'metrics/'
        data_dict['n_classes'] = 10
        data_dict['labels_type'] = 'numpy'
    elif dataset == 'imagenet':
        data_dict['data_path'] = computer['data_path']+'ImageNet/'
        data_dict['weights_path'] = model['model_path']+'weights/'
        data_dict['metrics_path'] = model['model_path']+'metrics/'
        data_dict['n_classes'] = 1000
        data_dict['labels_type'] = 'auto'
        data_dict['train_dir'] = data_dict['data_path'] + 'train'
        data_dict['val_dir'] = data_dict['data_path'] + 'validation'
    elif dataset == 'test':
        data_dict['data_path'] = computer['data_path'] + 'Test/'
        data_dict['weights_path'] = model['model_path']+'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 2
        data_dict['labels_type'] = 'auto'
    elif dataset == 'affectnet':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 11
        data_dict['class_names'] = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
        data_dict['labels_type'] = 'csv'
        # data_dict['csv_train_file'] = data_dict['data_path'] + 'training.csv'
        data_dict['csv_train_file'] = data_dict['data_path'] + 'training_modified.csv'
        data_dict['csv_val_file'] = data_dict['data_path'] + 'validation_modified.csv'
        # data_dict['img_dir'] = data_dict['data_path'] + 'Manually_Annotated_Images/'
        data_dict['train_dir'] = data_dict['data_path'] + 'training/'
        data_dict['val_dir'] = data_dict['data_path'] + 'validation/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
        data_dict['class_label'] = 'expression'
        data_dict['box_labels'] = ['face_x', 'face_y', 'face_width', 'face_height']
    elif dataset == 'affectnet-small':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 11
        data_dict['num_val_images'] = 5500
        data_dict['class_names'] = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
        data_dict['labels_type'] = 'csv'
        data_dict['loss_type'] = 'categorical_crossentropy'
        data_dict['csv_train_file'] = data_dict['data_path'] + 'training_small.csv'
        # data_dict['csv_val_file'] = data_dict['data_path'] + 'validation_small.csv'
        data_dict['csv_val_file'] = data_dict['data_path'] + 'validation_modified.csv'
        data_dict['train_dir'] = data_dict['data_path'] + 'training_small/'
        # data_dict['val_dir'] = data_dict['data_path'] + 'validation_small/'
        data_dict['val_dir'] = data_dict['data_path'] + 'validation/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
        af_val = [8975, 16184, 3037, 1746, 734, 453, 4004, 489, 4050, 1381, 9947]
        data_dict['class_label'] = 'expression'
        data_dict['box_labels'] = ['face_x', 'face_y', 'face_width', 'face_height']
    elif dataset == 'affectnet-one-batch':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 11
        data_dict['num_val_images'] = 512
        data_dict['class_names'] = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
        data_dict['labels_type'] = 'csv'
        data_dict['loss_type'] = 'categorical_crossentropy'
        data_dict['csv_train_file'] = data_dict['data_path'] + 'training_one_batch.csv'
        data_dict['csv_val_file'] = data_dict['data_path'] + 'validation_one_batch.csv'
        data_dict['train_dir'] = data_dict['data_path'] + 'training_one_batch/'
        data_dict['val_dir'] = data_dict['data_path'] + 'validation_one_batch/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
        data_dict['class_label'] = 'expression'
        data_dict['box_labels'] = ['face_x', 'face_y', 'face_width', 'face_height']
    elif dataset == 'categorical-affectnet-one-batch':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 11
        data_dict['labels_type'] = 'auto'
        data_dict['train_dir'] = data_dict['data_path'] + 'categorical_training_one_batch/'
        data_dict['val_dir'] = data_dict['data_path'] + 'categorical_validation_one_batch/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    elif dataset == 'affectnet-sub5-12500':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 5
        data_dict['labels_type'] = 'auto'
        data_dict['train_dir'] = data_dict['data_path'] + 'train_sub5_12500/'
        data_dict['val_dir'] = data_dict['data_path'] + 'val_sub5_12500/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    elif dataset == 'affectnet-sub5-12500-unbalanced':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 5
        data_dict['labels_type'] = 'auto'
        data_dict['train_dir'] = data_dict['data_path'] + 'train_sub5_12500_unbalanced/'
        data_dict['val_dir'] = data_dict['data_path'] + 'val_sub5_12500/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    # elif dataset == 'affectnet-sub8':
    #     data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
    #     data_dict['weights_path'] = model['model_path'] + 'weights/'
    #     data_dict['metrics_path'] = model['model_path'] + 'metrics/'
    #     data_dict['n_classes'] = 8
    #     data_dict['num_val_images'] = 4000
    #     data_dict['class_names'] = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    #     data_dict['labels_type'] = 'auto'
    #     data_dict['loss_type'] = 'weighted_categorical_loss'
    #     data_dict['train_dir'] = data_dict['data_path'] + 'train_sub8/'
    #     data_dict['val_dir'] = data_dict['data_path'] + 'val_sub8/'
    #     # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
    #     data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
    #     data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    elif dataset == 'affectnet-sub8':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 8
        data_dict['num_val_images'] = 8
        data_dict['class_names'] = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        data_dict['labels_type'] = 'auto'
        data_dict['loss_type'] = 'weighted_categorical_loss'
        data_dict['train_dir'] = data_dict['data_path'] + 'train_sub8/'
        data_dict['val_dir'] = data_dict['data_path'] + 'shape-test/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    elif dataset == 'affectnet-sub11-38500':
        data_dict['data_path'] = computer['data_path'] + 'AffectNet/'
        data_dict['weights_path'] = model['model_path'] + 'weights/'
        data_dict['metrics_path'] = model['model_path'] + 'metrics/'
        data_dict['n_classes'] = 11
        data_dict['labels_type'] = 'auto'
        data_dict['train_dir'] = data_dict['data_path'] + 'train_sub11_38500/'
        data_dict['val_dir'] = data_dict['data_path'] + 'val_sub11_38500/'
        # data_dict['mean_RGB'] = [0.485, 0.456, 0.406]
        data_dict['mean_RGB'] = [0.540, 0.441, 0.3934]
        data_dict['std_RGB'] = [0.229, 0.224, 0.225]
    elif 'monkey' in dataset:
        data_dict['data_path'] = computer['data_path']+'Monkey/'+dataset+'/'
        data_dict['weights_path'] = model['model_path']+'weights/'+model['name']+'/'+dataset+'/'
        data_dict['metrics_path'] = model['model_path']+'metrics/'+model['name']+'/'+dataset+'/'
        data_dict['n_classes'] = 2
        data_dict['labels_type'] = 'auto'
    else:
        raise ValueError('Wrong or no dataset selected !!')

    return data_dict
