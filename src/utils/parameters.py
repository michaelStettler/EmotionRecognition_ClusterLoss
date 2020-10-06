def load_model_parameters(model):
    model_dictionary = {'model_name': model}

    if 'RESNET50' in model.upper():
        model_dictionary['image_width'] = 224
        model_dictionary['image_height'] = 224
        model_dictionary['l2_regularization'] = 0.00001 / 2


def load_dataset_parameters(dataset_name, path):
    dataset_dictionary = {'dataset_name': dataset_name,
                          'mean_RGB': [0.485, 0.456, 0.406],
                          'std_RGB': [0.229, 0.224, 0.225]}

    if dataset_name.upper() == 'affectnet-small':
        dataset_dictionary['path'] = path + 'affectnet/'
        dataset_dictionary['n_classes'] = 11
        dataset_dictionary['num_val_images'] = 5500
        dataset_dictionary['class_names'] = ['Neutral', 'Happy', 'Sad',
                                             'Surprise', 'Fear', 'Disgust',
                                             'Anger', 'Contempt',
                                             'None', 'Uncertain', 'Non-Face']
        dataset_dictionary['labels_type'] = 'csv'
        dataset_dictionary['loss_type'] = 'categorical_crossentropy'
        dataset_dictionary['csv_training_file'] = \
            dataset_dictionary['path'] + 'training_small.csv'
        dataset_dictionary['csv_validation_file'] = \
            dataset_dictionary['data_path'] + 'validation_modified.csv'
        dataset_dictionary['training_directory'] = \
            dataset_dictionary['data_path'] + 'training_small/'
        dataset_dictionary['validation_directory'] = \
            dataset_dictionary['data_path'] + 'validation/'
        dataset_dictionary['mean_RGB'] = [0.540, 0.441, 0.3934]
        dataset_dictionary['std_RGB'] = [0.229, 0.224, 0.225]
        af_val = [8975, 16184, 3037, 1746, 734, 453, 4004, 489, 4050, 1381,
                  9947]
        dataset_dictionary['class_label'] = 'expression'
        dataset_dictionary['box_labels'] = ['face_x', 'face_y', 'face_width',
                                            'face_height']
