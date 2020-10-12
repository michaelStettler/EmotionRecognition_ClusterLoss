import tensorflow as tf


def load_model(model_parameters, dataset_parameters):

    if 'resnet50' in model_parameters['model_name']:
        model_template = tf.keras.applications.ResNet50(
            include_top=model_parameters['include_top'],
            weights=model_parameters['weights'],
            input_shape=(model_parameters['image_width'],
                         model_parameters['image_height'],
                         3),
            classes=dataset_parameters['num_classes']
        )

    elif 'resnet50v2' in model_parameters['model_name']:
        model_template = tf.keras.applications.ResNet50V2(
            include_top=model_parameters['include_top'],
            weights=model_parameters['weights'],
            input_shape=(model_parameters['image_width'],
                         model_parameters['image_height'],
                         3),
            classes=dataset_parameters['num_classes']
        )

    elif 'resnet101' in model_parameters['model_name']:
        model_template = tf.keras.applications.ResNet101(
            include_top=model_parameters['include_top'],
            weights=model_parameters['weights'],
            input_shape=(model_parameters['image_width'],
                         model_parameters['image_height'],
                         3),
            classes=dataset_parameters['num_classes']
        )

    elif 'resnet50_blob' in model_parameters['model_name']:
        model_template = tf.keras.applications.ResNet50(
            include_top=model_parameters['include_top'],
            weights=model_parameters['weights'],
            input_shape=(model_parameters['image_width'],
                         model_parameters['image_height'],
                         3),
            classes=dataset_parameters['num_classes']
        )

    # adds a l2 kernel regularization to each conv2D layer
    for layer in model_template.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.\
                l2(model_parameters['l2_regularization'])

    # construct multi GPU if possible
    if len(tf.config.list_physical_devices('GPU')) > 2:
        model = tf.keras.utils.multi_gpu_model(model_template)
    else:
        model = model_template

    # return the model template for saving issues with multi GPU
    return model, model_template

