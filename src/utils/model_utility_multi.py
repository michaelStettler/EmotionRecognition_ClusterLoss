import tensorflow as tf


def load_model(model_parameters, dataset_parameters):

    # setup for multi gpu
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # load optimizer with custom learning rate
    if model_parameters['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers. \
            SGD(lr=model_parameters['learning_rate'][0],
                momentum=0.9,
                nesterov=False)
    elif model_parameters['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            lr=model_parameters['learning_rate'][0])

    with strategy.scope():

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

        # adds a l2 kernel regularization to each conv2D layer
        for layer in model_template.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers. \
                    l2(model_parameters['l2_regularization'])

        # compile the model
        model_template.compile(loss=model_parameters['loss'],
                               optimizer=optimizer,
                               metrics=['mae', 'accuracy'])

    model_template.summary()

    # return the model template for saving issues with multi GPU
    return model_template
