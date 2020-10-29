import tensorflow as tf


def load_model(model_parameters, dataset_parameters):
    # setup for multi gpu
    strategy = tf.distribute.MirroredStrategy()
    print('** Number of devices: {} **'.format(strategy.num_replicas_in_sync))

    if model_parameters['lr_decay']:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=model_parameters['learning_rate'],
            decay_steps=model_parameters['lr_decay_steps'],
            decay_rate=model_parameters['lr_decay_rate'])
    else:
        learning_rate = model_parameters['learning_rate']

    # load optimizer with custom learning rate
    if model_parameters['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=model_parameters['momentum'])

    elif model_parameters['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

    with strategy.scope():

        if model_parameters['model_name'] == 'resnet50':
            model_template = tf.keras.applications.ResNet50(
                include_top=model_parameters['include_top'],
                weights=model_parameters['weights'],
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes']
            )

        elif model_parameters['model_name'] == 'resnet50v2':
            print('** loaded resnet50v2 **')
            model_template = tf.keras.applications.ResNet50V2(
                include_top=model_parameters['include_top'],
                weights=model_parameters['weights'],
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes'],
                classifier_activation=model_parameters['activation']
            )

        elif model_parameters['model_name'] == 'resnet101':
            model_template = tf.keras.applications.ResNet101(
                include_top=model_parameters['include_top'],
                weights=model_parameters['weights'],
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes']
            )

        if model_parameters['l2_regularization']:
            # adds a l2 kernel regularization to each conv2D layer
            print('** added l2 regularization **')
            for layer in model_template.layers:
                if isinstance(layer, tf.keras.layers.Conv2D) or \
                        isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_regularizer = tf.keras.regularizers. \
                        l2(model_parameters['l2_regularization'])

        if model_parameters['loss'] == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=model_parameters['from_logits'])
            print('** loss is categorical_crossentropy, from logits is {}'
                  .format(model_parameters['from_logits']))

        # compile the model
        model_template.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=['mae', 'accuracy'])

        print(model_template.get_config())

    # return the model template for saving issues with multi GPU
    return model_template
