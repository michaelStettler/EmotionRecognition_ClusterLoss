import tensorflow as tf

from src.utils.convert_json_dict import convert_keys_to_int
from src.model_functions.WeightedSoftmaxCluster import ClusterLayer
from src.model_functions.WeightedSoftmaxCluster import WeightedSoftmaxLoss2
from src.model_functions.WeightedSoftmaxCluster import WeightedClusterLoss


def load_model(model_parameters, dataset_parameters, train=False):
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

        if train:
            weights = None
        else:
            weights = model_parameters['weights']

        if model_parameters['model_name'] == 'resnet50':
            model_template = tf.keras.applications.ResNet50(
                include_top=model_parameters['include_top'],
                weights=weights,
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes']
            )

        elif model_parameters['model_name'] == 'resnet50v2':
            model_template = tf.keras.applications.ResNet50V2(
                include_top=model_parameters['include_top'],
                weights=weights,
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes'],
                classifier_activation=model_parameters['activation']
            )
            print('** loaded resnet50v2 **')
        elif model_parameters['model_name'] == 'resnet50v2_ClusterLoss':
            from src.models.resnet_prelu import ResNet50V2_prelu
            print('** loaded resnet50v2 ClusterLoss**')
            model_template = ResNet50V2_prelu(
                include_top=model_parameters['include_top'],
                weights=weights,
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classifier_activation=model_parameters['activation']
            )

        elif model_parameters['model_name'] == 'resnet101':
            model_template = tf.keras.applications.ResNet101(
                include_top=model_parameters['include_top'],
                weights=weights,
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes']
            )
        elif model_parameters['model_name'] == 'CORnet_S':
            from src.models.CORnet_S import CORnet_S
            model_template = CORnet_S(classes=dataset_parameters['num_classes'],
                                      from_logits=model_parameters[
                                          'from_logits'])

        elif model_parameters['model_name'] == 'vgg19':
            model_template = tf.keras.applications.VGG19(
                include_top=model_parameters['include_top'],
                weights=weights,
                input_shape=(model_parameters['image_width'],
                             model_parameters['image_height'],
                             3),
                classes=dataset_parameters['num_classes'],
                classifier_activation=model_parameters['activation']
            )
            print('** loaded VGG19 **')
        else:
            raise ValueError("Model does not exists in load_model")

        if model_parameters['l2_regularization']:
            # adds a l2 kernel regularization to each conv2D layer
            print('** added l2 regularization **')
            for layer in model_template.layers:
                if isinstance(layer, tf.keras.layers.Conv2D) or \
                        isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_regularizer = tf.keras.regularizers. \
                        l2(model_parameters['l2_regularization'])

        if model_parameters['use_cluster_loss']:
            # -------------------------------------------------------------------------------------------------------------
            # add cluster
            # cl_weights = [float(134414 / 24882), float(134414 / 3750),
            #               float(134414 / 3803), float(134414 / 6378),
            #               float(134414 / 134414), float(134414 / 74874),
            #               float(134414 / 25759), float(134414 / 14090)]
            # # cl_weights = [float(3750 / 24882), float(3750 / 3750),  # as paper
            # #               float(3750 / 3803), float(3750 / 6378),
            # #               float(3750 / 134414), float(3750 / 74874),
            # #               float(3750 / 25759), float(3750 / 14090)]
            # "class_weights": [5.402, 35.844, 35.344, 21.074, 1.0, 1.79, 5.218, 9.539],
            cl_weights = dataset_parameters['class_weights']
            print("class weights", cl_weights)

            labels = tf.keras.Input(shape=(dataset_parameters['num_classes'],),
                                    dtype='int32')
            inputs = tf.keras.Input(shape=(224, 224, 3), dtype='float32')
            x = model_template(inputs)
            x = tf.keras.layers.Dense(512, name='fc2')(x)
            x = tf.keras.layers.PReLU()(x)
            output = tf.keras.layers.Dense(dataset_parameters['num_classes'],
                                           name='output')(x)
            cluster = ClusterLayer(
                num_classes=dataset_parameters['num_classes'],
                class_weight=cl_weights,
                name='cluster')([x, labels])
            model_template = tf.keras.Model(inputs=[inputs, labels],
                                            outputs=[output, cluster])
            print("Cluster layer added")

            # compile the model
            # model_template.compile(loss={'output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            model_template.compile(loss={'output': WeightedSoftmaxLoss2(10, cl_weights, from_logits=True),
                                'cluster': WeightedClusterLoss(cl_weights, _lambda=0.01)},
                          optimizer=optimizer,
                          metrics={'output': ['mae', tf.keras.metrics.CategoricalAccuracy()]})
        else:
            if model_parameters['loss'] == 'categorical_crossentropy':
                loss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=model_parameters['from_logits'])
                print('** loss is categorical_crossentropy, from logits is {}'
                      .format(model_parameters['from_logits']))

            # compile the model
            model_template.compile(loss=loss,
                                   optimizer=optimizer,
                                   metrics=['mae', 'accuracy'])

    # return the model template for saving issues with multi GPU
    return model_template
