"""
Script to train and save weights of the models

run full training: python3 train_model.py -m resnet18 -d affectnet_small -g 0,1,2 -c b -v 6 -da 2 -r 01
run chop off: python3 train_model.py -m vgg16 -d affectnet_one_batch -c m -mode tl -da 1 -r 01

"""

import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras import backend as K
from sklearn.utils import class_weight
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation, MaxPooling2D


import sys
import os.path
import glob
from argparse import ArgumentParser
import numpy as np
import time

sys.path.insert(0, '../utils')
from parameters import *
from utils import *
from model_utils import *
from dataset import *

print("keras version:", keras.__version__)
print("floatx", keras.backend.floatx())


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))


def save_metrics(history, metrics):
    metrics[0] = metrics[0] + history.losses
    metrics[1] = metrics[1] + history.accuracies
    metrics[2] = metrics[2] + history.val_losses
    metrics[3] = metrics[3] + history.val_accuracies


def set_transfer_learning(base_model, template_model, model_params, data):
    # freeze the layers you don't want to train
    for layer in base_model.layers[:model_params['n_fix_layers']]:
        layer.trainable = False
    for layer in template_model.layers[:model_params['n_fix_layers']]:
        layer.trainable = False

    x = base_model.output
    x_template = template_model.get_output_at(-1)

    if model_params['name'] == 'vgg16':
        x = Flatten()(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(data['n_classes'], activation='softmax', name='predictions')(x)

        x_template = Flatten()(x_template)
        x_template = Dense(4096, activation='relu', name='fc1')(x_template)
        x_template = Dense(4096, activation='relu', name='fc2')(x_template)
        x_template = Dense(data['n_classes'], activation='softmax', name='predictions')(x_template)
    elif 'resnet50' in model_params['name']:  # ideas from https://github.com/nikhil-salodkar/facial_expression/blob/master/facial_expression.ipynb
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(8, activation='softmax')(x)

        x_template = GlobalAveragePooling2D()(x_template)
        x_template = Dense(512, activation='relu')(x_template)
        x_template = Dense(8, activation='softmax')(x_template)
    else:
        x = Dense(data['n_classes'], activation='softmax', name='fc' + str(data['n_classes']))(x)
        x_template = Dense(data['n_classes'], activation='softmax', name='fc' + str(data['n_classes']))(x_template)

    weights = 'imagenet'
    print('CHANGE WEIGHTS NAMMMMEEEEEEE')
    return Model(base_model.inputs, x, name=model_params['name'] + '-tl-' + weights), \
           Model(template_model.inputs, x_template, name=model_params['name'] + '-tl-' + weights)


def train_model(model_name='Simple',
                dataset='monkey_2',
                mode='full',
                weights='imagenet',
                computer='a',
                run='00',
                da='0',
                version='0',
                task='classification',
                class_weights=None):

    # record starting time
    start = time.time()

    # load the hyper parameters
    model_params = load_model_params(model_name, version, class_weights)
    computer = load_computer_params(computer, model_params)
    data = load_dataset_params(dataset, model_params, computer)

    # define if we want to train from scratch or apply transfer learning
    if mode == 'full':
        include_top = True
        weights = None
    elif mode == 'tl':  # transfer learning
        include_top = False
    else:
        raise ValueError('Please select a training mode')

    # model template serves to save the model even with multi GPU training
    model, model_template = load_model(include_top, weights, model_params, data)

    # modifiy the model in case of transfert learning
    if mode == 'tl':
        model, model_template = set_transfer_learning(model, model_template, model_params, data)

    model.summary()
    # define the loss
    loss = load_loss(data['loss_type'], model_params)

    if task == 'regression':
        loss = keras.losses.mean_squared_error

    if model_params['optimizer'] == 'sgd':
        optimizer = keras.optimizers.SGD(lr=model_params['lr'][0], momentum=0.9, nesterov=False)
    elif model_params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=0.000024, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])
    # need to compile the template as well!!!!!!!!! the template is used for saving
    model_template.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mae', 'accuracy'])
    history = LossHistory()
    metrics = [[], [], [], []]

    print("data_processing loading")
    data_start_time = time.time()
    print("1")
    train_generator, validation_generator = get_generator(data, model_params, da, task)
    print("done loading data_processing (%.2fs)" % (time.time() - data_start_time))
    print()

    # define names
    if mode == 'full':
        if class_weights is not None:
            model_save_name = '%s_%s_%s_da-%s_v-%s_cw-%s_%s' % (model_name, task, dataset, da, version, class_weights, run)
        else:
            model_save_name = '%s_%s_%s_da-%s_v-%s_%s' % (model_name, task, dataset, da, version, run)
        # checkpoint_save_name = data_processing['weights_path'] + 'checkpoint_%s_%s_%s_da-%s_v-%s_%s_{epoch:02d}-{val_acc:.2f}.h5' % (model_name, task, dataset, da, version, run)
        checkpoint_save_name = data['weights_path'] + 'checkpoint_%s_%s_%s_da-%s_v-%s_%s.h5' % (model_name, task, dataset, da, version, run)
    elif mode == 'tl':
        if class_weights is not None:
            model_save_name = '%s_%s_%s_tl-%s_w-%s_da-%s_v-%s_cw-%s_%s' % (model_name, task, dataset, model_params['n_fix_layers'], weights, da, version, class_weights, run)
        else:
            model_save_name = '%s_%s_%s_tl-%s_w-%s_da-%s_v-%s_%s' % (model_name, task, dataset, model_params['n_fix_layers'], weights, da, version, run)
        # checkpoint_save_name = data_processing['weights_path'] + 'checkpoint_%s_%s_%s_tl-%s_w-%s_da-%s_v-%s_%s_{epoch:02d}-{val_acc:.2f}' % (model_name, task, dataset, model_params['n_fix_layers'], weights, da, version, run)
        checkpoint_save_name = data['weights_path'] + 'checkpoint_%s_%s_%s_tl-%s_w-%s_da-%s_v-%s_%s.h5' % (model_name, task, dataset, model_params['n_fix_layers'], weights, da, version, run)
    else:
        model_save_name = 'my_model'
        checkpoint_save_name = 'checkpoint_my_model_{epoch:02d}-{val_acc:.2f}.h5'

    # checkpoint
    # checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_save_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint, history]
    callbacks_list = [history]

    # train the model using the different learning rate according to the epochs
    print(model_params['num_epochs'])
    for i, num_epoch in enumerate(model_params['num_epochs']):
        K.set_value(model.optimizer.lr, model_params['lr'][i])

        # ************************************************************************************************************ #
        # fit_generator
        model.fit_generator(train_generator, epochs=num_epoch,
                            validation_data=validation_generator,
                            validation_steps=128,
                            callbacks=callbacks_list,
                            workers=12,
                            # use_multiprocessing=True,
                            class_weight=class_weights)
        save_metrics(history, metrics)
        print("metrics updated")
        # ************************************************************************************************************ #

        # ************************************************************************************************************ #
        # # train_on_batch
        # for epoch in range(num_epoch):
        #     print("epoch", epoch)
        #     batches = 0
        #     print("labels", train_generator.class_indices)
        #     for x_batch, y_batch in train_generator:
        #         # model.train_on_batch(X_batch, Y_batch, sample_weight=class_weights)
        #         sample_weights = class_weight.compute_sample_weight('balanced', y_batch)
        #         print("shape y_batch", np.shape(y_batch))
        #         print(y_batch)
        #         print("-----------------------------------")
        #         print(np.sum(y_batch, axis=0))
        #         print("sample_weights")
        #         print(sample_weights)
        #         # print("sample_weights")
        #         # print(sample_weights)
        #         loss = model.train_on_batch(x_batch, y_batch, sample_weight=sample_weights)
        #         # write_log(callback, train_names, logs, batch_no)
        #         metrics[0].append(loss[0])
        #         metrics[1].append(loss[2])
        #         print("batch", batches, loss)
        #         print()
        #
        #         if batches >= 151:
        #             break
        #         batches += 1
        #
        #     # batches = 0
        #     # for x_batch, y_batch in validation_generator:
        #     #     sample_weights = class_weight.compute_sample_weight('balanced', y_batch)
        #     #     loss = model.test_on_batch(x_batch, y_batch, sample_weight=sample_weights)
        #     #     metrics[2].append(loss[0])
        #     #     metrics[3].append(loss[2])
        #     #     print("batch", batches, loss)
        #     #
        #     #     if batches >= 1:
        #     #         break
        #     #     batches += 1
        #     validation_generator.reset()
        #     loss = model.evaluate_generator(validation_generator,
        #                                     max_queue_size=10,
        #                                     workers=9,
        #                                     use_multiprocessing=True,
        #                                     verbose=1)
        #     metrics[2].append(loss[0])
        #     metrics[3].append(loss[2])
        #     print("batch", loss)

        # ************************************************************************************************************ #

    print("model trained!")
    evaluation = model.evaluate_generator(validation_generator,
                                          workers=12,
                                          # use_multiprocessing=True,
                                          verbose=1)
    print("evaluation", evaluation)
    print("evaluation", model.metrics_names)
    model_template.save(data['weights_path']+model_save_name+'.h5')
    np.save(data['metrics_path']+'metrics_%s' % model_save_name, metrics)
    print("model saved")

    total_time = time.time() - start
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print("Total time: %.0f hours %.0f min  %.0f sec" % (h, m, s))
    print("mode save name:", model_save_name)


if __name__ == '__main__':
    weights = 'imagenet'  # needed only for transfer learning

    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        default='simple',
                        help="select which model to use: 'resnet18', 'resnet50', 'vgg16', 'simple'")
    parser.add_argument("-d", "--dataset",
                        default='imagenet',
                        help="select which dataset to train on: 'imagenet', 'affectnet', 'test', 'monkey'")
    parser.add_argument("-c", "--computer",
                        default='a',
                        help="select computer. a:980ti, b:2x1080ti, c.cluster")
    parser.add_argument("-r", "--run",
                        default='00',
                        help="set the run number")
    parser.add_argument("-g", "--gpu",
                        default='0',
                        help="set the gpu to use")
    parser.add_argument("-v", "--version",
                        default='0',
                        help="set the version to use")
    parser.add_argument("-mode", "--mode",
                        default='full',
                        help="select if train all (full) or use transfer learning (tl)")
    parser.add_argument("-t", "--task",
                        default='classification',
                        help="select the kind of learning, classification or regression")
    parser.add_argument("-da", "--data_augmentation",
                        default='0',
                        help="select which data_processing augmentation to perform")
    parser.add_argument("-cw", "--class_weights",
                        default=None,
                        help="select which class weights to set into the weighted loss")
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    computer = args.computer
    run = args.run
    gpus = args.gpu
    version = args.version
    mode = args.mode
    task = args.task
    da = args.data_augmentation
    cw = args.class_weights

    print("------------------------------------------------------------")
    print("                       Summary                              ")
    print()
    print("model:", model_name, "dataset:", dataset, "task:", task)
    print("computer:", computer, "run:", run, "gpu:", gpus, "version:", version, "da:", da)
    print()
    print("------------------------------------------------------------")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    train_model(model_name=model_name,
                dataset=dataset,
                mode=mode,
                weights=weights,
                computer=computer,
                run=run,
                da=da,
                version=version,
                task=task,
                class_weights=cw)

    print()
    print()
    print("------------------------------------------------------------")
    print("                       Summary End                          ")
    print()
    print("model:", model_name, "dataset:", dataset, "task:", task)
    print("computer:", computer, "run:", run, "gpu:", gpus, "version:", version, "da:", da)
    print()
    print("------------------------------------------------------------")
    print()
