"""

will run train_model.py using a loop that will go over the define parameter
python3 multiple_train.py -c a
"""

from train_model import *

from argparse import ArgumentParser

if __name__ == '__main__':
    model_name = 'resnet50v2'  # resnet18 resnet50 vgg16 simple
    dataset = 'affectnet-sub8'  # imagenet affectnet test monkey affectnet_sub5_12500
    mode = 'full'  # full tl
    weights = 'imagenet'  # needed only for transfer learning
    run = '01'
    da = '1'
    version = '00'
    cw = '2'
    task = 'classification'

    parser = ArgumentParser()
    parser.add_argument("-c", "--computer",
                        default='a',
                        help="select computer. a:980ti, b:2x1080ti, c:cluster")
    parser.add_argument("-g", "--gpu",
                        default='0',
                        help="set the gpu to use")
    args = parser.parse_args()

    computer = args.computer
    gpus = args.gpu

    # loop_params = ['1', '5', '6', '7', '8', '9', '10']
    # loop_params = ['3a', '3b', '3c']
    # loop_params = ['1a', '1b', '1c', '1d']
    # loop_params = ['5a']
    loop_params = ['2d', '2e']
    # loop_params = ['10']

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    for i in loop_params:
        # train_model(model_name=model_name, dataset=dataset, mode=mode, weights=weights, computer=computer, run=i)
        train_model(model_name=model_name,
                    dataset=dataset,
                    mode=mode,
                    weights=weights,
                    computer=computer,
                    run=run,
                    da=da,
                    version=i,
                    task=task,
                    class_weights=cw)
