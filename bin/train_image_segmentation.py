import os
from os import mkdir, path
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import datetime
import json
import argparse

from src.data.data_loading_img_seg import load_data, input_output_paths
from src.models.autoencoder import Autoencoder
from src.models.unet import UNet
from src.utils.plotting import *
from src.trainer import Trainer


def main(config_path):
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    # configuration
    with open(config_path) as json_file:
        config = json.load(json_file)

    data_path = config['paths']['data']
    if 'opti' in config:
        learning_rate = config['opti']['learning_rate']
        weight_decay = config['opti']['weight_decay']
        num_epochs = config['opti']['num_epochs']
        log_step = config['opti']['log_step']
    else:
        learning_rate = 0.0001#0.00005
        weight_decay = 0
        num_epochs = 300
        log_step = 20

    # checkpoint saving folder and name
    model_folder_name = 'model/'
    if not (path.isdir(model_folder_name)):
        mkdir(model_folder_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # training data: paths to input images and target images
    input_img_paths, target_img_paths = \
        input_output_paths(data_path + 'train/')

    # training data and create data loader
    train_data, train_data_loader = \
        load_data(input_img_paths, target_img_paths)
    print("Number of training samples: ", len(train_data))

    # create network model
    if config['network']['type'] == 'autoencoder':
        nn_model = Autoencoder(config['network'])
    elif config['network']['type'] == 'u-net':
        nn_model = UNet(config['network'])
    else:
        print('model type not available ...')

    # move to GPU if possible
    if torch.cuda.is_available():
        nn_model = nn_model.cuda()
        print("Model moved to GPU in order to speed up training.")

    summary(nn_model, (1, 512, 512))

    # optimizer settings
    criterion = nn.BCELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    # load previous checkpoint if available
    last_epoch = 0
    model_path = config['paths']['model']
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        nn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        nn_model.train() # put in training mode

    # train model
    trainer = Trainer(nn_model, criterion, optimizer, model_folder_name)
    train_loss = trainer.train(train_data_loader, num_epochs, log_step,
                               timestamp, start_epoch=last_epoch)

    # save model for inference
    trainer.save_checkpoint(num_epochs, train_loss, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to the config '
                                                        'file',
                        default='config.json')
    args = parser.parse_args()
    main(config_path=args.config_path)
