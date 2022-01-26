import torch
from torchsummary import summary
import json
import argparse

from src.data.data_loading_img_seg import load_data, input_output_paths
from src.models.autoencoder import Autoencoder
from src.models.unet import UNet
from src.trainer import Trainer
from src.utils.plotting import image_reconstruction


def main(config_path):
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    with open(config_path) as json_file:
        config = json.load(json_file)

    data_path = config['paths']['data']
    model_path = config['paths']['model']

    # load training set
    input_img_paths, target_img_paths = \
        input_output_paths(data_path + 'train/')
    train_data, train_data_loader = \
        load_data(input_img_paths, target_img_paths)
    print("Number of train samples: ", len(train_data))

    # load dev set
    input_img_paths, target_img_paths = \
        input_output_paths(data_path + 'dev/')
    dev_data, dev_data_loader = \
        load_data(input_img_paths, target_img_paths)
    print("Number of dev samples: ", len(dev_data))

    # create model
    if 'type' in config['network']:
        if config['network']['type'] == 'autoencoder':
            nn_model = Autoencoder(config['network'])
        elif config['network']['type'] == 'u-net':
            nn_model = UNet(config['network'])
        else:
            print('model type not available ...')
    else:
        nn_model = Autoencoder(config['network'])

    # load model parameters
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')

    if 'type' in config['network']:
        model_params = checkpoint['model_state_dict']
    else:
        model_params = checkpoint

    nn_model.load_state_dict(model_params)
    nn_model.eval() #to set dropout and batch normalization to evaluation mode

    summary(nn_model, (1, 512, 512))

    criterion = torch.nn.BCELoss()
    trainer = Trainer(nn_model, criterion, optimizer='', save_path='')
    train_loss = trainer.eval_model(train_data_loader)
    dev_loss = trainer.eval_model(dev_data_loader)
    print('training loss: ', train_loss)
    print('dev loss: ', dev_loss)

    image_reconstruction(nn_model, dev_data_loader)
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to the config '
                                                        'file',
                        default='config.json')
    args = parser.parse_args()
    main(config_path=args.config_path)
