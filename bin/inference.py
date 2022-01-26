import torch
import json
import argparse

from src.data.data_loading_img_seg import load_data, input_output_paths
from src.models.autoencoder import Autoencoder
from src.models.unet import UNet
from src.utils.plotting import image_reconstruction


def main(config_path):
    torch.manual_seed(0)

    with open(config_path) as json_file:
        config = json.load(json_file)

    data_path = config['paths']['data']
    model_path = config['paths']['model']

    # load test set
    input_img_paths, _ = \
        input_output_paths(data_path + 'test/')
    target_img_paths = input_img_paths
    dev_data, dev_data_loader = load_data(
        input_img_paths, target_img_paths, shuffling=False)
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

    image_reconstruction(nn_model, dev_data_loader)

    print('finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to the config '
                                                        'file',
                        default='config.json')
    args = parser.parse_args()
    main(config_path=args.config_path)
