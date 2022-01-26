import torch
import json
import scipy.ndimage.morphology as mp
from PIL import Image
import argparse

from src.data.data_loading_img_seg import load_data, input_output_paths
from src.models.autoencoder import Autoencoder
from src.models.unet import UNet


def retrieve_filenames(file_paths):

    filenames = []

    for file in file_paths:
        filename = file.rsplit('/', 1)[-1]
        filename = filename.rsplit('.', 1)[0]
        filenames.append(filename)

    return filenames


def main(config_path, data_path, save_path, num_imgs):

    with open(config_path) as json_file:
        config = json.load(json_file)

    model_path = config['paths']['model']

    # load dataset
    input_img_paths, _ = input_output_paths(data_path)
    target_img_paths = input_img_paths
    data, data_loader = load_data(input_img_paths, target_img_paths,
                                  shuffling=False)
    img_names = retrieve_filenames(input_img_paths)
    print("Number of images: ", len(data))

    # retrieve image size
    im = Image.open(input_img_paths[0])
    dataset_img_size = im.size

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
    nn_model.eval()

    # apply model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_id = 0

    for sample in data_loader:
        print('starting with new batch, current img_id: ', img_id)
        img, _ = sample
        img = img.to(device)
        _, output = nn_model(img)

        for image in output:
            a = image.detach().cpu().numpy()[0, :, :]
            a[(a > 0.5)] = 1
            a[(a <= 0.5)] = 0
            a = mp.binary_fill_holes(a)
            a = mp.binary_closing(a)
            # a = median_filter(a, size=20)
            # a = mp.binary_dilation(a, iterations=5)
            img_export = Image.fromarray(a)
            img_export = img_export.resize(dataset_img_size)
            if img_id < num_imgs:
                img_export.save(save_path + img_names[img_id] + '_mask.tif')
            img_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to the config '
                                                        'file',
                        default='config.json')
    parser.add_argument('--data_path', type=str, help='path to data folder '
                                                      'that contains images '
                                                      'to be segmented',
                        required=True)
    parser.add_argument('--save_path', type=str, help='target path for '
                                                      'created masks',
                        required=True)
    parser.add_argument('--num_imgs', type=int, help='number of images to '
                                                     'process in the data '
                                                     'folder',
                        required=True)
    args = parser.parse_args()
    main(config_path=args.config_path, data_path=args.data_path,
         save_path=args.save_path, num_imgs=args.num_imgs)
