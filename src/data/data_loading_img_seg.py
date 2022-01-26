from os import listdir
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
# ------------------------------------------------------------------
# functions
# ------------------------------------------------------------------


def load_data(input_path, target_path, shuffling=True):
    bs = 16 # batch size
    nw = 12  # num workers

    transform_in = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize(512),
                                       transforms.Normalize(0.5, 0.5)])

    transform_out = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(512)])

    train_data = ImageDataSet(input_path, target_path,
                              transform_in, transform_out)
    train_data_loader = data.DataLoader(
        train_data, batch_size=bs, shuffle=shuffling, num_workers=nw,
        pin_memory=True)

    return train_data, train_data_loader


def input_output_paths(data_path):

    filenames = listdir(data_path)
    filenames.sort()
    input_filenames = []
    output_filenames = []
    for filename in filenames:
        if filename.startswith('input_') and \
                (filename.endswith('.tiff') or filename.endswith('.tif')):
            input_filenames.append(filename)
        elif filename.startswith('output_') and \
                (filename.endswith('.tiff') or filename.endswith('.tif')):
            output_filenames.append(filename)
    input_img_paths = np.core.defchararray.add(
        np.repeat(data_path, len(input_filenames)), input_filenames)
    target_img_paths = np.core.defchararray.add(
        np.repeat(data_path, len(output_filenames)), output_filenames)

    return input_img_paths, target_img_paths
# ------------------------------------------------------------------
# classes
# ------------------------------------------------------------------


class ImageDataSet():
    def __init__(self, input_paths, target_paths,
                 transform_in=None, transform_out=None):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __getitem__(self, index):
        with open(self.input_paths[index], 'rb') as f:
            input_img = Image.open(f)
            input_img.load()
        with open(self.target_paths[index], 'rb') as f:
            target_img = Image.open(f)
            target_img.load()

        if self.transform_in:
            input_img = self.transform_in(input_img)
        if self.transform_out:
            target_img = self.transform_out(target_img)

        return input_img, target_img

    def __len__(self):
        return len(self.input_paths)
