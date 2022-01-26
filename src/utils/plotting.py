import torch
import matplotlib.pyplot as plt
import numpy as np


def image_reconstruction(nn_model, sample_data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    num_samples = len(sample_data_loader.dataset)
    plot_grid_size = int(np.ceil(np.sqrt(num_samples)))
    plt.figure(1)
    plt.figure(2)
    k = 1
    j = 1
    for sample in sample_data_loader:
        img, _ = sample
        img = img.to(device)
        _, output = nn_model(img)
        for image in img:
            plt.figure(1)
            plt.subplot(plot_grid_size, plot_grid_size, j)
            plt.imshow(image.detach().cpu().numpy()[0, :, :])
            j += 1
        for image in output:
            plt.figure(2)
            plt.subplot(plot_grid_size, plot_grid_size, k)
            a = image.detach().cpu().numpy()[0, :, :]
            a[(a > 0.5)] = 1
            a[(a <= 0.5)] = 0
            # a = mp.binary_fill_holes(a)
            plt.imshow(a)
            k += 1
    plt.show()