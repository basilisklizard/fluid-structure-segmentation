import numpy as np
from PIL import Image
import csv
import argparse
from scipy.ndimage.filters import gaussian_filter

from src.data.synthetic_data import ParameterSet, ParticleDistribution, \
    Triangle, Rectangle, Disk


# -----------------------------------------------------------------------------
# utility functions
# -----------------------------------------------------------------------------
def convert_to_image(img):

    min_val, max_val = np.min(img), np.max(img)
    if min_val != max_val:
        img = np.uint8((img-min_val)/(max_val-min_val) * 255)
    else:
        img = np.uint8(img)
    return img


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(save_folder):

    # Parameters
    RAND_SEED = 1
    num_training_img = 1  # number of training images (input/output pairs)
    img_size = 1024         # size of square image

    ps_range = [0.001, 0.3]  # particle densities (of all particles, size 1
    # & 2)
    ps2_range = [0.1, 0.2]  # particle size 2 densities (fraction of total)
    g_noise_mu = 0.1        # added gaussian noise mean
    g_noise_sigma = 0.05     # added gaussian noise std
    g_blur = [0.5, 0.7]       # gaussian blur sigma range
    min_intensity = [0, 0.3]    # min brightness level
    max_num_shape = 3       # max number of triangles per img
    scale = [100, 300]      # scaling factor for shapes
    rot = [0, 2*np.pi]      # rotation in rad for shapes
    max_intensity = [0, 1]  # intensity range for shapes

    csv_header = ['particle_density_range', ps_range,
                  'particle_size2_density_range', ps2_range,
                  'gaussian_noise', [g_noise_mu, g_noise_sigma],
                  'gaussian_blur', g_blur,
                  'min_brightness', min_intensity,
                  'max_num_shape', max_num_shape,
                  'scale_range', scale, 'rot_range', rot,
                  'max_intensity_range', max_intensity,
                  'random_seed', RAND_SEED]

    # Scaling (utility) function
    scaling = lambda vec, scal_range: \
        vec * (scal_range[1]-scal_range[0]) + scal_range[0]

    # Create training data parameter set
    np.random.seed(RAND_SEED)
    training_data_params = []
    p_set = ParameterSet(scaling, ps_range, ps2_range, min_intensity,
                         max_num_shape, scale, rot, img_size, max_intensity,
                         g_blur)
    csv_output = [csv_header, p_set.get_csv_header()]
    for param_set in range(num_training_img):
        p_set = ParameterSet(scaling, ps_range, ps2_range, min_intensity,
                             max_num_shape, scale, rot, img_size,
                             max_intensity, g_blur)
        csv_output.append(p_set.get_csv_output_line())
        training_data_params.append(p_set)

    # write csv-file
    with open(save_folder + 'data_overview.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(csv_output)

    counter = 0
    for param_set in training_data_params:

        print("p1 density: ", param_set.ps,
              " | p2 density: ", param_set.ps2,
              " | min intensity: ", param_set.min_intensity,
              " | blur sigma: ", param_set.blur_sigma,
              " | num_trig: ", param_set.num_trig,
              " | num_rect: ", param_set.num_rect,
              " | num disk: ", param_set.num_disk)

        # output (masked) image
        img_out = np.zeros((img_size, img_size))

        # input (particle) image
        pd = ParticleDistribution(img_size, param_set.ps, param_set.ps2,
                                  g_noise_mu, g_noise_sigma,
                                  param_set.min_intensity)
        img_in = pd.get_particle_image()

        # place objects in input and output images
        for trig_param in param_set.trigs:
            tr = Triangle()
            tr.get_pixel_form(
                trig_param.scale, trig_param.rot, trig_param.x, trig_param.y)
            tr.put_shape_into_image(trig_param.intensity, img_in, True)
            tr.put_shape_into_image(1, img_out)

        for rect_param in param_set.rects:
            rt = Rectangle()
            rt.get_pixel_form(
                rect_param.scale, rect_param.rot, rect_param.x, rect_param.y)
            rt.put_shape_into_image(rect_param.intensity, img_in, True)
            rt.put_shape_into_image(1, img_out)

        for disk_param in param_set.disks:
            ds = Disk()
            ds.get_pixel_form(
                disk_param.scale, disk_param.rot, disk_param.x, disk_param.y)
            ds.put_shape_into_image(disk_param.intensity, img_in, True)
            ds.put_shape_into_image(1, img_out)

        img_in = gaussian_filter(img_in, param_set.blur_sigma)

        # save image files
        img_in = convert_to_image(img_in)
        im = Image.fromarray(img_in)
        im.save(save_folder + "input_" + str(format(counter, '05d'))
                + ".tiff")
        img_out = convert_to_image(img_out)
        im = Image.fromarray(img_out)
        im.save(save_folder + "output_" + str(format(counter, '05d'))
                + ".tiff")

        counter += 1

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, help='target folder to '
                                                        'save data in',
                        required=True)
    args = parser.parse_args()
    main(save_folder=args.save_folder)