import os
from PIL import Image
import argparse


def main(data_path, target_path, num_imgs):

    # crop and save
    left = 100
    upper = 0
    width = 1024
    height = 1024
    box = (left, upper, left + width, upper + height)
    k = 0
    img_names = os.listdir(data_path)
    img_names.sort()

    for img_name in img_names:
        if img_name.endswith(".tif") and k < num_imgs:
            print(img_name)

            img_path = os.path.join(data_path, img_name)
            img = Image.open(img_path)

            img = img.crop(box)

            img.save(target_path + "/input_" + img_name)

            k = k+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to the data '
                                                      'folder',
                        required=True)
    parser.add_argument('--target_path', type=str, help='path to folder '
                                                        'where processed '
                                                        'images are saved',
                        required=True)
    parser.add_argument('--num_imgs', type=int, help='number of images to '
                                                     'process in the data '
                                                     'folder',
                        required=True)
    args = parser.parse_args()
    main(data_path=args.data_path, target_path=args.target_path,
         num_imgs=args.num_imgs)
