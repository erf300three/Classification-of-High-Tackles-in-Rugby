import os
import argparse
from PIL import Image


def flip_images(in_dir):
    """
    This function flips all images in a directory horizontally

    Input:
        in_dir: string, path to the directory containing images to flip

    Output:
        All images in the directory are flipped horizontally and saved with the same name with "_flipped" appended
    """
    for file in os.listdir(in_dir):
        if file.endswith(".jpg"):
            img = Image.open(in_dir + '/' + file)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(in_dir + '/' + file[:-4] + "_flipped.jpg")


def main():
    """
    This script flips all images in a directory horizontally using the PIL library and saves them with the same name
    with "_flipped" appended

    Arguments:
        --dir: string, path to the directory containing images to flip
    """

    parser = argparse.ArgumentParser(description='Flip images')
    parser.add_argument("--dir", type=str, help="Directory to perform the operation on")
    args = parser.parse_args()
    flip_images(args.dir)

if __name__ == "__main__":
    main()
