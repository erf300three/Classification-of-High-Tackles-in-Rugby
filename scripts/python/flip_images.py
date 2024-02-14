from PIL import Image
import os
import argparse


def flip_images(dir):
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            img = Image.open(dir + '/' + file)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(dir + '/' + file[:-4] + "_flipped.jpg")


def main():
    """
    Main function
    :return: true if video conversion is successful 
    """

    parser = argparse.ArgumentParser(description='Flip images')
    parser.add_argument("--dir", type=str, help="Directory to perform the operation on")
    args = parser.parse_args()
    flip_images(args.dir)

if __name__ == "__main__":
    main()