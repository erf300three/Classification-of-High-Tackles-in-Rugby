import os
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *

# These functions were created at the start of the project when we were experimenting with different
# data manipulation techniques. We decided to use the moviepy library for video manipulation as it was
# more flexible and easier to use than the opencv library. We used the moviepy library to trim, resize,
# flip and crop videos. We also used the os library to rename video files to a uniform naming convention.

def get_all_vids(in_dir):
    """
    Returns a list of all video files in the directory
    :param dir: directory to search
    :return: list of video files
    """
    vids = []
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            # Replace with uniform naming convention for all video files.
            file = file.replace("-", "")
            file = file.replace("  ", "_")
            file = file.replace(" ", "_")
            file = os.path.join(in_dir, file)
            vids.append(file)
    return vids

def rename_all_vids(in_dir):
    """
    Renames all video files in the directory to a uniform naming convention
    :param dir: directory to search
    :return: list of video files
    """
    base_path = os.path.abspath(in_dir)
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            # Replace with uniform naming convention for all video files.
            temp = file.replace("-", "")
            temp = temp.replace("  ", "_")
            temp = temp.replace(" ", "_")
            os.rename(base_path + "/" + file, base_path + "/" + temp)


def trim_all_vids(in_dir, out_dir, start_time, end_time):
    """
    Trims all videos in the directory
    :param dir: directory to search
    :param start_time: start time of the trim
    :param end_time: end time of the trim
    :return: whether the operation was successful
    """
    base_path = os.path.abspath(in_dir)
    output_path = base_path[:base_path.find("data") + 4] + out_dir
    base_path = base_path + "/"
    print(base_path)
    print(output_path)
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            print(file)
            ffmpeg_extract_subclip(base_path + file, start_time, end_time, targetname=f"{output_path}{file[:-4]}_trimmed.mp4")
    return True

def change_resolution(in_dir, out_dir, x_res, y_res):
    """
    Changes the resolution of all videos in the directory
    :param dir: directory to search
    :param x_res: x resolution
    :param y_res: y resolution
    :return: whether the operation was successful
    """
    base_path = os.path.abspath(in_dir)
    output_path = base_path[:base_path.find("data") + 4] + out_dir
    base_path = base_path + "/"
    print(base_path)
    print(output_path)
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            print(file)
            clip = VideoFileClip(base_path + file)
            clip_resized = clip.fx(vfx.resize, width=x_res, height=y_res)
            clip_resized.write_videofile(output_path + file[:-4] + "_resized.mp4")
    return True

def flip_horizontally(in_dir, out_dir):
    """
    Flips all videos in the directory horizontally
    :param dir: directory to search
    :return: whether the operation was successful
    """
    base_path = os.path.abspath(in_dir)
    output_path = base_path[:base_path.find("data") + 4] + out_dir
    base_path = base_path + "/"
    print(base_path)
    print(output_path)
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            print(file)
            clip = VideoFileClip(base_path + file)
            clip_flipped = clip.fx(vfx.mirror_x)
            clip_flipped.write_videofile(output_path + file[:-4] + "_flipped.mp4")
    return True

def crop_specific_region(in_dir, out_dir, width, height, x, y):
    """
    Crops all videos in the directory to a specific region
    :param dir: directory to search
    :param width: width of the cropped region
    :param height: height of the cropped region
    :param x: x coordinate of the cropped region
    :param y: y coordinate of the cropped region
    :return: whether the operation was successful
    """
    base_path = os.path.abspath(in_dir)
    output_path = base_path[:base_path.find("data") + 4] + out_dir
    base_path = base_path + "/"
    print(base_path)
    print(output_path)
    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            print(file)
            clip = VideoFileClip(base_path + file)
            clip_cropped = clip.crop(x, y, width, height)
            clip_cropped.write_videofile(output_path + file[:-4] + "_cropped.mp4")
    return True

def main():
    """
    This is the main function that parses the arguments and calls the appropriate function

    Arguments:
        --op: string, operation to perform on the data (rename, trim, resize, flip, crop)
        --dir: string, directory to perform the operation on
        --out_dir: string, output directory
        --start_time: float, start time of the trim (in seconds)
        --end_time: float, end time of the trim (in seconds)
        --x_res: int, x resolution (in pixels)
        --y_res: int, y resolution (in pixels)
        --width: int, width of the cropped region (in pixels)
        --height: int, height of the cropped region (in pixels)
        --x: int, x coordinate of the cropped region (in pixels)
        --y: int, y coordinate of the cropped region (in pixels)
    """

    parser = argparse.ArgumentParser(description='Manipulate data')
    parser.add_argument("--op", type=str, help="Operation to perform on the data") # rename, trim, resize, flip, crop
    parser.add_argument("--dir", type=str, help="Directory to perform the operation on")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--start_time", type=float, help="Start time of the trim") # in seconds (trim)
    parser.add_argument("--end_time", type=float, help="End time of the trim") # in seconds (trim)
    parser.add_argument("--x_res", type=int, help="X resolution") # in pixels (resize)
    parser.add_argument("--y_res", type=int, help="Y resolution") # in pixels (resize)
    parser.add_argument("--width", type=int, help="Width of the cropped region") # in pixels (crop)
    parser.add_argument("--height", type=int, help="Height of the cropped region") # in pixels (crop)
    parser.add_argument("--x", type=int, help="X coordinate of the cropped region") # in pixels (crop)
    parser.add_argument("--y", type=int, help="Y coordinate of the cropped region") # in pixels (crop)
    args = parser.parse_args()

    match args.op:
        case "rename":
            rename_all_vids(args.dir)
        case "trim":
            trim_all_vids(args.dir, args.out_dir, args.start_time, args.end_time)
        case "resize":
            change_resolution(args.dir, args.out_dir, args.x_res, args.y_res)
        case "flip":
            flip_horizontally(args.dir, args.out_dir)
        case "crop":
            crop_specific_region(args.dir,args.out_dir, args.width, args.height, args.x, args.y)

if __name__ == "__main__":
    main()
