import moviepy.editor as moviepy
import os
import argparse

def convert_avi_to_mp4(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".avi"):
            clip = moviepy.VideoFileClip(input_dir + '/' + file)
            clip.write_videofile(output_dir + '/' + file[:-4] + ".mp4")
            # Delete old .avi version of the file   
            os.remove(input_dir + '/' + file)

# convert_avi_to_mp4("data/output_set/tracked", "data/output_set/tracked")

def convert_mov_to_mp4(input_dir, output_dir):
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".mov")]
    for filename in filenames:
        clip = moviepy.VideoFileClip(input_dir + '/' + filename)
        clip.write_videofile(output_dir + '/' + filename[:-4] + ".mp4")
        # Delete old .mov version of the file   
        os.remove(input_dir + '/' + filename)

def main():
    """
    Main function
    :return: true if video conversion is successful 
    """

    parser = argparse.ArgumentParser(description='Convert video formats data')
    parser.add_argument("--format", type=str, help="Format to convert from")
    parser.add_argument("--dir", type=str, help="Directory to perform the operation on")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    args = parser.parse_args()
    match args.format:
        case "avi":
            convert_avi_to_mp4(args.dir, args.out_dir)
        case "mov":
            convert_mov_to_mp4(args.dir, args.out_dir)

if __name__ == "__main__":
    main()