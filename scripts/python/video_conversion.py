import os
import argparse
import moviepy.editor as moviepy

def convert_avi_to_mp4(input_dir, output_dir):
    """
    This function converts all .avi files in a directory to .mp4 files

    Input:
        input_dir: path to the directory containing the .avi files
        output_dir: path to the directory where the .mp4 files will be saved
    Output:
        All .avi files in the input directory will be converted to .mp4 files and saved in the
        output directory
    """
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".avi")]
    for filename in filenames:
        clip = moviepy.VideoFileClip(os.path.join(input_dir, filename))
        clip.write_videofile(output_dir + '/' + filename[:-4] + ".mp4")
        # Delete old .avi version of the file
        os.remove(os.path.join(input_dir, filename))

def convert_mov_to_mp4(input_dir, output_dir):
    """
    This function converts all .mov files in a directory to .mp4 files

    Input:
        input_dir: path to the directory containing the .mov files
        output_dir: path to the directory where the .mp4 files will be saved
    Output:
        All .mov files in the input directory will be converted to .mp4 files and saved in the
        output directory
    """
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".mov")]
    for filename in filenames:
        clip = moviepy.VideoFileClip(os.path.join(input_dir, filename))
        clip.write_videofile(output_dir + '/' + filename[:-4] + ".mp4")
        # Delete old .mov version of the file
        os.remove(os.path.join(input_dir, filename))

def convert_mp4_to_h264(input_dir, output_dir):
    """
    This function converts all .mp4 files in a directory to .mp4 files with h264 codec

    Input:
        input_dir: path to the directory containing the .mp4 files
        output_dir: path to the directory where the .mp4 files will be saved
    Output:
        All .mp4 files in the input directory will be converted to .mp4 files with h264 codec 
        and saved in the output directory
    """
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith(".mp4")]
    for filename in filenames:
        clip = moviepy.VideoFileClip(os.path.join(input_dir, filename))
        clip.write_videofile(output_dir + '/' + filename[:-4] + ".mp4")
        # Delete old .mp4 version of the file   
        os.remove(os.path.join(input_dir, filename))

def main():
    """
    This function is the main function that is called when the script is run. It parses the command
    line arguments and calls the appropriate function to convert the video formats

    Arguments:
        --format: Format to convert from
        --dir: Directory to perform the operation on
        --out_dir: Output directory
    """
    parser = argparse.ArgumentParser(description='Convert video formats data')
    parser.add_argument("--format", type=str, help="Format to convert from", required=True)
    parser.add_argument("--dir", type=str, help="Input directory", required=True)
    parser.add_argument("--out_dir", type=str, help="Output directory", required=True)
    args = parser.parse_args()
    match args.format:
        case "avi":
            convert_avi_to_mp4(args.dir, args.out_dir)
        case "mov":
            convert_mov_to_mp4(args.dir, args.out_dir)
        case "mp4":
            convert_mp4_to_h264(args.dir, args.out_dir)

if __name__ == "__main__":
    main()
