
import os
import sys

def generate_dataset(dataset_in_path, bg_images_path, dataset_out_path, images_per_brick):
    """
        Generates a dataset of images using 3d models.

        Args:
            dataset_in_path: Directory contains a set of .dat files
            bg_images_path: Directory contains a set of images for background
            dataset_out_path: Output path for generated images e.g. dataset/
            images_per_brick: Number of images per brick

        Returns:
            None: Writes images to file
    """


    # read all 3d files
    files = []
    for file in os.listdir(dataset_in_path):
        if file.endswith(".dat"):
            files.append(file)

    # for each brick render IMAGES_PER_BRICk
    for i, file in enumerate(files):
        print("process file {} ({}/{})".format(os.path.join(dataset_in_path, file), i+1, len(files)))
        part_number = file[:-4] # remove extension to receive the part number
        if not os.path.exists(os.path.join(dataset_out_path + part_number)):
            os.makedirs(dataset_out_path + part_number)

        path_out = os.path.join(dataset_out_path, part_number) + "/"
        path_in = os.path.join(dataset_in_path, file)

        # have to execute a blender script
        command = ("blender -b -P render_brick.py -- "
                "-i='" + path_in + "' "
                "-b='" + bg_images_path + "' "
                "-n=" + str(images_per_brick) + " "
                "-s='" + path_out + "'"
                )
        print(command)
        # run blender python script to render images      
        os.system(command)

if __name__ == "__main__":
    import sys, argparse
    argv = sys.argv[1:]

    usage_text =  "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument(
        "-i", "--in_dataset", dest="dataset_in", type=str, required=False,
        default="res/parts_50/",
        help="Input folder for all .dat files"
    )

    parser.add_argument(
        "-b", "--bg_images", dest="bg_images", type=str, required=False,
        default="res/bg_noise/",
        help="Directory which holds the background images"
    )

    parser.add_argument(
        "-o", "--out_dataset", dest="dataset_out", type=str, required=False,
        default="../dataset/",
        help="Output folder for generated images"
    )

    parser.add_argument(
    "-n", "--images", dest="images_per_brick", type=int, required=False,
    default=50,
    help="Number of generated images per brick"
    )

    args = parser.parse_args(argv)
 
    if not argv:
        parser.print_help()
        sys.exit(-1)

    generate_dataset(args.dataset_in, args.bg_images, args.dataset_out, args.images_per_brick)
    