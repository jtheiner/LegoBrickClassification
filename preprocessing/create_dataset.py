import os
import sys
import argparse
from typing import List

from create_part_category_list import create_part_category_list


def get_blender_command(path_in, bg_images_path, images_per_brick, path_out, debug=True):
    # path or reference to blender executable
    if sys.platform == "darwin":
        blender = "/Applications/Blender/blender.app/Contents/MacOS/blender"
    else:
        blender = "blender"

    bg = "-b='" + bg_images_path + "' " if bg_images_path else ""
    debug = " --debug" if debug else ""

    return (blender + " -b -P "
            + os.path.dirname(__file__)
            + "/render_brick.py -- "
            + "-i='" + path_in + "' "
            + bg
            + "-n=" + str(images_per_brick) + " "
            + "-s='" + path_out + "'"
            + debug)


def run_shell_script(command: str):
    try:
        os.system(command)
    except OSError as oe:
        print("Error while executing blender script")
        print(oe)
        raise


def generate_single_category_dataset(dataset_in_path: str, category: str, bg_images_path: str, dataset_out_path: str,
                                     images_per_brick: int):
    """ Generates images for a given category.

    A folder is created for each part of the category.

    :param dataset_in_path: directory contains a set of .dat files
    :param category: lego category (see: create_part_category_list)
    :param bg_images_path: directory contains a set of images for background
    :param dataset_out_path: output path for generated images e.g. dataset/
    :param images_per_brick: number of images per brick
    :return: writes generated images to file
    """

    df = create_part_category_list(dataset_in_path)
    df = df[df.category.str.match(category)]

    if len(df.index) > 0:
        # read all 3d files
        files = []
        for file in os.listdir(dataset_in_path):
            if file.endswith(".dat") and file[:-4] in df.id.values:
                files.append(file)

        # for each brick render IMAGES_PER_BRICk
        for file in files:
            part_number = file[:-4]  # remove extension to receive the part number

            path_out = os.path.join(dataset_out_path, category, part_number)
            path_in = os.path.join(dataset_in_path, file)

            # create folders titled by category and part number
            if not os.path.exists(path_out):
                os.makedirs(path_out)

            # run blender python script to render images
            command = get_blender_command(path_in, bg_images_path, images_per_brick, path_out)
            run_shell_script(command)
    else:
        raise(ValueError("category '{}' not found".format(category)))


def generate_single_part_dataset(file_path: str, bg_images_path: str, dataset_out_path: str, images_per_brick: int):
    """Renders images for a given brick.

    :param file_path: path to .dat file
    :param bg_images_path: path to folder of images used as background
    :param dataset_out_path: output directory
    :param images_per_brick: number of images to render
    :return: writes rendered images to out_path
    """
    part_number = file_path[:-4]  # remove extension to receive the part number

    path_out = os.path.join(dataset_out_path, part_number) + "/"

    with open(file_path, 'r') as f:
        line = f.readline()
        if "~Moved to" in line:
            print("Part id moved to another one!")

    # create folder titled by part number
    if not os.path.exists(os.path.join(dataset_out_path + part_number)):
        os.makedirs(dataset_out_path + part_number)

    # have to execute a blender script
    command = get_blender_command(file_path, bg_images_path, images_per_brick, path_out)
    run_shell_script(command)


def generate_dataset(dataset_in_path: str, bg_images_path: str, dataset_out_path: str, images_per_brick: int,
                     except_list: List[str]):
    """Generates a dataset of images using 3d models.

    A directory is created for each category and each category folder contains subdirectories for each individual brick.

    :param dataset_in_path: directory contains a set of .dat files
    :param bg_images_path: directory contains a set of images for background
    :param dataset_out_path: output path for generated images e.g. dataset/
    :param images_per_brick: number of images per brick
    :param except_list: list of categories in order to skip
    :return: Writes generated images to file
    """

    # read all 3d files
    files = []
    for file in os.listdir(dataset_in_path):
        if file.endswith(".dat"):
            files.append(file)

    # for each brick render IMAGES_PER_BRICk

    for i, file in enumerate(files):
        part_number = file[:-4]  # remove extension to receive the part number
        print("processing file {} ({}/{})".format(part_number, i+1, len(files)))
        path_in = os.path.join(dataset_in_path, file)
        with open(path_in, 'r') as f:
            line = f.readline()
            # check whether file number is moved to another and skip if true
            if "~Moved to" in line:
                print("skip part: {}".format(line))
                continue
            # skip unimportant categories
            label = line[2:-1]
            if '~' in label:
                label = label.replace('~', '')
            if label.startswith('_'):
                label = label.replace('_', '')
            if label.startswith('='):
                label = label.replace('=', '')
            category = label.split(' ')[0]
            if category in except_list:
                print("skip part: {}".format(line))
                continue

        path_in = os.path.join(dataset_in_path, file)
        path_out = os.path.join(dataset_out_path, category, part_number)
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # run blender python script to render images
        command = get_blender_command(path_in, bg_images_path, images_per_brick, path_out)
        run_shell_script(command)


if __name__ == "__main__":

    argv = sys.argv[1:]

    usage_text = "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument(
        "-t", "--type", dest="type", required=True, choices=["part", "category", "full"],
        help="Generate images for a single 'part', 'category' or 'full'"
    )

    parser.add_argument(
        "-i", "--input", dest="dataset_in", type=str, required=True,
        help="Input folder for all .dat files or path to file if category 'part' is selected"
    )

    parser.add_argument(
        "-b", "--bg_images", dest="bg_images", type=str, required=False,
        help="Directory which holds the background images"
    )

    parser.add_argument(
        "-o", "--out_dataset", dest="dataset_out", type=str, required=False,
        default="results/dataset/",
        help="Output folder for generated images"
    )

    parser.add_argument(
        "-n", "--images", dest="images_per_brick", type=int, required=False,
        default=1,
        help="Number of generated images per brick"
    )

    parser.add_argument(
        "-c", "--category", dest="category", type=str, required=False,
        default="Brick",
        help="Category label for image generation (see: create_part_category_list for details)"
    )

    args = parser.parse_args(argv)

    if args.type == "part":
        generate_single_part_dataset(args.dataset_in, args.bg_images, args.dataset_out, args.images_per_brick)

    elif args.type == "category":
        generate_single_category_dataset(args.dataset_in, args.category, args.bg_images, args.dataset_out,
                                         args.images_per_brick)

    elif args.type == "full":
        # skip all parts which are assigned to one of these categories
        except_list = ["Minifig", "Sticker", "Duplo", "Figure", "Pov-RAY"]
        generate_dataset(args.dataset_in, args.bg_images, args.dataset_out, args.images_per_brick, except_list)
