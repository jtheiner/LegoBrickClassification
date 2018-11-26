import os
import pandas as pd


def create_part_category_list(dat_directory):
    """
    Extract IDs, labels and categories from .dat files.
    Stores content to csv file

    :param dat_directory: path to .dat files
    :return: pandas dataframe with columns [id, label, category]
    """

    # read parts directory containing all .dat files
    files = []
    for file in os.listdir(dat_directory):
        if file.endswith(".dat"):
            files.append(file)

    parts = []
    for filename in files:
        part_number = filename[:-4]
        with open(os.path.join(dat_directory, filename), 'r') as f:
            first_line = f.readline()
            if "~Moved to" in first_line:
                continue
            else:
                label = first_line[2:-1]  # skip zero and space
                if '~' in label:
                    label = label.replace('~', '')
                if label.startswith('_'):
                    label = label.replace('_', '')
                if label.startswith('='):
                    label = label.replace('=', '')
                category = label.split(' ')[0]
                parts.append([part_number, label, category])

    df = pd.DataFrame().from_records(parts, columns=["id", "label", "category"])

    print(df.category.value_counts())

    df.sort_values(by=["category", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("results/parts_category_list.csv", index=False)
    return df


def generate_single_category_dataset(dataset_in_path, category, bg_images_path, dataset_out_path, images_per_brick):
    df = create_part_category_list(dataset_in_path)
    df = df[df.category == category]

    # read all 3d files
    files = []
    for file in os.listdir(dataset_in_path):
        print(file)
        if file.endswith(".dat") and file[:-4] in df.id.values:
            files.append(file)

    # for each brick render IMAGES_PER_BRICk
    for file in files:
        part_number = file[:-4]  # remove extension to receive the part number

        path_out = os.path.join(dataset_out_path, part_number) + "/"
        path_in = os.path.join(dataset_in_path, file)

        # create folder titled by part number
        if not os.path.exists(os.path.join(dataset_out_path + part_number)):
            os.makedirs(dataset_out_path + part_number)

        # have to execute a blender script
        command = ("blender -b -P "
                   + os.path.dirname(__file__)
                   + "/render_brick.py -- "
                     "-i='" + path_in + "' "
                                        "-b='" + bg_images_path + "' "
                                                                  "-n=" + str(images_per_brick) + " "
                                                                                                  "-s='" + path_out + "'")

        # run blender python script to render images
        os.system(command)


def generate_single_part_dataset(file_path, bg_images_path, dataset_out_path, images_per_brick):
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
    command = ("blender -b -P " + os.path.dirname(__file__) + "/render_brick.py -- "
                                                              "-i='" + file_path + "' "
                                                                                   "-b='" + bg_images_path + "' "
                                                                                                             "-n=" + str(
        images_per_brick) + " "
                            "-s='" + path_out + "'"
               )

    os.system(command)

    return None


def generate_dataset(dataset_in_path, bg_images_path, dataset_out_path, images_per_brick, part_limit=None,
                     except_list=None):
    """
        Generates a dataset of images using 3d models.

        Args:
            dataset_in_path: Directory contains a set of .dat files
            bg_images_path: Directory contains a set of images for background
            dataset_out_path: Output path for generated images e.g. dataset/
            images_per_brick: Number of images per brick
            except_list: list of categories in order to skip

        Returns:
            None: Writes images to file
    """

    # read all 3d files
    files = []
    for file in os.listdir(dataset_in_path):
        if file.endswith(".dat"):
            files.append(file)

    # for each brick render IMAGES_PER_BRICk

    counter = 0
    for file in files:
        part_number = file[:-4]  # remove extension to receive the part number

        path_out = os.path.join(dataset_out_path, part_number) + "/"
        path_in = os.path.join(dataset_in_path, file)
        if except_list != None:
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

        if part_limit != None and counter >= part_limit:
            sys.exit(0)
        counter += 1

        # create folder titled by part number
        if not os.path.exists(os.path.join(dataset_out_path + part_number)):
            os.makedirs(dataset_out_path + part_number)

        # have to execute a blender script
        command = ("blender -b -P " + os.path.dirname(__file__) + "/render_brick.py -- "
                                                                  "-i='" + path_in + "' "
                                                                                     "-b='" + bg_images_path + "' "
                                                                                                               "-n=" + str(
            images_per_brick) + " "
                                "-s='" + path_out + "'"
                   )

        # run blender python script to render images
        # print(command)
        os.system(command)


if __name__ == "__main__":
    import sys, argparse

    argv = sys.argv[1:]

    usage_text = "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument(
        "-t", "--type", dest="type", type=str, required=False,
        default="category",
        help="Generate images for a single 'part', 'category' or 'full'"
    )

    parser.add_argument(
        "-i", "--in_dataset", dest="dataset_in", type=str, required=False,
        default="res/parts_13463/",
        help="Input folder for all .dat files"
    )

    parser.add_argument(
        "-b", "--bg_images", dest="bg_images", type=str, required=False,
        default="res/bg_noise/",
        help="Directory which holds the background images"
    )

    parser.add_argument(
        "-o", "--out_dataset", dest="dataset_out", type=str, required=False,
        default="results/dataset/",
        help="Output folder for generated images"
    )

    parser.add_argument(
        "-n", "--images", dest="images_per_brick", type=int, required=False,
        default=10,
        help="Number of generated images per brick"
    )

    parser.add_argument(
        "-c", "--category", dest="category", type=str, required=False,
        default="Brick",
        help="Category label for image generation"
    )

    args = parser.parse_args(argv)

    if args.type == "part":
        generate_single_part_dataset(args.dataset_in, args.bg_images, args.dataset_out,
                                     args.images_per_brick)

    elif args.type == "category":
        generate_single_category_dataset(args.dataset_in, args.category, args.bg_images, args.dataset_out,
                                         args.images_per_brick)

    elif args.type == "full":
        # skip all parts which are assigned to one of these categories
        except_list = ["Minifig", "Sticker", "Duplo", "Figure", "Pov-RAY"]
        generate_dataset(args.dataset_in, args.bg_images, args.dataset_out, args.images_per_brick,
                         except_list=except_list)
