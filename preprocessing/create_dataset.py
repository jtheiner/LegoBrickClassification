
import os
import sys

PATH = "res/parts_50/"  # directory holding the .dat files
PATH_BG_IMAGES = "res/bg_noise/" # directory holding the background images
PATH_OUT = "../dataset/" # output directory
IMAGES_PER_BRICK = 50

# read all 3d files
files = []
for file in os.listdir(PATH):
    if file.endswith(".dat"):
        files.append(file)

# for each brick render IMAGES_PER_BRICk
for i, file in enumerate(files):
    print("process file {} ({}/{})".format(os.path.join(PATH, file), i+1, len(files)))
    part_number = file[:-4] # remove extension to receive the part number
    if not os.path.exists(os.path.join(PATH_OUT + part_number)):
        os.makedirs(PATH_OUT + part_number)

    path_out = os.path.join(PATH_OUT, part_number) + "/"
    path_in = os.path.join(PATH, file)

    # have to execute a blender script
    command = ("blender -b -P render_brick.py -- "
               "-i='" + path_in + "' "
               "-b='" + PATH_BG_IMAGES + "' "
               "-n=" + str(IMAGES_PER_BRICK) + " "
               "-s='" + path_out + "'"
               )
    print(command)
    # run blender python script to render images      
    os.system(command)
    