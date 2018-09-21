import bpy
from mathutils import Euler

import os
import random
from math import pi


def render_brick(brick_file_path, background_file_path, number_of_images, render_folder, cfg):

    # remove all elements in scene
    bpy.ops.object.select_by_layer()
    bpy.ops.object.delete(use_global=False)

    # create world
    world = bpy.data.worlds.new("World")
    world.use_sky_paper = True
    bpy.context.scene.world = world

    # create camera
    bpy.ops.object.add(type='CAMERA')
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # create light
    bpy.ops.object.lamp_add(type='SUN', radius=1, view_align=False, location=(0,-1,0), rotation=(pi/2, 0,0))

    # create object
    bpy.ops.import_scene.importldraw(filepath=brick_file_path)
    bpy.data.objects.remove(bpy.data.objects['LegoGroundPlane'])

    # after loading brick, move camera position
    cam.location = (0, -1, 0)
    cam.rotation_euler = Euler((pi/2, 0, 0), 'XYZ')

    # brick selection
    for obj in bpy.data.objects:
        if (obj.name.endswith(".dat")):
            brick = obj
            break

    # render and image settings
    if(not os.path.exists(render_folder)):
        os.mkdir(render_folder)
    bpy.context.scene.render.engine = 'BLENDER_RENDER'
    rnd = bpy.data.scenes['Scene'].render
    rnd.resolution_x = cfg["width"]
    rnd.resolution_y = cfg["height"]
    rnd.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.quality = 70 # jpeg compression


    # list of possible background images
    images = []
    valid_ext = [".jpg", ".png"]
    for f in os.listdir(background_file_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() in valid_ext:
            images.append(os.path.join(background_file_path, f))

    for i in range(0, number_of_images):
        # brick settings
        brick_scale_factor = round(random.uniform(cfg['zoom_min'], cfg['zoom_max']), 1)
        brick_rotX = round(random.uniform(cfg['rot_min'], cfg['rot_max']), 1)
        brick_rotY = round(random.uniform(cfg['rot_min'], cfg['rot_max']), 1)
        brick_rotZ = round(random.uniform(cfg['rot_min'], cfg['rot_max']), 1)
        brick_posX = round(random.uniform(cfg['pos_min'], cfg['pos_max']), 1)
        brick_posY = round(random.uniform(cfg['pos_min'], cfg['pos_min']), 1)
        brick_posZ = round(random.uniform(cfg['pos_min'], cfg['pos_min']), 1)
        brick.scale = (brick_scale_factor, brick_scale_factor, brick_scale_factor)
        brick.location = (brick_posX, brick_posY, brick_posZ)
        brick.rotation_euler = (brick_rotX, brick_rotY, brick_rotZ)

        print("scale factor: {}".format(brick_scale_factor))
        print("position (x,y,z): {}, {}, {}".format(brick_posX, brick_posY, brick_posZ))
        print("rotation (x,y,z): {}, {}, {}".format(brick_rotX, brick_rotY, brick_rotZ))


        # select random background image
        bg_image = random.choice(images)
        image = bpy.data.images.load(bg_image)

        # set background image
        tex = bpy.data.textures.new(bg_image, 'IMAGE')
        tex.image = image
        slot = world.texture_slots.add()
        slot.texture = tex
        slot.use_map_horizon = True

        # render image
        rnd.filepath = os.path.join(render_folder, str(i) + '.jpg')
        bpy.ops.render.render(write_still=True)

        # remove current background
        world.texture_slots.clear(0)


if __name__ == '__main__':
    # check if script is opened in blender program
    import sys, json, argparse
    if(bpy.context.space_data == None):
        cwd = os.path.dirname(os.path.abspath(__file__))
    else:
        cwd = os.path.dirname(bpy.context.space_data.text.filepath)

    # get folder of script and add current working directory to path
    sys.path.append(cwd)

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:] # get all after first --

    # when --help or no args are given
    usage_text = (
        "Run blender in background mode with this script:"
        " blender -b -P " + __file__ + "-- [options]"
    )

    parser = argparse.ArgumentParser(description=usage_text)

    # create arguments
    parser.add_argument(
        "-i", "--input_files_path", dest="input", type=str, required=True,
        help="Input folder for 3d models"
    )

    parser.add_argument(
        "-b", "--background_files_path", dest="background", type=str, required=True,
        help="Input folder for background images"
    )

    parser.add_argument(
        "-n", "--images_per_brick", dest="number", type=int, required=False, default=1,
        help="Input folder for background images"
    )

    parser.add_argument(
          "-s", "--save", dest="save", type=str, required=False, default="./",
        help="Output folder"
    )

    args = parser.parse_args(argv)
    if not argv:
        parser.print_help()
        sys.exit(-1)
    if not (args.input or args.background):
        print("Error: Some required arguments missing")
        parser.print_help()
        sys.exit(-1)


    # load config file for blender settings
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    # finally render images
    render_brick(args.input, args.background, args.number, args.save, cfg)
