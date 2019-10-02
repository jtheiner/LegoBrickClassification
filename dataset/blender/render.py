# blender imports
import bpy
from mathutils import Euler, Vector

# python imports
import argparse
import sys
import json
import os
import logging
import random
from copy import deepcopy

import numpy as np

print('numpy version: {}'.format(np.__version__))
sys.path.append('dataset')
from blender import sphere
from blender.utils import hex2rgb, deg2rad, random_like_color


def _init_world(cfg_bg, cfg_light, brick_file_path):
    # remove all elements in scene
    bpy.ops.object.select_by_layer()
    bpy.ops.object.delete(use_global=False)

    # create world
    world = bpy.data.worlds.new("World")
    world.use_sky_paper = True
    bpy.context.scene.world = world

    # set world background
    world.use_sky_blend = True
    world.horizon_color = hex2rgb(cfg_bg['horizon_color'])
    world.zenith_color = hex2rgb(cfg_bg['zenith_color'])
    world.ambient_color = hex2rgb(cfg_bg['ambient_color'])

    # load object from file
    bpy.ops.import_scene.importldraw(filepath=brick_file_path)
    bpy.data.objects.remove(bpy.data.objects['LegoGroundPlane'])

    # create camera
    bpy.ops.object.add(type='CAMERA')
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # move camera position
    cam.location = cfg['world']['cam']['location']
    cam.rotation_euler = Euler(deg2rad(cfg['world']['cam']['rotation']), 'XYZ')

    bpy.ops.object.lamp_add(type='SUN',
                            radius=1,
                            view_align=False,
                            location=(0, 0, 3)
                            )

    # create light
    bpy.ops.object.lamp_add(type=cfg_light['type'],
                            radius=cfg_light['radius'],
                            view_align=False,
                            location=cfg_light['location'],
                            rotation=deg2rad(cfg_light['rotation']),
                            )
    light = bpy.context.object
    light.data.energy = cfg_light['energy']
    if not cfg_light['random']:  # set light to camera position
        constraint = light.constraints.new('COPY_LOCATION')
        constraint.target = bpy.data.objects['Camera']

    return world, cam, light


def _get_brick():
    # brick selection
    for obj in bpy.data.objects:
        if (obj.name.endswith(".dat")):
            logging.debug('object name: %s', obj.name)
            brick = obj
            logging.debug('brick: %s', brick)
            return brick
    logging.error('unable to select brick')
    raise ValueError('brick cannot selected')


def _render_settings(render_folder, render_cfg):
    os.makedirs(render_folder, exist_ok=True)
    bpy.context.scene.render.engine = 'BLENDER_RENDER'
    render = bpy.data.scenes['Scene'].render
    render.resolution_x = render_cfg['width']
    render.resolution_y = render_cfg['height']
    render.resolution_percentage = render_cfg['resolution']
    bpy.context.scene.render.image_settings.file_format = render_cfg['format']
    bpy.context.scene.render.image_settings.color_mode = render_cfg['color_mode']
    bpy.context.scene.render.image_settings.quality = render_cfg['quality']  # compression in range [0, 100]
    return render


def _init_brick(brick, cfg_brick):
    brick.location = cfg_brick['location']
    _set_brick_color([cfg_brick['color']], brick, random_color=True)

    if len(brick.children) >= 1:  # brick with multiple parts
        logging.debug('brick with multiple objects: %s', brick.children)
        # join sub-elements to a new brick
        for obj in brick.children:
            obj.select = True
            bpy.context.scene.objects.active = obj
            # print(obj.dimensions)
            # print(obj.type)

        # print(bpy.context.selected_objects)
        bpy.ops.object.join()  # combine sub-elements
        bpy.ops.object.parent_clear(type='CLEAR')  # move group outside the parent

        # remove old brick
        bpy.data.objects.remove(bpy.data.objects[brick.name], True)
        # set the new brick
        new = False
        for obj in bpy.data.objects:
            if obj.name == bpy.context.selected_objects[0].name:
                new = True
                logging.debug('object name: %s', obj.name)
                brick = obj

                logging.debug('new brick selected: %s', brick)
        if not new:
            e = 'new brick could not be selected'
            logging.error(e)
            raise ValueError(e)

    # size normalization: set longest dimension to target size
    multiple_obj = False
    if cfg_brick['size_normalization']['enabled']:
        dim_target = cfg_brick['size_normalization']['target_dim']
        try:
            logging.debug(brick.dimensions)
            if brick.dimensions[0] == 0.0000:
                logging.debug(bpy.context.object.dimensions)
                scale_factor = dim_target / max(bpy.context.object.dimensions)
                bpy.context.object.dimensions = bpy.context.object.dimensions * scale_factor
                bpy.context.object.location = cfg_brick['location']
                #bpy.context.object.rotation_euler = Euler(deg2rad(cfg_brick['rotation']))
                multiple_obj = True
            else:
                scale_factor = dim_target / max(brick.dimensions)
                brick.dimensions = brick.dimensions * scale_factor
                logging.debug(brick.dimensions)
                brick.location = cfg_brick['location']
                brick.rotation_euler = Euler(deg2rad(cfg_brick['rotation']))
        except Exception as e:
            logging.error(e)
            raise e

    # set new origin to geometry center
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if bpy.context.object.name == obj.name:
            obj.select = True
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # set origin to center
            bpy.context.object.location = cfg_brick['location']
            obj.select = False

    # bpy.context.scene.update()
    if multiple_obj:
        brick = bpy.context.object
        bpy.ops.object.select_all(action='DESELECT')
    else:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if brick.name == obj.name:
                obj.select = True
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # set origin to center
                bpy.context.object.location = cfg_brick['location']
                obj.select = False
    return brick


def check_blender():
    if (bpy.context.space_data == None):
        cwd = os.path.dirname(os.path.abspath(__file__))
    else:
        cwd = os.path.dirname(bpy.context.space_data.text.filepath)
    # get folder of script and add current working directory to path
    sys.path.append(cwd)


def _set_brick_color(colors, brick, random_color=False):
    color = hex2rgb(colors[0])
    if random_color:
        color = hex2rgb(random.choice(colors))
        logging.debug('brick random color: {}'.format(color))

    if not brick.active_material and len(brick.children) == 0:
        logging.error(ValueError('Missing material!'))
    if brick.active_material:
        brick.active_material.diffuse_color = color

    else:  # brick consists of more than one parts/materials
        for obj in brick.children:
            if len(obj.material_slots) == 0:
                bpy.context.scene.objects.active = obj
                bpy.ops.object.material_slot_add()
            if len(obj.material_slots) == 0:
                logging.error(ValueError('no available material slot'))
            obj.material_slots[0].material = bpy.data.materials['Material']
            obj.active_material.diffuse_color = color


def random_background_surface(numx=20, numy=20, amp=0.2, scale=0.5, location=(0., 0., -0.4)):
    # create mesh and object
    bpy.ops.surface.primitive_nurbs_surface_surface_add(location=(0, 0, 0))
    surfpatch = bpy.data.objects['SurfPatch']
    surfpatch.rotation_euler = (np.pi, 0, 0)
    surfpatch.scale = (100, 100, 100)
    surfpatch.location[2] = 91
    material = bpy.data.materials.new('surfpatch color')
    material.diffuse_color = (1, 1, 1)
    surfpatch.data.materials.append(material)

    texture = bpy.data.textures.new('surfpatch texture', 'NOISE')
    # texture.noise_scale = 10
    # texture.noise_depth = 1

    ts = material.texture_slots.add()
    ts.texture = texture
    bpy.data.materials['surfpatch color'].texture_slots[0].color = (0, 0, 0)
    bpy.data.materials['surfpatch color'].texture_slots[0].diffuse_color_factor = 0.5

    return surfpatch


"""
def random_background_surface_flat(numx=20, numy=20, amp=0.2, scale=0.5, location=(0., 0., -0.4)):
    # source: http://wiki.theprovingground.org/blender-py-randmesh

    verts = []
    faces = []
    for i in range(numx):
        for j in range(numy):
            x = scale * i
            y = scale * j
            z = random.random() * amp
            vert = (x, y, z)
            verts.append(vert)

    count = 0
    for i in range(numy * (numy - 1)):
        if count < numy - 1:
            A = i
            B = i + 1
            C = (i + numy) + 1
            D = (i + numy)
            face = (A, B, C, D)
            faces.append(face)
            count = count + 1
        else:
            count = 0

    # create mesh and object
    mymesh = bpy.data.meshes.new("random mesh")
    myobject = bpy.data.objects.new("random mesh", mymesh)

    # create mesh from python data
    mymesh.from_pydata(verts, [], faces)
    mymesh.update(calc_edges=True)

    # subdivide modifier
    myobject.modifiers.new("subd", type='SUBSURF')
    myobject.modifiers['subd'].levels = 3
    # show mesh as smooth
    mypolys = mymesh.polygons
    for p in mypolys:
        p.use_smooth = True
    bpy.context.scene.objects.link(myobject)

    # set new origin to geometry center
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.name == 'random mesh':
            obj.select = True
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # set origin to center
            obj.select = False

    myobject.location = location

    material = bpy.data.materials.new('mesh color')
    myobject.data.materials.append(material)
    material.diffuse_color = (1, 1, 1)
    return myobject
"""


def render_brick(brick_file_path, number_of_images, render_folder, cfg):
    # create world, camera, light and background
    world, cam, light = _init_world(cfg['world']['background'], cfg['world']['light'], brick_file_path)
    logging.info('initialized world successfully')

    brick = _get_brick()

    # possible cam locations
    cfg_campos = cfg['world']['cam']['augmentation']
    if cfg_campos['enabled']:
        _, _, _, sphere_locations = sphere.get_positions(
            theta_range=deg2rad(cfg_campos['theta_range']),
            phi_range=deg2rad(cfg_campos['phi_range']),
            radius=cfg_campos['radius'],
            step_size=cfg_campos['step_size'],
            n_points_circle=cfg_campos['n_points_circle'],
            zlow=cfg_campos['zlow'],
            zhigh=cfg_campos['zhigh'])
        logging.debug('possible cam locations: {}'.format(len(sphere_locations)))

    render = _render_settings(render_folder, cfg['render'])

    # default location, rotation, color and size normalization
    brick = _init_brick(brick, cfg['brick'])

    # set camera view to center
    if cfg['world']['cam']['augmentation']['enabled']:
        constraint = cam.constraints.new('TRACK_TO')
        constraint.target = brick
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

    if cfg['world']['background']['surface']['enabled']:
        bg = random_background_surface()

    base_scale = deepcopy(brick.scale)
    # create n images using several augmentation parameters
    logging.info('start rendering %s images', number_of_images)
    for i in range(number_of_images):

        # randomly select one possible camera position
        if cfg['world']['cam']['augmentation']['enabled']:
            l = random.choice(sphere_locations)
            logging.debug('set new camera location: {}'.format(l))
            cam.location = l

        # randomly select one possible position for lighting
        if cfg['world']['light']['random']:
            light_position = random.choice(sphere_locations)
            logging.debug('set new light location: {}'.format(light_position))
            light.location = light_position

        # change bg color
        if cfg['world']['background']['surface']['enabled']:
            sf = cfg['world']['background']['surface']
            c = random_like_color(grayscale=sf['grayscale'], lower_limit=sf['lower_limit'], upper_limit=sf['upper_limit'])
            bg.data.materials['surfpatch color'].diffuse_color = c

        augmentation = cfg['brick']['augmentation']
        if augmentation['rotation']['enabled']:
            rotx = deg2rad(random.choice(augmentation['rotation']['xvalues']))
            brick.rotation_euler[0] = rotx
            logging.debug('rotation parameters: rotx {}, roty {}, rotz {}'.format(rotx, brick.rotation_euler[1],
                                                                                  brick.rotation_euler[2]))

        if augmentation['zoom']['enabled']:
            zoom_factor = random.uniform(augmentation['zoom']['min'], augmentation['zoom']['max'])
            brick.scale = zoom_factor * base_scale
            logging.debug('zoom factor: {}'.format(zoom_factor))
            logging.debug('brick scale {}'.format(brick.scale))

        if augmentation['translation']['enabled']:
            posx = random.uniform(augmentation['translation']['min'], augmentation['translation']['max'])
            posz = random.uniform(augmentation['translation']['min'], augmentation['translation']['max'])
            brick.location = (posx, 0.0, posz)
            logging.debug('brick location after translation: {}'.format(brick.location))
        if augmentation['color']['enabled']:
            _set_brick_color(augmentation['color']['colors'], brick, random_color=True)

        # render image
        brick_class = os.path.splitext(os.path.basename(brick_file_path))[0]
        render.filepath = os.path.join(render_folder, brick_class + '_' + str(i) + '.jpg')
        bpy.ops.render.render(write_still=True)

        # remove current background
        world.texture_slots.clear(0)

    return


def parse_args(parser):
    parser.add_argument(
        "-i", "--input_file_path", dest="input", type=str, required=True,
        help="Input 3d model"
    )

    parser.add_argument(
        "-n", "--images_per_brick", dest="number", type=int, required=False, default=1,
        help="number of images to render"
    )

    parser.add_argument(
        "-s", "--save", dest="save", type=str, required=False, default="./",
        help="output folder"
    )

    parser.add_argument(
        "-c", "--config", dest="config", required=False, default='augmentation.json',
        help="path to config file"
    )

    parser.add_argument(
        "-v", "--verbose", required=False, action="store_true",
        help="verbosity mode and log to file"
    )

    return parser.parse_args(argv)


if __name__ == '__main__':

    # check if script is opened in blender program
    check_blender()
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]  # get all after first --

    # when --help or no args are given
    usage_text = (
            "Run blender in background mode with this script:"
            " blender -b -P " + __file__ + "-- [options]"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    args = parse_args(parser)

    if not argv:
        parser.print_help()
        sys.exit(-1)
    if not (args.input or args.background):
        print("Error: Some required arguments missing")
        parser.print_help()
        sys.exit(-1)

    # load config file for blender settings
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', args.config)
    with open(cfg_path, 'r') as fr:
        cfg = json.load(fr)

    # init logger
    brick_id = os.path.splitext(os.path.basename(args.input))[0]
    format = '%(asctime)s [%(levelname)s] %(message)s'
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
        log_filename = os.path.join(args.save, 'logs', brick_id + '.log')
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(filename=log_filename, level=level, filemode='w', format=format)
    else:
        logging.basicConfig(level=level, filemode='w', format=format)
    logging.info('config file: %s', args.config)
    logging.getLogger().addHandler(logging.StreamHandler())

    # try:
    render_brick(args.input, args.number, args.save, cfg)
    # except Exception as e:
    #    logging.error(e)
