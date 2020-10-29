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

    # load object from file
    bpy.ops.import_scene.importldraw(filepath=brick_file_path)
    bpy.data.objects.remove(bpy.data.objects['LegoGroundPlane'])

    # create world
    world = bpy.data.worlds['World']
    
    # set world background
    path ='../LegoBrickClassification/Background2.jpg'
    img = bpy.data.images.load(path)
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    ShaderTexCo = nodes.new('ShaderNodeTexCoord')
    ShaderTexImg = nodes.new('ShaderNodeTexEnvironment')
    BSDF = nodes.new('ShaderNodeBackground')
    Output = nodes.new('ShaderNodeOutputWorld')
    img = bpy.data.images.load(path)
    ShaderTexImg.image = img
    links = world.node_tree.links
    link = links.new(ShaderTexCo.outputs[1],ShaderTexImg.inputs[0])
    link = links.new(ShaderTexImg.outputs[0],BSDF.inputs[0])
    #link = links.new(ShaderTexImg.outputs[1],BSDF.inputs[1])
    link = links.new(BSDF.outputs[0],Output.inputs[0])
    
    # create camera
    bpy.ops.object.add(type='CAMERA')
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # move camera position
    cam.location = cfg['world']['cam']['location']
    cam.rotation_euler = Euler(deg2rad(cfg['world']['cam']['rotation']), 'XYZ')

    bpy.ops.object.light_add(type='SUN',
                            radius=1,
                            align='WORLD',
                            location=(0, 0, 3)
                            )

    # create light
    bpy.ops.object.light_add(type=cfg_light['type'],
                            radius=cfg_light['radius'],
                            align='WORLD',
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
    bpy.context.scene.render.engine = 'CYCLES'
    render = bpy.data.scenes['Scene'].render
    render.resolution_x = render_cfg['width']
    render.resolution_y = render_cfg['height']
    render.resolution_percentage = render_cfg['resolution']
    bpy.context.scene.render.image_settings.file_format = render_cfg['format']
    bpy.context.scene.render.image_settings.color_mode = render_cfg['color_mode']
    bpy.context.scene.render.image_settings.quality = render_cfg['quality']  # compression in range [0, 100]
    bpy.context.scene.cycles.samples = 100
    bpy.context.scene.cycles.progressive = 'PATH'
    bpy.context.scene.cycles.max_bounces = 12
    bpy.context.scene.cycles.min_bounces = 0
    bpy.context.scene.cycles.glossy_bounces = 20
    bpy.context.scene.cycles.transmission_bounces = 12
    bpy.context.scene.cycles.volume_bounces = 20
    bpy.context.scene.cycles.transparent_max_bounces = 12
    bpy.context.scene.cycles.transparent_min_bounces = 0
    bpy.context.scene.cycles.use_progressive_refine = True
    #bpy.context.scene.render_aa = 'ON' # V2.81
    #bpy.context.scene.render.antialiasing_samples = '5' V2.81
    return render


def _init_brick(brick, cfg_brick):
    brick.location = cfg_brick['location']
    
    _set_brick_color([cfg_brick['color']], brick, random_color=True)
    
##    if len(brick.children) >= 1:  # brick with multiple parts
##        logging.debug('brick with multiple objects: %s', brick.children)
##        #scene = bpy.context.scene
##        # join sub-elements to a new brick
##        for obj in brick.children:
##            logging.info(obj)
##            #scene.collection.objects.link(obj)
##            #obj.select_set(True) #obj.select = True
##            #bpy.context.collection.objects.link(obj)  # bpy.context.scene.collection.objects.active(obj)
##            #logging.info(obj.dimensions)
##            
##        #bpy.ops.object.collection_objects_select()
##        logging.info(bpy.context.selected_objects)
##        logging.debug(brick.dimensions)
##        #bpy.ops.object.join()  # combine sub-elements
##        
        #bpy.ops.object.parent_clear(type='CLEAR')  # move group outside the parent

        #remove old brick
        #bpy.data.objects.remove(bpy.data.objects[brick.name], do_unlink = True)
        
        # set the new brick
##        logging.info("Selected=",bpy.context.selected_objects[0].name)
##        new = False
##        for obj in bpy.data.objects:
##            if obj.name == bpy.context.selected_objects[0].name:
##                new = True
##                logging.debug('object name: %s', obj.name)
##                brick = obj
##                logging.debug('new brick selected: %s', brick)
##        if not new:
##            e = 'new brick could not be selected'
##            logging.error(e)
##            raise ValueError(e)
    
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
                logging.debug(scale_factor)
                logging.debug(brick.dimensions)
                brick.location = cfg_brick['location']
                brick.rotation_euler = Euler(deg2rad(cfg_brick['rotation']))
                logging.debug(brick.rotation_euler)
                logging.debug(brick.scale)
                logging.debug(brick.location)
        except Exception as e:
            logging.error(e)
            raise e
    
    # set new origin to geometry center
    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.context.scene.objects:
        if bpy.context.object.name == obj.name:
            obj.select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # set origin to center
            bpy.context.object.location = cfg_brick['location']
            obj.select_set(False)
    
    # bpy.context.scene.update()
    if multiple_obj:
        brick = bpy.context.object
        bpy.ops.object.select_all(action='DESELECT')
    else:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if brick.name == obj.name:
                obj.select_set(True)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # set origin to center
                bpy.context.object.location = cfg_brick['location']
                obj.select_set(False)
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
        
        color = (hex2rgb(random.choice(colors)))
        color = color+ (1,) # Add Alpha
        logging.debug('brick random color: {}'.format(color))
    path ='../LegoBrickClassification/dust_dl.png'
    mat = brick.active_material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    ShaderTexCo = nodes.new('ShaderNodeTexCoord')
    ShaderTexCo.object = brick
    ShaderTexImg = nodes.new('ShaderNodeTexImage')
    BSDF = nodes.new('ShaderNodeBsdfPrincipled') #Split_RGB = nodes.new('ShaderNodeSeparateRGB')
    Output = nodes.new('ShaderNodeOutputMaterial')
    img = bpy.data.images.load(path)
    ShaderTexImg.image = img
    BSDF.inputs[0].default_value = color
    links = mat.node_tree.links
    link = links.new(ShaderTexCo.outputs[1],ShaderTexImg.inputs[0])
    link = links.new(ShaderTexImg.outputs[0],BSDF.inputs[3])
    link = links.new(ShaderTexImg.outputs[1],BSDF.inputs[18])
    link = links.new(BSDF.outputs[0],Output.inputs[0])
   
    if not brick.active_material and len(brick.children) == 0:
        logging.error(ValueError('Missing material!'))
    if brick.active_material:
        logging.info("brick_active")
        brick.active_material.diffuse_color = color
        
        
    
    else:  # brick consists of more than one parts/materials
        for obj in brick.children:
            logging.info(obj)
            if len(obj.material_slots) == 0:
                bpy.context.scene.objects.active = obj
                bpy.ops.object.material_slot_add()
            if len(obj.material_slots) == 0:
                logging.error(ValueError('no available material slot'))
                obj.active_material.diffuse_color = color
               





def random_background_surface(numx=20, numy=20, amp=0.2, scale=0.5, location=(0., 0., -0.4)):
    i=0
    # create mesh and object
    #bpy.ops.surface.primitive_nurbs_surface_surface_add(location=(0, 0, 0))
    #surfpatch = bpy.data.objects['SurfPatch']
    #surfpatch.rotation_euler = (np.pi, 0, 0)
    #surfpatch.scale = (100, 100, 100)
    #surfpatch.location[2] = 89
    #material = bpy.data.materials.new('surfpatch color')
    #material.diffuse_color = (128, 128, 128,128) #V2.81  4d not 3d
    #surfpatch.data.materials.append(material)

    #texture = bpy.data.textures.new('surfpatch texture', 'NOISE')
    #texture.noise_scale = 10
    #texture.noise_depth = 1
    #texture = bpy.data.textures.new('Bg',"IMAGE")
##    path ='/Users/petesmac/Documents/Machine Learning/LegoBrickClassification/Background.jpg'
##    img = bpy.data.images.load(path)
##    #texture.image = img
##    #ts = material.texture_slots.add()
##    #ts.texture = texture
##    #bpy.data.materials['surfpatch color'].texture_slots[0].color = (128,128,128,128)
##    #bpy.data.materials['surfpatch color'].texture_slots[0].diffuse_color_factor = 0.8
##    #mat = surfpatch.active_material
##    mat = bpy.data.worlds.new("World")
##    #mat = bpy.data.materials.exists(name="World")
##    #mat.use_nodes = True
##    nodes = mat.node_tree.nodes
##    nodes.clear()
##    #mat.diffuse_color = (0.5,0.5,0.5,0.8)
##    ShaderTexCo = nodes.new('ShaderNodeTexCoord')
##    #ShaderTexCo.object = surfpatch
##    #Mapping = nodes.new('ShaderNodeMapping')
##    ShaderTexImg = nodes.new('ShaderNodeTexImage')
##    BSDF = nodes.new('ShaderNodeBackground')
##    #Split_RGB = nodes.new('ShaderNodeSeparateRGB')
##    Output = nodes.new('ShaderNodeOutputMaterial')
##    img = bpy.data.images.load(path)
##    ShaderTexImg.image = img
##    links = mat.node_tree.links
##    link = links.new(ShaderTexCo.outputs[1],ShaderTexImg.inputs[0])
##    link = links.new(ShaderTexImg.outputs[0],BSDF.inputs[0])
##    link = links.new(ShaderTexImg.outputs[1],BSDF.inputs[1])
##    link = links.new(BSDF.outputs[0],Output.inputs[0])
    


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
    logging.info("render setup")
    # default location, rotation, color and size normalization
    brick = _init_brick(brick, cfg['brick'])
    logging.info("init brick")
    # set camera view to center
    if cfg['world']['cam']['augmentation']['enabled']:
        constraint = cam.constraints.new('TRACK_TO')
        constraint.target = brick
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
    logging.info("camer center")
    if cfg['world']['background']['surface']['enabled']:
        random_background_surface()
    logging.debug("background set")
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
            logging.info("Color: {}".format(c))
            #bg.data.materials['surfpatch color'].diffuse_color = c +(1,)

        augmentation = cfg['brick']['augmentation']
        if augmentation['rotation']['enabled']:
            rotx = deg2rad(random.choice(augmentation['rotation']['xvalues']))
            brick.rotation_euler[0] = rotx
            logging.debug('rotation parameters: rotx {}, roty {}, rotz {}'.format(rotx, brick.rotation_euler[1],
                                                                                  brick.rotation_euler[2]))

        if augmentation['zoom']['enabled']:
            zoom_factor = random.uniform(augmentation['zoom']['min'], augmentation['zoom']['max'])
            
            #brick.scale = zoom_factor * base_scale
            delta_location =cam.location- brick.location
            logging.info(delta_location)
            #exit(0)
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
        #world.texture_slots.clear(0)

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
    try:
        logging.info(args.input)
        render_brick(args.input, args.number, args.save, cfg)
    except Exception as e:
        logging.info("Render error")
        logging.error(e)
