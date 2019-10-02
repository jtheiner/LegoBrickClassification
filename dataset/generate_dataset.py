import subprocess
from functools import partial
from multiprocessing.pool import Pool
import os
import pandas as pd

dataset_files_base = 'resources/parts/complete-190827/ldraw/parts/'
dataset_path = 'data/dataset-15/dataset.csv'
config_fname = 'augmentation.json'
output_path = 'data/dataset-15/images'
number_of_images = 2000

df = pd.read_csv(dataset_path, encoding='utf-8', index_col='id')

dataset_files = [os.path.join(dataset_files_base, str(f) + '.dat') for f in df.index]
# dataset_files = dataset_files[:20]

# todo: load background images in memory before rendering process begins, possible in blender?
# alternative: remove already used "background material" in blender
def _render(idx_fname, output_path: str, list_length: int, config_fname: str, number_of_images=1):

    index, fname = idx_fname
    part_id = os.path.splitext(os.path.basename(fname))[0]
    if os.path.exists(os.path.join(output_path, part_id)):
        print('{} ({}/{}): already exists'.format(fname, index + 1, list_length))
        return
    print('{} ({}/{}): render'.format(fname, index + 1, list_length))
    render_script_path = os.path.join(os.path.dirname(__file__), 'blender', 'render.py')
    command = 'blender -b -P ' + render_script_path + ' --' \
              + ' -i ' + fname \
              + ' -c ' + config_fname \
              + ' -s ' + os.path.join(output_path, part_id) \
              + ' -n ' + str(number_of_images)
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    # except Exception as e:
    #     print(e)
    except subprocess.TimeoutExpired as e:
        print(e)
    finally:
        p.wait(timeout=5)


with Pool(1) as p:
    _partial = partial(_render,
                       output_path=output_path,
                       list_length=len(dataset_files),
                       config_fname=config_fname,
                       number_of_images=number_of_images)
    p.map(_partial, enumerate(dataset_files), chunksize=1)


#_render((0, os.path.join(dataset_files_base, '2698c01.dat')), output_path, 1, config_fname, number_of_images=5)


# for idx, dataset_file in enumerate(dataset_files):
#     _render((idx, dataset_file), output_path, len(dataset_files), config_fname, number_of_images)