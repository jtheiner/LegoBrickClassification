import subprocess
from functools import partial
from multiprocessing.pool import Pool
import os
import sys
import pandas as pd
import wget
from pathlib import Path
import zipfile
import logging
from datetime import datetime

ldraw_fname = "complete.zip"
dataset_files_base = './ldraw/parts/'
dataset_path = './DATA/LEGO-brick-images/Parts_Alt_2.csv'
config_fname = 'augmentation.json'#'thumbnail.json'#'augmentation.json'
output_path = './DATA/LEGO-brick-images/'
url = "http://www.ldraw.org/library/updates/complete.zip"

if (Path(dataset_files_base).is_dir() ==False):
    wget.download(url,ldraw_fname )
    with zipfile.ZipFile(ldraw_fname,"r") as zip_ref:
        zip_ref.extractall("")
number_of_images = 2


df = pd.read_csv(dataset_path, encoding='utf-8', index_col=0)
dataset_files = [(os.path.join(dataset_files_base, str(df.loc[f]['Part_num']) + '.dat'), os.path.join(output_path, str(df.loc[f]['Directory'])), number_of_images //df.loc[f] ['Num_var']) for f in df.index]

dataset_files = dataset_files[:20]
print(dataset_files)

# todo: load background images in memory before rendering process begins, possible in blender?
# alternative: remove already used "background material" in blender
def _render(idx_fname, list_length: int, config_fname: str):
    #print(idx_fname[:20])
    index, dataset = idx_fname
    fname,output_dir_path,number_of_images =dataset
    part_id = os.path.splitext(os.path.basename(fname))[0]
    if os.path.exists(os.path.join(output_dir_path)):
        print('{} ({}/{}): already exists'.format(fname, index + 1, list_length))
        return
    print('{} ({}/{}): render'.format(fname, index + 1, list_length))
    command_path =""
    if sys.platform =='darwin':
        command_path ="/Applications/blender.app/Contents/MacOS/" # required for OSX
    render_script_path = os.path.join(os.path.dirname(__file__), 'dataset','blender', 'render.py')
    #os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    #logging.info(part_id)
    #logging.info(output_dir_path)
    command = command_path + 'blender -b -P ' + render_script_path + ' --' \
              + ' -i ' + fname \
              + ' -c ' + config_fname \
              + ' -s ' + os.path.join(output_dir_path) \
              + ' -n ' + str(number_of_images)
    print(command)
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    # except Exception as e:
    #     print(e)
    except subprocess.TimeoutExpired as e:
        print(e)
    finally:
        p.wait(timeout=30*number_of_images )
if __name__ == '__main__':   

    with Pool(processes=1) as p:
        print(p)
        start = datetime.now()
        _partial = partial(_render,
                           list_length=len(dataset_files),
                           config_fname=config_fname)
        p.map(_partial, enumerate(dataset_files), chunksize=1)
       # p.close()
       # p.join()
        print ("pool elapsed", datetime.now() - start)
        
     
        
       


#_render((0, os.path.join(dataset_files_base, '2698c01.dat')), output_path, 1, config_fname, number_of_images=5)


## for idx, dataset_file in enumerate(dataset_files):
##     _render((idx, dataset_file,), output_path, len(dataset_files), config_fname, number_of_images)
##
