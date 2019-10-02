import argparse
import os
import subprocess
import logging
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import pandas as pd
import numpy as np
from skimage.measure import compare_ssim as ssim
import utils


def image_similarity(src, tgt):
    """Calculates an image similarity

    :param src: source image
    :param tgt: target image
    :return: similarity value in range [0,1]
    """
    return ssim(src, tgt, dynamic_range=tgt.max() - tgt.min())
    # return np.linalg.norm(tgt - src)  # mse


def create_part_category_list(dat_directory: str) -> pd.DataFrame:
    """
    Extract IDs, labels and categories from .dat files.
    Stores content to csv file

    :param dat_directory: path to .dat files
    :return: pandas dataframe with columns [id, label, category]
    """

    # read parts directory containing all .dat files
    files = [f for f in os.listdir(dat_directory) if f.endswith('.dat')]

    parts = []
    for filename in files:
        part_number = filename[:-4]
        with open(os.path.join(dat_directory, filename), 'r') as f:
            first_line = f.readline()
            if '~Moved to' in first_line:
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

    df = pd.DataFrame().from_records(parts, columns=['id', 'label', 'category'], index='id')
    df = df.sort_values(by=['category', 'label'])
    return df


def _create_thumbnail(idx_fname: Tuple, dataset_in_path: str, output_path: str, list_length):
    index, fname = idx_fname
    part_id = fname.split('.')[0]
    if os.path.isfile(os.path.join(output_path, part_id + '_0.jpg')):
        logging.debug('{} ({}/{}): already exists'.format(fname, index + 1, list_length))
        return
    logging.info('{} ({}/{}): render'.format(fname, index + 1, list_length))
    render_script_path = os.path.join(os.path.dirname(__file__), 'blender', 'render.py')
    thumbnail_config = 'thumbnail.json'
    input_file_path = os.path.join(dataset_in_path, fname)
    command = 'blender -b -P ' + render_script_path + ' --' \
              + ' -i ' + input_file_path \
              + ' -c ' + thumbnail_config \
              + ' -s ' + output_path
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        logging.error('stopped creating thumbnail for {}: {}'.format(fname, e))
    except subprocess.TimeoutExpired as e:
        logging.error('stopped creating thumbnail for {}: {}'.format(fname, e))
    finally:
        p.wait(timeout=5)


def create_thumbnails(files: List[str], dataset_in_path: str, output_path: str):
    with Pool(4) as p:
        _partial = partial(_create_thumbnail,
                           dataset_in_path=dataset_in_path,
                           output_path=output_path,
                           list_length=len(files))
        p.map(_partial, enumerate(files), chunksize=1)


def is_rendered(fnames: List[str], output_path: str):
    """Checks whether a file exists

    :param fnames: list of part id's
    :param output_path: base path for the file names
    :return:
    """
    rendered_images = {}
    for f in fnames:
        file_path = os.path.join(output_path, f + '_0.jpg')
        if os.path.isfile(file_path):
            rendered_images[f] = True
        else:
            rendered_images[f] = False
    return rendered_images


def identical_parts(index: int, sim_matrix: np.array, part_ids: List, thres=None) -> List[str]:
    """Returns a list containing identical parts given an index and a similarity matrix

    :param index: respective index of a  part id
    :param sim_matrix: similarity matrix
    :param part_ids: mapping for all part id's
    :param thres: similarity threshold
    :return: list of part id's
    """
    sims_binary = sim_matrix[index]
    x = np.ma.masked_greater_equal(sims_binary, thres).mask
    indices = np.where(x)[0]  # select the respective indices
    parts = [part_ids[k] for k in indices]  # list of part ids
    parts.sort()
    if parts is None or len(parts) == 0:
        raise ValueError
    return parts


def get_similarities(images: List[np.array], output_path, sims_cache_file=None, debug=None) -> np.array:
    """Calculates a similarity matrix.
    Each entry represents the structural similarity index between two images
    See: https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    :param images: list of images
    :param sims_cache_file: filename to store the calculated matrices (append .npz)
    :return: similarity matrix
    """

    # cache from file if already calculated
    if sims_cache_file and os.path.isfile(sims_cache_file):
        logging.info('loading similarity matrices from file')
        z = np.load(sims_cache_file)
        sim_matrix = z['sim_matrix']
        if sim_matrix.shape == (len(images), len(images)):
            return sim_matrix

    # create new similarity matrix
    sim_matrix = np.empty((len(images), len(images)))
    # todo: optimize calculation
    for i, img_src in enumerate(images):
        logging.info('calculating similarities for part {}/{}'.format(i + 1, len(images)))
        for j, img_tgt in enumerate(images):

            # image similarity based on mse
            sim_matrix[i, j] = image_similarity(img_src, img_tgt)

        if debug:
            # plot all similarities for each brick
            debug_path = os.path.join(output_path, 'debug', df.index.values[i])
            fname = df.index.values[j] + '.svg'
            if os.path.isfile(os.path.join(debug_path, fname)):
                logging.debug(os.path.join(debug_path, fname) + ' already exists')
                continue
            utils.plot_sims(sim_matrix[i], os.path.join(debug_path, fname))

            # plot thumbnails and similarities for the most similar bricks
            fname_top = df.index.values[j] + '-top_images_ssim.jpg'
            if os.path.isfile(os.path.join(debug_path, fname_top)):
                logging.debug(os.path.join(debug_path, fname_top) + ' already exists')
                continue
            utils.plot_top_similar_images(similarities=sim_matrix[i],
                                          output_path=os.path.join(debug_path, fname_top),
                                          labels=df.index.values,
                                          thumbnail_path=thumbnail_output)

    logging.info('saving similarity matrix to file: {}'.format(sims_cache_file))
    np.savez(sims_cache_file, sim_matrix=sim_matrix)

    return sim_matrix


def parse_args(parser):
    parser.add_argument('-d', '--dataset', type=str, required=False,
                        default='resources/parts/complete-190827/ldraw/parts/')
    parser.add_argument('-c', '--category', type=str, required=False,
                        default='Brick', help='render only for a specific category')
    parser.add_argument('-o', '--output', type=str,
                        default='data/datasets/brick', help='output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity mode, i.e. generate debug images')
    parser.add_argument('-l', '--limit', type=int, help='limit the number of parts')
    parser.add_argument('-b', '--bricks', type=str, nargs='+', required=False, help='set of bricks to render')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args(argparse.ArgumentParser())

    csv_output = os.path.join(args.output, 'dataset.csv')
    thumbnail_output = os.path.join(args.output, 'thumbnails')
    top_k = args.limit  # for debug purpose only: select only a subset

    # initialize logger
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    if args.verbose:
        log_file = os.path.join(args.output, 'thumbnail_generator.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w', format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, filemode='w', format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

    # collect dataset information - category selection
    df = create_part_category_list(args.dataset)

    # plot category distribution
    logging.debug(df.category.value_counts())
    utils.plot_category_distribution(df, args.output, lower_limit=30)
    if args.bricks:
        logging.info('ignoring category argument if specified, using list of bricks instead: {}'.format(args.bricks))
        df = df.loc[args.bricks]
        logging.info('selected {} bricks successfully'.format(len(df.index)))

    # select category
    if args.category and not args.bricks:
        df = df.loc[df['category'] == args.category]

    if args.limit:
        df = df.head(top_k)

    # create one thumbnail image for each available part
    create_thumbnails([p + '.dat' for p in df.index.values], args.dataset, thumbnail_output)

    # are thumbnails successfully rendered? check only for file exist
    rendered_images = is_rendered(df.index.values, thumbnail_output)

    df['thumbnail_rendered'] = rendered_images.values()

    # select only available
    df_other = df.loc[~df['thumbnail_rendered']]
    df = df.loc[df['thumbnail_rendered']]

    # load all images to measure the similarities
    image_paths = [os.path.join(thumbnail_output, f + '_0.jpg') for f in df.index.values]
    images = [utils.read_image(fname, grayscale=True, resize=(128, 128), as_float=True) for fname in image_paths]

    # calculate the image similarities
    distances_cache_file = os.path.join(args.output, 'similarities.npz')

    sim_matrix = get_similarities(images, args.output, distances_cache_file, debug=args.verbose)

    # append a column that contains a list of all equal parts
    df['identical'] = pd.Series(dtype='str')
    for i in range(len(df.index)):
        i_parts = identical_parts(i, sim_matrix=sim_matrix, part_ids=df.index.values, thres=0.98)
        df.at[df.index.values[i], 'identical'] = ' '.join(i_parts)
    # append a column that contains a list of all shape identical parts
    df['shape_identical_count'] = pd.Series(dtype='int')
    df['shape_identical'] = pd.Series(dtype='str')
    for i in range(len(df.index)):
        i_parts = identical_parts(i, sim_matrix=sim_matrix, part_ids=df.index.values, thres=0.90)
        df.at[df.index.values[i], 'shape_identical'] = ' '.join(i_parts)
        df.at[df.index.values[i], 'shape_identical_count'] = len(i_parts)

    # build and save the final dataset
    df = pd.concat([df, df_other], sort=False)
    df['shape_identical_count'] = df['shape_identical_count'].astype(int)

    df.to_csv(csv_output)
    logging.info('finished')
