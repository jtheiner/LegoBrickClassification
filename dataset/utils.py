import os
import logging

import numpy as np
import cv2
from skimage import img_as_float
import matplotlib.pyplot as plt


def plot_category_distribution(df, output_path, lower_limit=None):
    f = plt.figure(figsize=(12, 3))
    ax = f.add_subplot(111)
    if lower_limit:
        df['category'].value_counts()[:lower_limit].plot(kind='bar')
    else:
        df['category'].value_counts().plot(kind='bar')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.1, right=1.0, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
    ax.set(ylabel='count')
    ax.grid(which='both', axis='y')
    ax.set_yticks(np.arange(0, df['category'].value_counts().max() + 300, 100), minor=True)
    ax.grid(which='minor', alpha=0.2)

    distr_file = os.path.join(output_path, 'category_distribution.svg')
    os.makedirs(os.path.dirname(distr_file), exist_ok=True)
    logging.debug('saving category distribution to {}'.format(distr_file))
    f.tight_layout()
    plt.savefig(distr_file)
    plt.close()


def read_image(fname, grayscale=True, resize=None, as_float=None):
    """Loads and preprocess an image from file

    :param fname: path to image file
    :param grayscale: converts image to grayscale
    :param resize: resize to (width, height)
    :as_float: convert image to [0,1]
    :return: preprocessed image with values in [0,255]
    """
    img = cv2.imread(fname)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize:
        img = cv2.resize(img, resize)
    if as_float:
        img = img_as_float(img)
    return img


def plot_debug_images(src_label, tgt_label, mse, bin_mse, img_src, img_tgt, bin_src, bin_tgt, out_path):
    f = plt.figure(figsize=(6, 6))
    values = [src_label, tgt_label, round(mse, 8), round(bin_mse, 8)]
    title = f.suptitle('{}-{}: MSE raw: {}, SSIM: {}'.format(*values), fontsize=10)
    title.set_y(0.98)
    ax1 = f.add_subplot(221).imshow(img_src, cmap='gray')
    ax2 = f.add_subplot(222).imshow(img_tgt, cmap='gray')
    ax3 = f.add_subplot(223).imshow(bin_src, cmap='gray')
    ax4 = f.add_subplot(224).imshow(bin_tgt, cmap='gray')
    f.tight_layout()
    f.subplots_adjust(top=0.9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logging.debug('saving debug image: {}'.format(out_path))
    plt.savefig(out_path)
    plt.close()


def plot_sims(sims, output_path):
    f = plt.figure(figsize=(10, 5))
    ax1 = f.add_subplot(111)
    ax1.plot(sorted(sims, reverse=True))
    ax1.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax1.set_yticks(np.arange(0.5, 1.0, 0.025), minor=True)
    #ax1.set_xticks(np.arange(0, len(sims), 50))
    ax1.set_xticks(np.arange(0, len(sims), 5), minor=True)
    ax1.grid()
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    ax1.set_title('SSIM')
    f.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.debug('saving debug image for similarities: {}'.format(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_top_similar_images(similarities, output_path, labels, thumbnail_path, k=15, ascending=False):

    similarities = similarities
    sorted_indices = np.argsort(similarities)[::-1]
    if ascending:
        sorted_indices = sorted_indices[::-1]

    sorted_sims = similarities[sorted_indices]

    image_paths = [os.path.join(thumbnail_path, labels[i] + '_0.jpg') for i in sorted_indices]
    images = [read_image(f, grayscale=False, resize=(256, 256), as_float=True) for f in image_paths]

    f = plt.figure(figsize=(k * 2, 3))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
    for i in range(len(similarities)):
        ax = f.add_subplot(1, k, i + 1)
        ax.imshow(images[i], cmap='gray')
        title = '{}\n{}\n{}'.format(labels[i], round(sorted_sims[i], 8), i + 1)
        ax.set_title(title)
        plt.axis('off')
        if i + 1 == k:
            break

    logging.debug('saving plot_top_similar_images: {}'.format(output_path))
    plt.savefig(output_path)
    plt.close()