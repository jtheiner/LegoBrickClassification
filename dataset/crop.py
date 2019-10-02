from pathlib import Path
import os
import xml.etree.ElementTree as et
from PIL import Image
from multiprocessing import Pool

images_original_path = Path('resources/testset-15-original')
annotations_path = Path('resources/testset-15-annotations')
images_cropped_path = Path('resources/testset-15-cropped')

images_original_paths = [Path(os.path.join(dp, f)) for dp, dn, filenames in os.walk(images_original_path) for f in filenames if os.path.splitext(f)[1] == '.JPG']
annotations_paths = [Path(os.path.join(annotations_path, f)) for f in os.listdir(annotations_path) if f.endswith('.xml')]
annotations = {a.stem: a for a in annotations_paths}

target_size = (224, 224)

def crop_by_annotation(image_original_path, crop_border=200):

    if image_original_path.stem in annotations:
        annotation = et.parse(str(annotations[image_original_path.stem])).getroot()
        bounding_box = annotation.find('object').find('bndbox')
        xmin = int(bounding_box.find('xmin').text) - crop_border
        ymin = int(bounding_box.find('ymin').text) - crop_border
        xmax = int(bounding_box.find('xmax').text) + crop_border
        ymax = int(bounding_box.find('ymax').text) + crop_border

        img_original = Image.open(image_original_path)
        img_cropped = img_original.crop((xmin, ymin, xmax, ymax))  # left, upper, right, lower
        img_resized = img_cropped.resize(target_size)  # resize to target size

        part_id = image_original_path.parent.name
        Path.mkdir(images_cropped_path / part_id, exist_ok=True, parents=True)
        fout = images_cropped_path / part_id / image_original_path.name
        img_resized.save(fout)
        return

with Pool(8) as p:
    p.map(crop_by_annotation, images_original_paths)


# image_original_path = images_original_paths[0]
# crop_border=20