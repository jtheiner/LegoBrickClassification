# Lego Brick Recognition
<img align="right" width="128" height="128" src="/examples/rendered_brick_noise.jpg">
<img align="right" width="128" height="128" src="/examples/rendered_brick2_noise.jpg">

This project is inspired by Jacques Mattheij's blog entry: [Sorting 2 Metric Tons of Lego][1] who uses a Deep Learning approach to sort Lego bricks. Our first goal is to improve the software side, especially creating a generic dataset for the neural network training.
Due to the high number of bricks (10000+) and the related similarities, we plan a meaningful combination of categories and single bricks as classification task.

We would be happy if you would contribute to this project! We could then provide all necessary resource files, e.g. real test images or trained models.

## Table of Contents
1. Dataset Generation
    - Render a Single Brick
    - Data Analysis (Brick Similarity) and Preprocessing
    - Render Images
2. Classification
    - Preliminaries
    - Training
    - Evaluation
    - Inference
3. Requirements / Installation


## Dataset Generation
In order to generate images from a single 3d model we use Blender's Python interface and the [ImportLDraw][2] module.
All 3d models come from a collection of [LDraw™][3] an open standard for LEGO CAD programs that allow the user to create virtual LEGO models. 
Make sure you have successfully installed the module and set the ldraw path correctly.


### Render a Single Brick
Control of the blender script as stand-alone:  
`blender -b -P dataset/blender/render.py -- -i 9.dat --save ./ --images_per_brick 1 --config thumbnail-example.json`  
Hint for Mac users to access blender: `/Applications/Blender/blender.app/Contents/MacOS/blender -b -P ...`

### Data Analysis (Brick Similarity) and Preprocessing
To view the data for the first time, we generate thumbnails for each 3d file with fixed lightning conditions, background, camera settings, etc..
Furthermore, we extract some metadata such as category and label.
The category distribution of all available bricks is shown below.

<img src="/examples/category_distribution.svg">

To generate thumbnails i.e. for a the category 'Technic' run:
```python dataset/generate_thumbnails.py --dataset ./ldraw/parts/ --category Technic --output ./technic```

It is known that some parts have both different identifiers and are very similar to each other. For instance, a part differs only by small imprint on the front.
The question is how to deal with these two cases. For the first one it is essential to identify these visual equal parts that only differ in the part number.
The second case can be ignored if all parts are to be distinguished. However, it can be useful to consider bricks with the same shape as one class, which reduces the number of classes and the associated classification complexity.


Based on the generated thumbnails, a brick similarity can be calculated by comparing two images using the structural 
similarity image metric (SSIM).
By setting thresholds, we can identify identical parts as well as shape-similar parts.
In future, we plan further experiments and improvements e.g. by measuring the shape similarity.

The 10 most similar bricks with the part id '6549' (included itself) of a subset of all 'Technic' parts:

<img src="/examples/6538b-top_images_ssim.png">
Similarities to all other parts:

<img width="600" src="/examples/6538b.svg">

When the script is finished with creating thumbnails, a final CSV is written that contains lists of IDs for identical parts and shape-like parts.

| id | label | category | thumbnail_rendered | identical | shape_identical |
|---------|----------------------------------|----------|--------------------|-----------|---------------------------------------------------------------------|
| 6549 | Technic 4 Speed Shift Lever | Technic | True | 6549 | 2699 3705c01 3737c01 6549 73485 |
| 2698c01 | Technic Action Figure (Complete) | Technic | True | 2698c01 | 13670 24316 2698c01 3705 370526 4211815 4519 6587 87083 99008 u9011 |
| ... | ... | ... | ... | ... | ... |

### Render Images
- render images 2000 for each id
- image size: 224x224x3
- brick augmentation:
    - 38 official brick colors
    - rotation: in x direction due to no standardized origin: 0, 90, 180, 270
    - scaling: normalized one dimension and size variation via zooming factor  
    [ ] surface texture and reflexion
- setting background:
    - random images from IndoorCVPR09 dataset vs. noise
    - camera position: random on the surface on the upper half of a sphere
    - exposure: random spot on the sphere surface, radius and energy fixed  
    [ ] varying exposure
    

Script to render all images:
`python dataset/generate_dataset.py`  
TODO: Parameter via argparse

## Classification
In a first experiment, we focus on a small subset in order to test the capability of the synthetically generated 
training material. For this reason we manually selected 15 classes and created a real-world test set.
At this point, the knowledge of the identical and form-identical parts is irrelevant and will be considered searately
 in future.
    
### Training
- Common fastai image classification workflow as baseline
- Fine-tuning ResNeXt50 (and ResNet34), 5 epochs, fit_one_cycle 1e-3, batch size 42 (128)

### Evaluation
Evaluate on real-world testset!
- [x] created for 15 bricks. 20 images for each
- [x] crop and resize images manually
- [ ] train object detection model with squared bounding boxes

Results not satisfactory to some extent! The Difference between the synthetic generated training set and real-world validation set is to high!  
Possible Reasons (further investigation needed):
- brick size differ
- lighting conditions differ
- resolution differ
- background is relevant for generalization
- viewing angle differ

<img width="500" src="/examples/cm_resnext50_freezed.svg">

<img width="350" src="/examples/loss_resnext50_freezed.svg">

### Big TODOs
- reduce generalization gap by improving rendering process
- multi-instance training i.e. stack three images to prevent confusion in classification
- object detection model for preprocessing: crop squared images from video stream
- inference script for one-camera setting (object detection, ...) and multi-camera setting

## Requirements / Installation
- Blender 2.79 + [ImportLDraw][2]
- `conda env create -f environment.yml`


[1]: https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/
[2]: https://github.com/TobyLobster/ImportLDraw
[3]: http://www.ldraw.org/
