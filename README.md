# LegoBrickClassification
This project is inspired by Jacques Mattheij's blog entry: [Sorting 2 Metric Tons of Lego][1] who uses a Deep Learning approach to sort Lego bricks. Our first goal is to improve the software side, especially creating a generic dataset for the neural network training.
Due to the high number of bricks (10000+) and the related similarities, we plan a meaningful combination of categories and single bricks as classification task.

## Dataset Generation
In order to generate images from a single 3d model we use Blender's Python interface and the [ImportLDraw][2] module. 
All 3d models come from a collection of [LDraw™][5] an open standard for LEGO CAD programs that allow the user to create virtual LEGO models. 
To improve the network training process we use rotation, scaling and translation of the brick and randomly insert a background image. 
The background images are taken from a very small subset of the [Indoor Scene Recognition Dataset][3].
For rotation all front perspectives are excluded in addition to a small range, e.g. 10 degree rotation in x direction
 is not permitted.  For this reason, it is easier to identify the brick.

<img src="/results/readme_examples/example1.jpg" width="224">
<img src="/results/readme_examples/example2.jpg" width="224">

### Usage

#### Single brick:
Render one image for a single brick:
```python
python create_dataset.py -t="part" -i=<path to .dat file>
```
Alternatively, you can run the corresponding blender script itself using:
```bash
blender -b -P render_brick.py -- -i=<.dat file> -n=<number of images> -b=<background images path> -s=<output directory>
```
Note, on MacOs you can run blender with `/Applications/Blender/blender.app/Contents/MacOS/blender -b -P ...`


#### Bricks for a given category:
Generate images for one category. A subfolder with images is created for each brick in the category. File structure:`
<output directory>/<category>/<part id>/<img id>.jpg` 
```python
python create_dataset.py -t="category" -i=<path to .dat files> -c=<category>
```

#### Bricks for all categories:
Generate images for all bricks in the directory. Each brick is assigned to its respective category.
```python
python create_dataset.py -t="full" -i=<path to .dat files> 
```


### Parameters:

 Argument | Optional  | Description |
|-------------| :-----: | -------------|
| -t  / --type | False    | Type for image generation: Choose "part", "category" or "full" |
| -i  / --input | False    | Directory of .dat files or path to file if "part" is selected |
| -b  / --background-images | True | Directory of images used as background. An image is randomly selected for each output image. |
| -n  / --images      |   True | Number of images to render for each brick. Default: 1|
| -c  / --category      |   True | Generate only images for a given category. Specify only if --type is "category" |
| -o / --out_dataset | True | Output directory for generated images. By default, "./results/dataset/" is used.|



### Parts Distribution 
Parts distribution over all categories which include more than 10 individual parts from `ldraw/parts/` directory.
<img src="/results/readme_examples/category_counts.svg">

[1]: https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/
[2]: https://github.com/TobyLobster/ImportLDraw
[3]: http://web.mit.edu/torralba/www/indoor.html
[4]: http://image-net.org/
[5]: http://www.ldraw.org/