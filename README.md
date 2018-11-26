# LegoBrickClassification
This project is inspired by Jacques Mattheij's blog entry: [Sorting 2 Metric Tons of Lego][1] who uses a Deep Learning approach to sort Lego bricks. Our first goal is to improve the software side, especially creating a generic dataset for the neural network training.
Due to the high number of bricks (10000+) and the related similarities, we plan a meaningful combination of categories and single bricks as classification task.

# Dataset Generation
To generate images from a single 3d model we use the Blender script `render_brick.py`. All 3d models come from a collection of [LDraw™][5] an open standard for LEGO CAD programs that allow the user to create virtual LEGO models. To improve the network training process we use rotation, scaling and translation of the brick and randomly insert a background image (indoor scene). For rotation all front perspectives are excluded in addition to a small range, e.g. 10 degree rotation in x direction is not permitted. For this reason, it is easier to identify the brick. A result is shown below. Established Blender with the [ImportLDraw][2] module you can run this script like: `blender -b -P render_brick.py -- -i='<path to .dat file>' -b='<path to background image>' -n=<number of images> -s='<output path>'`. To generate the full datasets execute `create_dataset.py` with specified parameters. By default, this script generates a dataset for the class "Brick" with 100 images for each class. In addition parts with a new part number and currently unnecessary categories like Minifig or Duplo will be ignored. The background images descended from a very small subset from [Indoor Scene Recognition Dataset][3].

<img src="/examples/rendered_brick_noise.jpg" width="224">
 
##### Parts distribution over all categories which include more than 10 individual parts:
<img src="/examples/category_counts.svg">


# Classification

### Transfer Learning
For first tests, we use Transfer Learning with the aim to retrain existing network architectures like VGG19 trained on [ImageNet][3]. This allows a model creation with significantly reduced training data and time.
We simply cut the last layer(s) and retrain with our classes. `train_model.py -d='image dataset directory'` builds the train and test set from the generated images and retrains the VGG19. Selected 45 partly similar classes and retrained on 100 instances per class, the accuracy is around 80%, but has the capability for improvement.

### Classes
Due to the high number of bricks and limited hardware ressources of an sorting machine it is useful to assign bricks a category. These categories are extracted from the brick label of the 3d file.

#### Variant 1
Train a CNN to assign each brick a category. It is handy to group underrepresented categories.

#### Variant 2
Train a CNN to assign each brick its label dependend on the size of the category.


### Todo
- [x] Dataset Generation: Find optimal parameters in `config.json` and use random brick colors
- [ ] Fix color change for nested 3d objects
- [ ] (Re-)train different CNN architectures like VGG-19 and test performance for different settings (number of classes, images per class, training parameters). 
- [ ] Optimize Transfer Learning approach
- [ ] Build a validation dataset of real world images (label part id and category manually)
- [ ] Train an AlexNet from scratch with a limited number of classes


[1]: https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/
[2]: https://github.com/TobyLobster/ImportLDraw
[3]: http://web.mit.edu/torralba/www/indoor.html
[4]: http://image-net.org/
[5]: http://www.ldraw.org/
