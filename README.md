# LegoBrickClassification
This project is inspired by Jacques Mattheij's blog entry: [Sorting 2 Metric Tons of Lego][1] who uses a Deep Learning approach to sort Lego bricks. Our first goal is to improve the software side, especially creating a generic dataset for the neural network training.
Due to the high number of bricks (10000+) and the related similarities, we plan a meaningful combination of categories and single bricks as classification task.

# Dataset Generation
To generate images from a single 3d model we use the Blender script `render_brick.py`. All 3d models come from a collection of [LDraw™][5] an open standard for LEGO CAD programs that allow the user to create virtual LEGO models. To improve the network training process we use rotation, scaling and translation of the brick and randomly insert a background image (indoor scene). For rotation all front perspectives are excluded in addition to a small range, e.g. 10 degree rotation in x direction is not permitted. For this reason, it is easier to identify the brick. A result is shown below. Established Blender with the [ImportLDraw][2] module you can run this script like: `blender -b -P render_brick.py -- -i='<path to .dat file>' -b='<path to background image>' -n=<number of images> -s='<output path>'`. To generate the full dataset adjust the parameters in `create_dataset.py`. This script executes the Blender script for all available 3d models. The background images descended from a very small subset from [Indoor Scene Recognition Dataset][3].

<img src="/preprocessing/examples/rendered_brick_noise.jpg" width="224">

# Classification

### Transfer Learning
For first tests, we use Transfer Learning with the aim to retrain existing network architectures like VGG16 or Xception trained on [ImageNet][3]. This allows a model creation with significantly reduced training data and time.
We simply cut the last layer(s) and retrain with our classes. `train_model.py` builds the train and test set from the generated images and retrains some well known networks.



### Todo
- [x] Dataset Generation: Find optimal parameters in `config.json` and use random brick colors
- [ ] (Re-)train different CNN architectures like VGG-16 and test performance for different settings (number of classes, images per class, training parameters). 
- [ ] Optimize Transfer Learning approach


[1]: https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/
[2]: https://github.com/TobyLobster/ImportLDraw
[3]: http://web.mit.edu/torralba/www/indoor.html
[4]: http://image-net.org/
[5]: http://www.ldraw.org/
