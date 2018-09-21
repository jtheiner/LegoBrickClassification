# LegoBrickClassification
This project is inspired by Jacques Mattheij's blog entry: [Sorting 2 Metric Tons of Lego][1] who uses a Deep Learning approach to sort Lego bricks. Our first goal is to improve the software side, especially creating a generic dataset for the neural network training.
Due to the high number of bricks (10000+) and the related similarities, we plan a meaningful combination of categories and single bricks as classification task.

# Dataset Generation
To generate images from a single 3d model we use the Blender script `render_brick.py`. To improve the network training process we use rotation, scaling and translation of the brick and randomly insert a background image. A result is shown below. Established Blender with the [ImportLDraw][2] module you can run this script like this: `blender -b -P render_brick.py -- -i='<path to .dat file>' -b='<path to background images>' -n=<number of images> -s='<output path>'`. To generate the full dataset adjust the parameters in `create_dataset.py`. This script executes the Blender script for all available 3d models.

<img src="/preprocessing/examples/rendered_brick_noise.jpg" width="224">

### Todo
- [ ] Dataset Generation: Find optimal parameters in `config.json` and use random brick colors
- [ ] (Re-)train different CNN architectures like VGG-16 or ResNet and test performance. 


[1]: https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/
[2]: https://github.com/TobyLobster/ImportLDraw
