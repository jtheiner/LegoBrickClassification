#from fastai import *
from fastai.callbacks import SaveModelCallback
#from fastai.vision import *
from fastai.train import ClassificationInterpretation, DatasetType, load_learner
from fastai.vision import get_transforms, ImageList, cnn_learner, accuracy, jitter, open_image, learner
from torchvision import models as tv_models
from matplotlib import pyplot as plt
from pathlib import Path

base_path = Path('data', 'dataset-15')
dataset_path = base_path
img_size = 224
bs = 128
arch = tv_models.resnext50_32x4d
tfms = get_transforms(do_flip=True, flip_vert=True, max_warp=0.0, max_zoom=1.0)
data = (ImageList.from_folder(dataset_path)
        .split_by_folder(train='images', valid='testset-15-cropped')
        .label_from_folder()
        .transform(tfms)
        .databunch(bs=bs)
        .normalize()
        )
data.valid_dl = data.valid_dl.new(shuffle=True)

# plot one image with transformations
# example_img = open_image('/home/hoth/Desktop/lego-brick-recognition/data/datasets/train-15/images/3008/3008_0.jpg')
# example_img.apply_tfms(tfms[0], size=224).show(figsize=(10, 10))
# plt.show()

Path.mkdir(base_path / 'classification', exist_ok=True)
# view data
data.show_batch(rows=10, ds_type=DatasetType.Train)
plt.savefig(base_path / 'classification' / 'batch_example_train.svg')
data.show_batch(rows=10, ds_type=DatasetType.Valid)
plt.savefig(base_path / 'classification' / 'batch_example_valid.svg')

learn = cnn_learner(data, arch, pretrained=True, metrics=[accuracy])
# learn.unfreeze()

# learn.load(arch.__name__)

# learn.lr_find()
# learn.recorder.plot()
# plt.show()

cbs = [SaveModelCallback(learn, monitor='accuracy', name='best')]
learn.fit_one_cycle(5, max_lr=1e-3, callbacks=cbs)
learn.save(arch.__name__)


learn.recorder.plot_losses()
plt.savefig(base_path / 'classification' / 'loss.svg')

learn.show_results(rows=20)
plt.savefig(base_path / 'classification' / 'show_results.svg')

preds, y, losses = learn.get_preds(with_loss=True)
interpreter = ClassificationInterpretation(learn, preds, y, losses)
interpreter.plot_confusion_matrix(normalize=True, figsize=(8, 8))
plt.savefig(base_path / 'classification' / 'cm.svg')

interpreter.most_confused()

interpreter.plot_top_losses(10, figsize=(10, 10))
plt.savefig(base_path / 'classification' / 'top_losses.svg')
