# A Case for Object Compositionality in Deep Generative Models of Images

![clevr-generated](illustrations/clevr_generated.png)

This is the code repository complementing the
["A Case for Object Compositionality in Deep Generative Models of Images"](https://arxiv.org/pdf/1810.10340.pdf).

## Datasets

The following provides an overview of the datasets that were used. Corresponding
.tfrecords files for all custom datasets are available [here](https://goo.gl/Eub81x).

### Multi-MNIST

In these dataset each image consists of 3 (potentially overlapping) MNIST digits.
Digits are obtained from the original dataset (using train/test/valid respectively),
re-scaled and randomly placed in the image. A fixed-offset from the border ensures
that digits appear entirely in the image.

#### Independent

All digits 0-9 have an equal chance of appearing in the image. This roughly
results in a uniform distribution over all 3-tuples of digits in the images. In
practice we generate multi-mnist images in batches, meaning that we repeatedly
select a batch of 100 MNIST digits that are then used to generate 100
multi-MNIST images. Each digit in the batch has an equal chance of appearing in
the multi-MNIST image, such that we roughly achieve uniformity. Digits are drawn
on top of each other and clipped in 0.0-1.0 range.

#### Triplet

All 3-tuples of digits that are triplets, eg. (2, 2, 2) have an equal chance of
appearing in the image. As before we use batching, i.e. from a batch of images a
digit id is selected, after which 3 digits of that type are randomly sampled
from the batch. Digits are drawn on top of each other and clipped in 0.0-1.0
range.

#### RGB Occluded

Digits are sampled [uniformly](#uniform) and colored either
<span style="color:red">red</span>, <span style="color:green">green</span> or
<span style="color:blue">blue</span>, such that exactly one of each color
appears in a Multi-MNIST image. Digits are drawn one by one into the canvas
without blending colors, such that overlapping digits occlude one another.

### CIFAR10 + RGB MM

Draws digits from [rgb occluded](#rgb-occluded) images on top of a randomly
sampled CIFAR10 image (resized to 64 x 64 using bilinear interpolation).

### CLEVR

The CLEVR dataset can be downloaded from
[here](https://cs.stanford.edu/people/jcjohns/clevr/). We use the "main" dataset
containing 70.000 train images to train our model, and 10.000 of the test images
to compute FID scores.

During data loading we resize the raw images (320 x 480) to (160, 240) using
bilinear interpolation, and take a center crop to obtain (128 x 128) images.
These are then normalized to 0.0-1.0 before being fed to the network.

## Training a model

### Setup conda environment
```
conda env create -f compare_gan/src/multi_gan/environment.yml
conda activate multigan_env
python -m pip install -e . --user
# when you are done
conda deactivate
```

### Setup data
For debug experiment, download data from [here](https://drive.google.com/drive/folders/1K7IeRbwSKA8Ho6SZwNMeK2lwsQsUuiIG) into `multigan_data`.

For CLEVR experiment, download data from [here](https://cs.stanford.edu/people/jcjohns/clevr/) and put the `images` folder into `multigan_data/clevr`.

### Setup a task
Debug model:
```
python compare_gan/bin/compare_gan_generate_tasks --workdir workdir/debug --experiment multi_gan-debug
```
MultiGAN on CLEVR with k=5 components:
```
python compare_gan/bin/compare_gan_generate_tasks --workdir workdir/clevr-k5 --experiment multi_gan-clevr-k5
```

### Run the task
Debug model:
```
python compare_gan/bin/compare_gan_run_one_task --workdir workdir/debug --multigan_dataset_root multigan_data
```
MultiGAN on CLEVR with k=5 components:
```
python compare_gan/bin/compare_gan_run_one_task --workdir workdir/clevr-k5 --multigan_dataset_root multigan_data
```

## Train with different configs

Edit `compare_pan/src/multi_gan/run_one_task` and add a new experiment to `GetMetaTasks(experiment_name)`.
Now you can create the experiment as before with:
```
python compare_gan/bin/multi_gan_generate_tasks --workdir workdir/YOUR-MULTI_GAN-EXP --experiment your-multi_gan-exp
```
