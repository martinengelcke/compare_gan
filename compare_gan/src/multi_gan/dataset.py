# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides access to Datasets and their parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import json
import random
from PIL import Image
import PIL
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("multigan_dataset_root",
                    "/tmp/datasets/multi_gan",
                    "Folder which contains all datasets.")

# Multi-MNIST configs.
MULTI_MNIST_CONFIGS = [
    "multi-mnist-3-uniform", "multi-mnist-3-triplet",
    "multi-mnist-3-uniform-rgb-occluded",
    "multi-mnist-3-uniform-rgb-occluded-cifar10"]


def unpack_clevr_image(image_data):
  """Returns an image and a label. 0-1 range."""
  image = tf.read_file(image_data)
  image = tf.image.decode_png(image, channels=3)
  image = tf.reshape(image, [1, 320, 480, 3])
  image = tf.image.resize_bilinear(image, size=(160, 240))
  image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, 128, 128))
  image = tf.cast(image, tf.float32) / 255.0
  dummy_label = tf.constant(value=0, dtype=tf.int32)
  return image, dummy_label


def load_clevr(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  return tf.data.Dataset.list_files(
    os.path.join(FLAGS.multigan_dataset_root,
                 "clevr/images/%s/*.png" % split_name)
    ).map(unpack_clevr_image, num_parallel_calls=num_threads)

def load_clevr_3to6(dataset_name, split_name, num_threads, buffer_size):
  """
  Based on code from ogroth.
  """
  del dataset_name
  # --- Get filenames ---
  num_objects_min = 3
  num_objects_max = 6
  data_root = os.path.join(FLAGS.multigan_dataset_root, "clevr")
  image_files = []
  image_annotations = []
  if split_name == 'train':  # Merge train and val splits
    for split in ['train', 'val']:
      # Load file names and annotations
      image_files.extend([
          os.path.join(data_root, 'images', split, fn) \
          for fn in sorted(os.listdir(os.path.join(data_root, 'images', split)))
      ])
      with open(os.path.join(data_root, 'scenes', 'CLEVR_%s_scenes.json' % split)) as fp:
        train_scenes = json.load(fp)['scenes']
      image_annotations.extend(train_scenes)
    # Filter by number of objects
    obj_cnt_list = list(
        zip(
            image_files,
            [len(scn['objects']) for scn in image_annotations]
        )
    )
    obj_cnt_filter_index = [
        t[1] >= num_objects_min and t[1] <= num_objects_max \
        for t in obj_cnt_list
    ]
    # Apply filter
    image_files = [image_files[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
    image_annotations = [image_annotations[idx] for idx, t in enumerate(obj_cnt_filter_index) if t]
    print(">>> Loaded %d images from split 'train+val'." % len(image_files))
  elif split_name == 'val':  # Load validation data only
    image_files.extend([
          os.path.join(data_root, 'images', 'val', fn) \
        for fn in sorted(os.listdir(os.path.join(data_root, 'images', 'test')))
    ])
    image_files = image_files[:100]
    print(">>> Loaded %d images from split 'val'." % len(image_files))
  elif split_name == 'test':  # Load test data only
    image_files.extend([
          os.path.join(data_root, 'images', 'test', fn) \
        for fn in sorted(os.listdir(os.path.join(data_root, 'images', 'test')))
    ])
    image_files = image_files[:100]
    print(">>> Loaded %d images from split 'test'." % len(image_files))
  else:
    raise ValueError("Invalid split!")
  # --- Create TF dataset ---
  def datamap(file_path):
    image = tf.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.reshape(image, [1, 320, 480, 3])
    image = tf.image.resize_bilinear(image, size=(128, 128))
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32) / 255.0
    dummy_label = tf.constant(value=0, dtype=tf.int32)
    return image, dummy_label
  return tf.data.Dataset.from_generator(
      lambda: image_files,
      output_types=tf.string,
    ).map(datamap, num_parallel_calls=num_threads)


def clevr_obc(data_root, split_name, num_threads, buffer_size):
  """
  Based on code from ogroth.
  """
  # --- Get filenames ---
  if split_name in ['train', 'val']:
      data_dir = '%s/train/' % data_root
      ordered = False
  else:
      data_dir = '%s/test/' % data_root
      ordered = True 
  dirs = []
  for d1 in os.listdir(data_dir):
      if d1[-4:]=='.png':
          dirs.append('%s/%s' % (data_dir, d1))
  # --- Create TF dataset ---
  def datamap(file_path):
    image = tf.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.reshape(image, [1, 96, 96, 3])
    image = tf.image.resize_bilinear(image, size=(128, 128))
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32) / 255.0
    dummy_label = tf.constant(value=0, dtype=tf.int32)
    return image, dummy_label
  return tf.data.Dataset.from_generator(
      lambda: dirs,
      output_types=tf.string,
    ).map(datamap, num_parallel_calls=num_threads)


def load_clevr_obc_5(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  data_root = os.path.join(FLAGS.multigan_dataset_root, "clevr_obc_5")
  return clevr_obc(data_root, split_name, num_threads, buffer_size)


def load_clevr_obc_5vbg(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  data_root = os.path.join(FLAGS.multigan_dataset_root, "clevr_obc_5vbg")
  return clevr_obc(data_root, split_name, num_threads, buffer_size)


def load_clevr_obc_3(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  data_root = os.path.join(FLAGS.multigan_dataset_root, "clevr_obc_3")
  return clevr_obc(data_root, split_name, num_threads, buffer_size)


def load_shapestacks5(dataset_name, split_name, num_threads, buffer_size):
  """
  Based on code from ogroth.
  """
  del dataset_name
  # --- Get filenames ---
  data_root = os.path.join(FLAGS.multigan_dataset_root, "shapestacks")
  data_dir = '%s/recordings/' % data_root
  dir = [data_dir+f for f in os.listdir(data_dir) if f[0]=='e']
  dirs = []
  for i in range(len(dir)):
      dir_name = dir[i]
      if os.path.exists(dir_name) and int(dir_name.split('-h=')[-1][0])<=5:
          name = [dir_name+'/'+f for f in os.listdir(dir_name) if f[-4:]=='.png' and (f[-13]=='_' or int(f[-12])<2)]
          if len(name) > 0:
              ## Look at the h option and load only 2...
              dirs.extend(name)
  # --- Create TF dataset ---
  def datamap(file_path):
    image = tf.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.reshape(image, [1, 224, 224, 3])
    image = tf.image.resize_bilinear(image, size=(128, 128))
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32) / 255.0
    dummy_label = tf.constant(value=0, dtype=tf.int32)
    return image, dummy_label
  return tf.data.Dataset.from_generator(
      lambda: dirs,
      output_types=tf.string,
    ).map(datamap, num_parallel_calls=num_threads)


def load_bowl2balls(data_root, train=True):
  # gather list of video directories
  if train:
    data_dir = os.path.join(data_root, 'train')
  else:
    data_dir = os.path.join(data_root, 'test')
  video_dirs = []
  for d in os.listdir(data_dir):
    video_dirs.append(os.path.join(data_dir, d, 'render'))
  # gather filenames from directories
  for video_dir in video_dirs:
    img_files = []
    _seq_len = len([f for f in os.listdir(video_dir) if f[-4:]=='.jpg'])
    for idx in range(_seq_len):
      img_file = f'{video_dir}/{3*idx:04d}.jpg'
      img_files.append(img_file)
  # build tf.dataset from filenames
  dataset = tf.dataset.Dataset.from_generator(lambda: img_files, output_types=tf.string)
  # map preprocessing function over filenames
  def _preprocess(file_path):
    image = tf.read_file(file_path)
    image = tf.image.decode_jpg(image, channels=3)
    image = tf.expand_dims(image, axis=0)  # add batch dimension
    image = tf.image.resize_bilinear(image, size=(128, 128))
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32) / 255.0
    dummy_label = tf.constant(value=0, dtype=tf.int32)
    return image, dummy_label
  return dataset.map(_preprocess, num_parallel_calls=num_threads)


def load_realtraffic(data_root):
  video_dirs = [os.path.join(data_root, f) for f in os.listdir(data_root) if f[0]=='f']
  img_files = []
  for video_dir in video_dirs:
    img_seq = sorted([video_dir+'/'+f for f in os.listdir(video_dir) if f[-5:]=='.jpeg'])
    for idx in range(len(img_seq)-3):
      img_files.append(img_seq[idx:idx+2:2])
  # build tf.dataset from filenames
  dataset = tf.dataset.Dataset.from_generator(lambda: img_files, output_types=tf.string)
  # map preprocessing function over filenames
  def _preprocess(file_path):
    image = tf.read_file(file_path)
    image = tf.image.decode_jpg(image, channels=3)
    image = tf.expand_dims(image, axis=0)  # add batch dimension
    image = tf.image.resize_bilinear(image, size=(128, 128))
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32) / 255.0
    dummy_label = tf.constant(value=0, dtype=tf.int32)
    return image, dummy_label
  return dataset.map(_preprocess, num_parallel_calls=num_threads)


def unpack_multi_mnist_image(split_name, k, rgb, image_data):
  """Returns an image and a label in [0, 1] range."""
  c_dim = 3 if rgb else 1

  value = tf.parse_single_example(
      image_data,
      features={"%s/image" % split_name: tf.FixedLenFeature([], tf.string),
                "%s/label" % split_name: tf.FixedLenFeature([k], tf.int64)})

  image = tf.decode_raw(value["%s/image" % split_name], tf.float32)
  image = tf.reshape(image, [64, 64, c_dim])
  image = image / 255.0 if rgb else image
  label = tf.cast(value["%s/label" % split_name], tf.int32)
  return image, label


def load_multi_mnist(dataset_name, split_name, num_threads, buffer_size):
  k = int(dataset_name.split("-")[2])
  rgb = "rgb" in dataset_name
  unpack = functools.partial(unpack_multi_mnist_image, split_name, k, rgb)
  filename = os.path.join(FLAGS.multigan_dataset_root,
                          "%s-%s.tfrecords" % (dataset_name, split_name))

  return tf.data.TFRecordDataset(
      filename,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack, num_parallel_calls=num_threads)


def get_dataset_params():
  """Returns a dictionary containing dicts with hyper params for datasets."""

  params = {
      "clevr": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr",
          "eval_test_samples": 10000
      },
      "clevr-3-to-6": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr-3-to-6",
          "eval_test_samples": 10000
      },
      "clevr-obc-5": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr-obc-5",
          "eval_test_samples": 10000
      },
      "clevr-obc-5vbg": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr-obc-5vbg",
          "eval_test_samples": 10000
      },
      "clevr-obc-3": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr-obc-3",
          "eval_test_samples": 10000
      },
      "shapestacks5": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "shapestacks5",
          "eval_test_samples": 10000
      },
      "bowl2balls": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "bowl2balls",
          "eval_test_samples": 10000
      },
      "realtraffic": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "realtraffic",
          "eval_test_samples": 10000
      },
  }

  # Add multi-mnist configs.
  for dataset_name in MULTI_MNIST_CONFIGS:
    c_dim = 3 if "rgb" in dataset_name else 1
    params.update({
        dataset_name: {
            "input_height": 64,
            "input_width": 64,
            "output_height": 64,
            "output_width": 64,
            "c_dim": c_dim,
            "dataset_name": dataset_name,
            "eval_test_samples": 10000
        }
    })

  return params


def get_datasets():
  """Returns a dict containing methods to load specific dataset."""

  datasets = {
      "clevr": load_clevr,
      "clevr-3-to-6": load_clevr_3to6,
      "clevr-obc-5": load_clevr_obc_5,
      "clevr-obc-5vbg": load_clevr_obc_5vbg,
      "clevr-obc-3": load_clevr_obc_3,
      "shapestacks5": load_shapestacks5,
      "bowl2balls": load_bowl2balls,
      "realtraffic": load_realtraffic,
  }

  # Add multi-mnist configs.
  for dataset_name in MULTI_MNIST_CONFIGS:
    datasets[dataset_name] = load_multi_mnist

  return datasets
