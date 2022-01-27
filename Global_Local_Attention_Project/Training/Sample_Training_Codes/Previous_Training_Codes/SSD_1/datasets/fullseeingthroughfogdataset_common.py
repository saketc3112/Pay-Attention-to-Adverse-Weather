# Copyright 2015 Paul Balanca. All Rights Reserved.
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
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim


def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # change
    keys_to_features= {
        #'key': tf.VarLenFeature(dtype=tf.int64),
        #'name': tf.FixedLenFeature((), tf.string),
        'image/cam_stereo_left_lut': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='png'),
        'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/angle': tf.VarLenFeature(dtype=tf.float32),
        'image/object/truncation': tf.VarLenFeature(dtype=tf.float32),
        'image/object/occlusion': tf.VarLenFeature(dtype=tf.int64),
        'image/object/object/bbox3d/height': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/width': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/length': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/x': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/y': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/z': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox3d/alpha3d': tf.VarLenFeature(dtype=tf.float32),

        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        #'image/object/bbox/label': tf.FixedLenFeature([1], tf.int64),
        #'image/key': tf.FixedLenFeature((), tf.string),
        'image/object/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/shape/cam_stereo_left_lut': tf.FixedLenFeature([3], tf.int64),
        #'lidar/point_key': tf.VarLenFeature(dtype=tf.float32),
        #'lidar/shape': tf.VarLenFeature(dtype=tf.int64),
        #'gated/key': tf.FixedLenFeature((), tf.string),
        #'gated/shape': tf.VarLenFeature(dtype=tf.int64),
    }
    # change
    items_to_handlers = {
        #'key': slim.tfexample_decoder.Tensor('key'),
        #'name': slim.tfexample_decoder.Tensor('name'),
        'image': slim.tfexample_decoder.Image(image_key='image/cam_stereo_left_lut', format_key='image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape/cam_stereo_left_lut'),
        #'class/text': slim.tfexample_decoder.Tensor('image/object/class/text'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        #'object/difficult': slim.tfexample_decoder.Tensor('image/object/difficult'),
        #'object/truncated': slim.tfexample_decoder.Tensor('image/object/truncation'),
        #'bbox/angle': slim.tfexample_decoder.Tensor('image/object/bbox/angle'),
        #'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        #'object/truncation': slim.tfexample_decoder.Tensor('image/object/truncation'),
        #'object/occlusion': slim.tfexample_decoder.Tensor('image/object/occlusion'),
        #'object/bbox3d': slim.tfexample_decoder.BoundingBox(
        #    ['height', 'width', 'length', 'x','y','z'],'image/object/object/bbox3d/'),
        #'bbox3d/alpha3d': slim.tfexample_decoder.Image('image/object/object/bbox3d/alpha3d'),
        
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    #if dataset_utils.has_labels(dataset_dir):
    #    labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
