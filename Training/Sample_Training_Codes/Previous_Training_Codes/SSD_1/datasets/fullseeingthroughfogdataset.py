# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data for the SeeingThroughFog Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import fullseeingthroughfogdataset_common

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
# change

#FILE_PATTERN = '%s_%06d.swedentfrecord'
FILE_PATTERN = '*.swedentfrecord'

# change
SPLITS_TO_SIZES = {
    'train_clear_day': 2183,
    'test_clear_day': 200,
    'test_clear_night': 877,
}
# change
ITEMS_TO_DESCRIPTIONS = {
    'image_data': 'A color image of varying height and width.',
    'gated_data': 'Gated camera images of varying height and width.',
    'lidar_data': 'Lidar Data .bin files.',
    'image_shape': 'Shape of the image',
    'lidar_shape': 'Shape of the Lidar data',
    'gated_shape': 'Shape of the Gated camera image',
    'label': 'Common labels for image, gated, and lidar data',
    'name': 'Entry ID (Files name used for training)',
    'total_id': 'Total ID',
}

# 0 Pedestrian: 'Pedestrian', 'Pedestrian_is_group', 'person'
# 1 Truck: LargeVehicle', 'train', 'LargeVehicle_is_group', 
# 2 Car: 'PassengerCar_is_group', 'Vehicle_is_group', 'PassengerCar', 'Vehicle', 
# 3 Cyclist: 'RidableVehicle_is_group', 'RidableVehicle' 
# 4 DontCare: 'Obstacle', 'DontCare', 

# change
NUM_CLASSES = 5
#NUM_CLASSES = 6


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

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
    # change
    # Allowing None in the signature so that dataset_factory can use the default.
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
        

    if not file_pattern:
        file_pattern = FILE_PATTERN
    return fullseeingthroughfogdataset_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)