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
"""Contains the definition for inception v2 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim

from nets import inception_utils

# pylint: disable=g-long-lambda
trunc_normal = lambda stddev: tf.truncated_normal_initializer(
    0.0, stddev)


def inception_v2_base(inputs_camera, inputs_gated, inputs_lidar, 
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      use_separable_conv=True,
                      data_format='NHWC',
                      include_root_block=True,
                      scope=None):  
  """Inception v2 (6a2).

  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',
      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will
      not be available.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    use_separable_conv: Use a separable convolution for the first layer
      Conv2d_1a_7x7. If this is False, use a normal convolution instead.
    data_format: Data format of the activations ('NHWC' or 'NCHW').
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
        # Checkpoint
  print("=============================Checkpoint Train Images============================================")
  print("Camera shape:",inputs_camera)
  print("Gated shape:",inputs_gated)
  print("Lidar Shape:", inputs_lidar)

  # Input Resize
  #inputs_image = tf.reshape(inputs_image, shape=[224,224, 3])
  #inputs_gated = tf.reshape(inputs_gated, shape=[224,224, 3])

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  if data_format != 'NHWC' and data_format != 'NCHW':
    raise ValueError('data_format must be either NHWC or NCHW.')
  if data_format == 'NCHW' and use_separable_conv:
    raise ValueError(
        'separable convolution only supports NHWC layout. NCHW data format can'
        ' only be used when use_separable_conv is False.'
    )

  concat_dim = 3 if data_format == 'NHWC' else 1
  with tf.variable_scope(scope, 'InceptionV2', [inputs_camera, inputs_gated, inputs_lidar]):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

      net_camera = inputs_camera
      net_gated = inputs_gated
      net_lidar = inputs_lidar
      if include_root_block:
        # Note that sizes in the comments below assume an input spatial size of
        # 224x224, however, the inputs can be of any size greater 32x32.

        # 224 x 224 x 3
        end_point = 'Conv2d_1a_7x7'
        

        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          # Camera model
          net_camera = slim.separable_conv2d(
              inputs_camera,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          # Gated model
          net_gated = slim.separable_conv2d(
              inputs_gated,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          # Lidar model
          net_lidar = slim.separable_conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        else:
          # Use a normal convolution instead of a separable convolution.
          # Camera Model
          net_camera = slim.conv2d(
              inputs_camera,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          # Gated model
          net_gated = slim.conv2d(
              inputs_gated,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          # Lidar model
          net_lidar = slim.conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 112 x 112 x 64
        end_point = 'MaxPool_2a_3x3'
        # Camera model
        net_camera = slim.max_pool2d(net_camera, [3, 3], scope=end_point, stride=2)
        # Gated model
        net_gated = slim.max_pool2d(net_gated, [3, 3], scope=end_point, stride=2)
        # Lidar model
        net_lidar = slim.max_pool2d(net_lidar, [3, 3], scope=end_point, stride=2)
        # Fusion
        #net = tf.concat(axis=concat_dim, values=[net_camera, net_gated])
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        # Camera Model
        net_camera = slim.conv2d(
            net_camera,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))
        # Gated model
        net_gated = slim.conv2d(
            net_gated,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))
        # Lidar model
        net_lidar = slim.conv2d(
            net_lidar,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))

        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2c_3x3'
        # Camera Model
        net_camera = slim.conv2d(net_camera, depth(192), [3, 3], scope=end_point)
        # Gated Model
        net_gated = slim.conv2d(net_gated, depth(192), [3, 3], scope=end_point)
        # Lidar Model
        net_lidar = slim.conv2d(net_lidar, depth(192), [3, 3], scope=end_point)
        # Fusion
        #net = tf.concat(axis=concat_dim, values=[net_camera, net_gated])
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 192
        end_point = 'MaxPool_3a_3x3'
        # Camera Model
        net_camera_common = slim.max_pool2d(net_camera, [3, 3], scope=end_point, stride=2)
        # Gated Model
        net_gated_common = slim.max_pool2d(net_gated, [3, 3], scope=end_point, stride=2)
        # Lidar Model
        net_lidar_common = slim.max_pool2d(net_lidar, [3, 3], scope=end_point, stride=2)
        # Inputs
        """in_camera = net_camera_common
        in_gated = net_gated_common
        in_lidar = net_lidar_common

        # Fusion concat
        net = tf.concat(axis=concat_dim, values=[net_camera_common, net_gated_common, net_lidar_common])
        net_camera_common = net
        net_gated_common = net
        net_lidar_common = net
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points

      # 28 x 28 x 192
      # Inception module.
      # Camera Model
      end_point = 'Mixed_3b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_cam = slim.conv2d(net_camera_common, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_cam = slim.conv2d(
              net_camera_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_cam = slim.conv2d(branch_1_cam, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_cam = slim.conv2d(
              net_camera_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_cam = slim.conv2d(branch_2_cam, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_cam = slim.conv2d(branch_2_cam, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_cam = slim.avg_pool2d(net_camera_common, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_cam = slim.conv2d(
              branch_3_cam, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_cam, branch_1_cam, branch_2_cam, branch_3_cam])
        # gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(net_gated_common, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net_gated_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_gated = slim.conv2d(branch_1_gated, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net_gated_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net_gated_common, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net_gated = tf.concat(
            axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])
    
        # lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net_lidar_common, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net_lidar_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_lidar = slim.conv2d(branch_1_lidar, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net_lidar_common, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net_lidar_common, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')        
        net_lidar = tf.concat(
            axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        # *********
        # Fusion
        #net = tf.concat(axis=concat_dim, values=[net_camera, net_gated])
        # Fusion
        # Inputs
        #inputs = net_camera + net_gated
#        in_camera = net_camera
#        in_gated = net_gated

        # Camera Model
        # Local Attribute
#        net_local_camera = slim.conv2d(in_camera, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
#        net_local_camera = slim.batch_norm(net_local_camera)
#        net_local_camera = tf.nn.relu(net_local_camera)
#        net_local_camera = slim.conv2d(net_local_camera, 16, [1, 1])
    #net_local = slim.conv2d(net_local, 64, [1, 1],stride=1,weights_initializer=trunc_normal(1.0),scope=end_point)
#        net_local_camera = slim.batch_norm(net_local_camera)

        # Global Attribute
        # Global average pooling.
        #net_global = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
#        net_global_camera = slim.avg_pool2d(in_camera, [23,32], stride=34)
#        net_global_camera = slim.conv2d(net_global_camera, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
#        net_global_camera = slim.batch_norm(net_global_camera)
#        net_global_camera = tf.nn.relu(net_global_camera)
#        net_global_camera = slim.conv2d(net_global_camera, 16, [1, 1])
    #net_global = slim.conv2d(net_global,depth(16), [1, 1],stride=1,weights_initializer=trunc_normal(1.0),scope=end_point)
#        net_global_camera = slim.batch_norm(net_global_camera)
#        net_global_camera = tf.keras.layers.UpSampling2D(size=(23,32))(net_global_camera)

        # Combine Global Local Attributes and Calculating Weights
#        net_camera_gl = net_local_camera + net_global_camera
        #net = tf.concat(axis=concat_dim, values=[net_global + net_local])
#        weights_camera = slim.conv2d(net_camera_gl, 1, [1, 1])
#        weights_camera = tf.nn.softmax(weights_camera)
        

        # Gated Model
        # Local Attribute
#        net_local_gated = slim.conv2d(in_gated, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
#        net_local_gated = slim.batch_norm(net_local_gated)
#        net_local_gated = tf.nn.relu(net_local_gated)
#        net_local_gated = slim.conv2d(net_local_gated, 16, [1, 1])
    #net_local = slim.conv2d(net_local, 64, [1, 1],stride=1,weights_initializer=trunc_normal(1.0),scope=end_point)
#        net_local_gated = slim.batch_norm(net_local_gated)

        # Global Attribute
        # Global average pooling.
        #net_global = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
#        net_global_gated = slim.avg_pool2d(in_gated, [23,32], stride=34)
#        net_global_gated = slim.conv2d(net_global_gated, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
#        net_global_gated = slim.batch_norm(net_global_gated)
#        net_global_gated = tf.nn.relu(net_global_gated)
#        net_global_gated = slim.conv2d(net_global_gated, 16, [1, 1])
#        net_global_gated = slim.batch_norm(net_global_gated)
#        net_global_gated = tf.keras.layers.UpSampling2D(size=(23,32))(net_global_gated)

        # Combine Global Local Attributes and Calculating Weights
#        net_gated_gl = net_local_gated + net_global_gated
        #net = tf.concat(axis=concat_dim, values=[net_global + net_local])
#        weights_gated = slim.conv2d(net_gated_gl, 1, [1, 1])
#        weights_gated = tf.nn.softmax(weights_gated)

        # Weighted Fusion
        #net = 2*branch_0 + 2*branch_1
#        net = net_camera*weights_camera + net_gated*weights_gated
        
        end_points[end_point] = net_camera
        if end_point == final_endpoint: return net_camera, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      # Camera Model
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_cam = slim.conv2d(net_camera, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_cam = slim.conv2d(
              net_camera, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_cam = slim.conv2d(branch_1_cam, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_cam = slim.conv2d(
              net_camera, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_cam = slim.conv2d(branch_2_cam, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_cam = slim.conv2d(branch_2_cam, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_cam = slim.avg_pool2d(net_camera, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_cam = slim.conv2d(
              branch_3_cam, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_cam, branch_1_cam, branch_2_cam, branch_3_cam])

      # Gated Model
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(net_gated, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net_gated, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_gated = slim.conv2d(branch_1_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net_gated, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net_gated, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net_gated = tf.concat(
            axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])

      # Lidar Model
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net_lidar, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net_lidar, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_lidar = slim.conv2d(branch_1_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net_lidar, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net_lidar, [3, 3], scope='AvgPool_0a_3x3')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net_lidar = tf.concat(
            axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        # Fusion
        """in_camera = net_camera
        in_gated = net_gated
        in_lidar = net_lidar
        
        net_camera = tf.keras.layers.ZeroPadding2D(padding=((0,1),(1,1)))(net_camera)
        net_gated = tf.keras.layers.ZeroPadding2D(padding=((0,1),(1,1)))(net_gated)
        net_lidar = tf.keras.layers.ZeroPadding2D(padding=((0,1),(1,1)))(net_lidar)

        # Camera
        # Global Attribute
        #net_global = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
        net_global_camera = slim.avg_pool2d(in_camera, [14,13], stride=14)
        net_global_camera = slim.conv2d(net_global_camera, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
        net_global_camera = slim.batch_norm(net_global_camera)
        net_global_camera = tf.nn.relu(net_global_camera)
        net_global_camera = slim.conv2d(net_global_camera, 16, [1, 1])
        net_global_camera = slim.batch_norm(net_global_camera)
        net_global_camera = tf.keras.layers.UpSampling2D(size=(14,13))(net_global_camera)
        weights_camera_global = slim.conv2d(net_global_camera, 1, [1, 1])
        weights_camera_global = tf.nn.sigmoid(weights_camera_global)

        # Gated
        # Global Attribute
        #net_global = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
        net_global_gated = slim.avg_pool2d(in_gated, [14,13], stride=14)
        net_global_gated = slim.conv2d(net_global_gated, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
        net_global_gated = slim.batch_norm(net_global_gated)
        net_global_gated = tf.nn.relu(net_global_gated)
        net_global_gated = slim.conv2d(net_global_gated, 16, [1, 1])
        net_global_gated = slim.batch_norm(net_global_gated)
        net_global_gated = tf.keras.layers.UpSampling2D(size=(14,13))(net_global_gated)
        weights_gated_global = slim.conv2d(net_global_gated, 1, [1, 1])
        weights_gated_global = tf.nn.sigmoid(weights_gated_global)

        # Lidar
        # Global Attribute
        #net_global = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True)
        net_global_lidar = slim.avg_pool2d(in_lidar, [14,13], stride=14)
        net_global_lidar = slim.conv2d(net_global_lidar, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
        net_global_lidar = slim.batch_norm(net_global_lidar)
        net_global_lidar = tf.nn.relu(net_global_lidar)
        net_global_lidar = slim.conv2d(net_global_lidar, 16, [1, 1])
        net_global_lidar = slim.batch_norm(net_global_lidar)
        net_global_lidar = tf.keras.layers.UpSampling2D(size=(14,13))(net_global_lidar)
        weights_lidar_global = slim.conv2d(net_global_lidar, 1, [1, 1])
        weights_lidar_global = tf.nn.sigmoid(weights_lidar_global)

#        weights_global = tf.concat(axis=concat_dim, values=[weights_camera_global, weights_gated_global, weights_lidar_global])
#        weights_global = tf.nn.softmax(weights_global, axis=concat_dim)

        #weights_camera = weights[:, :, :, 0:1] 
        #weights_gated = weights[:, :, :, 1:2]
#        weights_camera_global, weights_gated_global, weights_lidar_global = tf.split(weights_global, 3, axis=concat_dim)

        net_global = net_camera*weights_camera_global + net_gated*weights_gated_global + net_lidar*weights_lidar_global
        
        #net = net_local + net_global
        net = tf.concat(axis=concat_dim, values=[net_local, net_global])"""

        net = tf.concat(axis=concat_dim, values=[net_camera, net_gated, net_lidar])
        
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points, net, net, net
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points, weights_camera, weights_gated
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v2_base_ssd_p(inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      use_separable_conv=True,
                      data_format='NHWC',
                      include_root_block=True,
                      scope=None):
  """Inception v2 (6a2).
  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',
      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will
      not be available.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    use_separable_conv: Use a separable convolution for the first layer
      Conv2d_1a_7x7. If this is False, use a normal convolution instead.
    data_format: Data format of the activations ('NHWC' or 'NCHW').
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  if data_format != 'NHWC' and data_format != 'NCHW':
    raise ValueError('data_format must be either NHWC or NCHW.')
  if data_format == 'NCHW' and use_separable_conv:
    raise ValueError(
        'separable convolution only supports NHWC layout. NCHW data format can'
        ' only be used when use_separable_conv is False.'
    )

  concat_dim = 3 if data_format == 'NHWC' else 1
  with tf.variable_scope(scope, 'InceptionV2', [inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy]):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

      net = inputs_camera
      if include_root_block:
        # Note that sizes in the comments below assume an input spatial size of
        # 224x224, however, the inputs can be of any size greater 32x32.

        # 224 x 224 x 3
        end_point = 'Conv2d_1a_7x7'

        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net = slim.separable_conv2d(
              inputs_camera,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        else:
          # Use a normal convolution instead of a separable convolution.
          net = slim.conv2d(
              inputs_camera,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
        # 112 x 112 x 64
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(
            net,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
        # 56 x 56 x 192
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points

      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v2_base_ssd(inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      use_separable_conv=True,
                      data_format='NHWC',
                      include_root_block=True,
                      scope=None):
  """Inception v2 (6a2).
  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',
      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will
      not be available.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    use_separable_conv: Use a separable convolution for the first layer
      Conv2d_1a_7x7. If this is False, use a normal convolution instead.
    data_format: Data format of the activations ('NHWC' or 'NCHW').
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  if data_format != 'NHWC' and data_format != 'NCHW':
    raise ValueError('data_format must be either NHWC or NCHW.')
  if data_format == 'NCHW' and use_separable_conv:
    raise ValueError(
        'separable convolution only supports NHWC layout. NCHW data format can'
        ' only be used when use_separable_conv is False.'
    )

  concat_dim = 3 if data_format == 'NHWC' else 1
  with tf.variable_scope(scope, 'InceptionV2', [inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy]):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

      net = inputs_camera
      if include_root_block:
        # Note that sizes in the comments below assume an input spatial size of
        # 224x224, however, the inputs can be of any size greater 32x32.

        # 224 x 224 x 3
        end_point = 'Conv2d_1a_7x7'

        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net_camera = slim.separable_conv2d(
              inputs_camera,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          net_gated = slim.separable_conv2d(
              inputs_gated,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Gated')
          net_lidar = slim.separable_conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Lidar')
        else:
          # Use a normal convolution instead of a separable convolution.
          net_camera = slim.conv2d(
              inputs_camera,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
          net_gated = slim.conv2d(
              inputs_gated,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Gated')
          net_lidar = slim.conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Lidar')
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 112 x 112 x 64
        end_point = 'MaxPool_2a_3x3'
        net_camera = slim.max_pool2d(net_camera, [3, 3], scope=end_point, stride=2)
        #net_gated = slim.max_pool2d(net_gated, [3, 3], scope=end_point, stride=2)
        #net_lidar = slim.max_pool2d(net_lidar, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        net_camera = slim.conv2d(
            net_camera,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))
        #net_gated = slim.conv2d(
        #    net_gated,
        #    depth(64), [1, 1],
        #    scope=end_point+'_Gated',
        #    weights_initializer=trunc_normal(0.1))
        #net_lidar = slim.conv2d(
        #    net_lidar,
        #    depth(64), [1, 1],
        #    scope=end_point+'_Lidar',
        #    weights_initializer=trunc_normal(0.1))
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2c_3x3'
        net_camera = slim.conv2d(net_camera, depth(192), [3, 3], scope=end_point)
        #net_gated = slim.conv2d(net_gated, depth(192), [3, 3], scope=end_point+'_Gated')
        #net_lidar = slim.conv2d(net_lidar, depth(192), [3, 3], scope=end_point+'_Lidar')
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 192
        end_point = 'MaxPool_3a_3x3'
        net_camera = slim.max_pool2d(net_camera, [3, 3], scope=end_point, stride=2)
        #net_gated = slim.max_pool2d(net_gated, [3, 3], scope=end_point+'_Gated', stride=2)
        #net_lidar = slim.max_pool2d(net_lidar, [3, 3], scope=end_point+'_Lidar', stride=2)
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points

      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v2_base_ssd_n(inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      use_separable_conv=True,
                      data_format='NHWC',
                      include_root_block=True,
                      scope=None):
  """Inception v2 (6a2).
  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',
      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will
      not be available.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    use_separable_conv: Use a separable convolution for the first layer
      Conv2d_1a_7x7. If this is False, use a normal convolution instead.
    data_format: Data format of the activations ('NHWC' or 'NCHW').
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  if data_format != 'NHWC' and data_format != 'NCHW':
    raise ValueError('data_format must be either NHWC or NCHW.')
  if data_format == 'NCHW' and use_separable_conv:
    raise ValueError(
        'separable convolution only supports NHWC layout. NCHW data format can'
        ' only be used when use_separable_conv is False.'
    )

  concat_dim = 3 if data_format == 'NHWC' else 1
  with tf.variable_scope(scope, 'InceptionV2', [inputs_camera, inputs_gated, inputs_lidar, inputs_camera_entropy, inputs_gated_entropy, inputs_lidar_entropy]):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

      net = inputs_camera
      if include_root_block:
        # Note that sizes in the comments below assume an input spatial size of
        # 224x224, however, the inputs can be of any size greater 32x32.

        # 224 x 224 x 3
        end_point = 'Conv2d_1a_7x7'

        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net_camera = slim.separable_conv2d(
              inputs_camera,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Camera')
        else:
          # Use a normal convolution instead of a separable convolution.
          net_camera = slim.conv2d(
              inputs_camera,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Camera')
        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net_gated = slim.separable_conv2d(
              inputs_gated,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Gated')
        else:
          # Use a normal convolution instead of a separable convolution.
          net_gated = slim.conv2d(
              inputs_gated,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Gated')
        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net_lidar = slim.separable_conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Lidar')
        else:
          # Use a normal convolution instead of a separable convolution.
          net_lidar = slim.conv2d(
              inputs_lidar,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point+'_Lidar')
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 112 x 112 x 64
        end_point = 'MaxPool_2a_3x3'
        net_camera = slim.max_pool2d(net_camera, [3, 3], scope=end_point, stride=2)
        #net_gated = slim.max_pool2d(net_gated, [3, 3], scope=end_point, stride=2)
        #net_lidar = slim.max_pool2d(net_lidar, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        net_camera = slim.conv2d(
            net_camera,
            depth(64), [1, 1],
            scope=end_point+'_Camera',
            weights_initializer=trunc_normal(0.1))
        net_gated = slim.conv2d(
            net_gated,
            depth(64), [1, 1],
            scope=end_point+'_Gated',
            weights_initializer=trunc_normal(0.1))
        net_lidar = slim.conv2d(
            net_lidar,
            depth(64), [1, 1],
            scope=end_point+'_Lidar',
            weights_initializer=trunc_normal(0.1))
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2c_3x3'
        net_camera = slim.conv2d(net_camera, depth(192), [3, 3], scope=end_point+'_Camera')
        net_gated = slim.conv2d(net_gated, depth(192), [3, 3], scope=end_point+'_Gated')
        net_lidar = slim.conv2d(net_lidar, depth(192), [3, 3], scope=end_point+'_Lidar')
        end_points[end_point] = net_camera
        if end_point == final_endpoint:
          return net_camera, end_points
        # 56 x 56 x 192
        end_point = 'MaxPool_3a_3x3'
        net_camera = slim.max_pool2d(net_camera, [3, 3], scope=end_point+'_Camera', stride=2)
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 192, [3, 3],stride=2)
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
   #     net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
        #net_gated = slim.max_pool2d(net_gated, [3, 3], scope=end_point+'_Gated', stride=2)
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 192, [3, 3],stride=2)
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
#        net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
        #net_lidar = slim.max_pool2d(net_lidar, [3, 3], scope=end_point+'_Lidar', stride=2)
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 192, [3, 3],stride=2)
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
#        net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights #(256,256,3)
        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points
     
      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_camera = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_Camera')
        with tf.variable_scope('Branch_1'):
          branch_1_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Camera')
          branch_1_camera = slim.conv2d(branch_1_camera, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3_Camera')
        with tf.variable_scope('Branch_2'):
          branch_2_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_Camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_Camera')
        with tf.variable_scope('Branch_3'):
          branch_3_camera = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_Camera')
          branch_3_camera = slim.conv2d(
              branch_3_camera, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_Camera')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_camera, branch_1_camera, branch_2_camera, branch_3_camera])
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 256, [3, 3],stride=2, scope='Block2_Camera')
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
        #net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
      # Gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_Gated')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Gated')
          branch_1_gated = slim.conv2d(branch_1_gated, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3_Gated')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_Gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_Gated')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_Gated')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_Gated')
        net_gated = tf.concat(
            axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 256, [3, 3],stride=2, scope='Block2_Gated')
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
        #net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
        # Lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_Lidar')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Lidar')
          branch_1_lidar = slim.conv2d(branch_1_lidar, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3_Lidar')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_Lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_Lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_Lidar')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_Lidar')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_Lidar')
        net_lidar = tf.concat(
            axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 256, [3, 3],stride=2, scope='Block2_Lidar')
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
        #net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights

        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_camera = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_camera')
        with tf.variable_scope('Branch_1'):
          branch_1_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_1_camera = slim.conv2d(branch_1_camera, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_camera')
        with tf.variable_scope('Branch_2'):
          branch_2_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_camera')
        with tf.variable_scope('Branch_3'):
          branch_3_camera = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_camera')
          branch_3_camera = slim.conv2d(
              branch_3_camera, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_camera')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_camera, branch_1_camera, branch_2_camera, branch_3_camera])
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 320, [3, 3],stride=2, scope='Block3_Camera')
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
        #net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
      # Gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_gated')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_1_gated = slim.conv2d(branch_1_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_gated')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_gated')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_gated')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_gated')
        net_gated = tf.concat(
            axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 320, [3, 3],stride=2, scope='Block3_Gated')
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
        #net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
      # Lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1_lidar')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_1_lidar = slim.conv2d(branch_1_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_lidar')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3_lidar')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_lidar')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_lidar')
        net_lidar = tf.concat(
            axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 320, [3, 3],stride=2, scope='Block3_Lidar')
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
        #net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights

        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""        
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_camera = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_0_camera = slim.conv2d(branch_0_camera, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3_camera')
        with tf.variable_scope('Branch_1'):
          branch_1_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_1_camera = slim.conv2d(
              branch_1_camera, depth(96), [3, 3], scope='Conv2d_0b_3x3_camera')
          branch_1_camera = slim.conv2d(
              branch_1_camera, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3_camera')
        with tf.variable_scope('Branch_2'):
          branch_2_camera = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3_camera')
        net_camera = tf.concat(axis=concat_dim, values=[branch_0_camera, branch_1_camera, branch_2_camera])
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 1216, [3, 3],stride=4, scope='Block4_Camera')
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
        #net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
      # Gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_0_gated = slim.conv2d(branch_0_gated, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3_gated')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_1_gated = slim.conv2d(
              branch_1_gated, depth(96), [3, 3], scope='Conv2d_0b_3x3_gated')
          branch_1_gated = slim.conv2d(
              branch_1_gated, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3_gated')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3_gated')
        net_gated = tf.concat(axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated])
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 1216, [3, 3],stride=4, scope='Block4_Gated')
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
        #net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
      # Lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_0_lidar = slim.conv2d(branch_0_lidar, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3_lidar')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_1_lidar = slim.conv2d(
              branch_1_lidar, depth(96), [3, 3], scope='Conv2d_0b_3x3_lidar')
          branch_1_lidar = slim.conv2d(
              branch_1_lidar, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3_lidar')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3_lidar')
        net_lidar = tf.concat(axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar])
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 1216, [3, 3],stride=4, scope='Block4_Lidar')
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
        #net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights       
        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_camera = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1_camera')
        with tf.variable_scope('Branch_1'):
          branch_1_camera = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_1_camera = slim.conv2d(
              branch_1_camera, depth(96), [3, 3], scope='Conv2d_0b_3x3_camera')
        with tf.variable_scope('Branch_2'):
          branch_2_camera = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_camera')
        with tf.variable_scope('Branch_3'):
          branch_3_camera = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_camera')
          branch_3_camera = slim.conv2d(
              branch_3_camera, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_camera')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_camera, branch_1_camera, branch_2_camera, branch_3_camera])
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 576, [3, 3],stride=4, scope='Block5_Camera')
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
        #net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
      # Gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1_gated')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_1_gated = slim.conv2d(
              branch_1_gated, depth(96), [3, 3], scope='Conv2d_0b_3x3_gated')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_gated')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_gated')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_gated')
        net_gated = tf.concat(
              axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 576, [3, 3],stride=4, scope='Block5_Gated')
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
        #net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
      # Lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1_lidar')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_1_lidar = slim.conv2d(
              branch_1_lidar, depth(96), [3, 3], scope='Conv2d_0b_3x3_lidar')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_lidar')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_lidar')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_lidar')
        net_lidar = tf.concat(
              axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 576, [3, 3],stride=4, scope='Block5_Lidar')
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
        #net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights
        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_camera = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1_camera')
        with tf.variable_scope('Branch_1'):
          branch_1_camera = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_1_camera = slim.conv2d(branch_1_camera, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_camera')
        with tf.variable_scope('Branch_2'):
          branch_2_camera = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_camera')
          branch_2_camera = slim.conv2d(branch_2_camera, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_camera')
        with tf.variable_scope('Branch_3'):
          branch_3_camera = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_camera')
          branch_3_camera = slim.conv2d(
              branch_3_camera, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_camera')
        net_camera = tf.concat(
            axis=concat_dim, values=[branch_0_camera, branch_1_camera, branch_2_camera, branch_3_camera])
        """net_camera_entropy = slim.conv2d(inputs_camera_entropy, 576, [3, 3],stride=4, scope='Block6_Camera')
        net_camera_weights = tf.nn.sigmoid(net_camera_entropy)
        #net_camera = tf.keras.layers.UpSampling2D(size=(2,2))(net_camera)
        net_camera = net_camera*net_camera_weights
      # Gated
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_gated= slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1_gated')
        with tf.variable_scope('Branch_1'):
          branch_1_gated = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_1_gated = slim.conv2d(branch_1_gated, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_gated')
        with tf.variable_scope('Branch_2'):
          branch_2_gated = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_gated')
          branch_2_gated = slim.conv2d(branch_2_gated, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_gated')
        with tf.variable_scope('Branch_3'):
          branch_3_gated = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_gated')
          branch_3_gated = slim.conv2d(
              branch_3_gated, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_gated')
        net_gated = tf.concat(
                axis=concat_dim, values=[branch_0_gated, branch_1_gated, branch_2_gated, branch_3_gated])
        net_gated_entropy = slim.conv2d(inputs_gated_entropy, 576, [3, 3],stride=4, scope='Block6_Gated')
        net_gated_weights = tf.nn.sigmoid(net_gated_entropy)
        #net_gated = tf.keras.layers.UpSampling2D(size=(2,2))(net_gated)
        net_gated = net_gated*net_gated_weights
      # Lidar
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0_lidar = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1_lidar')
        with tf.variable_scope('Branch_1'):
          branch_1_lidar = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_1_lidar = slim.conv2d(branch_1_lidar, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_lidar')
        with tf.variable_scope('Branch_2'):
          branch_2_lidar = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3_lidar')
          branch_2_lidar = slim.conv2d(branch_2_lidar, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3_lidar')
        with tf.variable_scope('Branch_3'):
          branch_3_lidar = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3_lidar')
          branch_3_lidar = slim.conv2d(
              branch_3_lidar, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1_lidar')
        net_lidar = tf.concat(
                axis=concat_dim, values=[branch_0_lidar, branch_1_lidar, branch_2_lidar, branch_3_lidar])
        net_lidar_entropy = slim.conv2d(inputs_lidar_entropy, 576, [3, 3],stride=4, scope='Block6_Lidar')
        net_lidar_weights = tf.nn.sigmoid(net_lidar_entropy)
        #net_lidar = tf.keras.layers.UpSampling2D(size=(2,2))(net_lidar)
        net_lidar = net_lidar*net_lidar_weights
        # Fusion
        net = tf.concat(axis=concat_dim, values = [net_camera, net_gated, net_lidar])"""
        net = net_camera
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v2(inputs_camera, inputs_gated,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV2',
                 global_pool=False):
  """Inception v2 model for classification.

  Constructs an Inception v2 network for classification as described in
  http://arxiv.org/abs/1502.03167.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  # Final pooling and prediction
  with tf.variable_scope(
      scope, 'InceptionV2', [inputs_camera, inputs_gated], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      # Camera Model
      net_camera, end_points = inception_v2_base(
          inputs_camera, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)
      # Gated Model
      net_gated, end_points = inception_v2_base(
          inputs_gated, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)
      # Fusion
      net = tf.concat(axis=concat_dim, values = [net_camera, net_gated])
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(
              input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_{}x{}'.format(*kernel_size))
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v2.default_image_size = 224


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


inception_v2_arg_scope = inception_utils.inception_arg_scope
