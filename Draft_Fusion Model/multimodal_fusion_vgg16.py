"""
According to the SeeingThroughFog paper following modifications are done to existing 
VGG16 model architecture:
1. Channels are reduced to half of the original network.
2. Considered Conv4 to Conv10 feature layers in the architecture
"""

# Currently this model takes 2 inputs (Camera and Gated images)
def multimodal_fusion_vgg16(inputs_cam, inputs_gated,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           reuse=None,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Modifications are performed to Oxford Net VGG 16-Layers version D Example."""
  
  concat_dim = 3 if data_format == 'NHWC' else 1

  with tf.variable_scope(
      scope, 'vgg_16', [inputs_cam, inputs_gated], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # camera model block 1
      net_camera = slim.repeat(inputs_camera, 1, slim.conv2d, 64, [3, 3], scope='conv2_camera')
      net_camera = slim.max_pool2d(net_camera, [2, 2], scope='pool2_camera')

      # gated model block 1
      net_gated = slim.repeat(inputs_gated, 1, slim.conv2d, 64, [3, 3], scope='conv2_gated')
      net_gated = slim.max_pool2d(net_gated, [2, 2], scope='pool2_gated')

      # Deep Feature Exchange 1
      net = tf.concat(axis=concat_dim, values = [pool2_camera, pool2_gated])
      
      # camera model block 2
      net_camera = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3_camera')
      net_camera = slim.max_pool2d(net_camera, [2, 2], scope='pool3_camera')

      # gated  model block 2
      net_gated = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3_gated')
      net_gated = slim.max_pool2d(net_gated, [2, 2], scope='pool3_gated')

      # Deep Feature Exchange 2
      net = tf.concat(axis=concat_dim, values = [pool3_camera, pool3_gated])

      # camera model block 3
      net_camera = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv4_camera')
      net_camera = slim.max_pool2d(net_camera, [2, 2], scope='pool4_camera')

      # gated  model block 3
      net_gated = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv4_gated')
      net_gated = slim.max_pool2d(net_gated, [2, 2], scope='pool4_gated')

      # Deep Feature Exchange 3
      net = tf.concat(axis=concat_dim, values = [pool4_camera, pool4_gated])

      # camera model block 4
      net_camera = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv5_camera')
      net_camera = slim.max_pool2d(net_camera, [2, 2], scope='pool5_camera')

      # gated  model block 4
      net_gated = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv5_gated')
      net_gated = slim.max_pool2d(net_gated, [2, 2], scope='pool5_gated')

      # Deep Feature Exchange 4
      net = tf.concat(axis=concat_dim, values = [pool5_camera, pool5_gated])


      # camera model block 5
      net_camera = slim.conv2d(net, 2048, [7, 7], padding=fc_conv_padding, scope='fc6_camera')
      
      # gated model block 5
      net_gated = slim.conv2d(net, 2048, [7, 7], padding=fc_conv_padding, scope='fc6_gated')

      # Deep Feature Exchange 5
      net = tf.concat(axis=concat_dim, values = [fc6_camera, fc6_gated])

      # Dropout
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      # camera model block 6
      net_camera = slim.conv2d(net, 2048, [1, 1], scope='fc7_camera')

      # gated model block 6
      net_gated = slim.conv2d(net, 2048, [1, 1], scope='fc7_gated')

      # Deep Feature Exchange 6
      net = tf.concat(axis=concat_dim, values = [fc7_camera, fc7_gated])

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(
            input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points