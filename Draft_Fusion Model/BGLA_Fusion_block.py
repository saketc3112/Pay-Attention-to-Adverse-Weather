# Tensorflow Slim
import tensorflow.compat.v1 as tf
import tf_slim as slim

# Bayesian Global Local Attention Fusion Block
# Inprogress
def BGLA(layer1, layer2, bayesian_weights_1, bayesian_weights_2):
    # Inputs
    inputs = branch_0 + branch_1

    # Local Attribute
    net_local = slim.conv2d(inputs, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
    net_local = slim.batch_norm(net_local)
    net_local = slim.conv2d(net_local, 16, [1, 1], activation_fn=tf.nn.relu)
    #net_local = slim.conv2d(net_local, 64, [1, 1],stride=1,weights_initializer=trunc_normal(1.0),scope=end_point)
    net_local = slim.batch_norm(net_local)

    # Global Attribute
    net_global = slim.avg_pool2d(inputs, [1,1], stride=2)
    net_global = slim.conv2d(net_global, 16, [1, 1],stride=1,weights_initializer=trunc_normal(1.0))
    net_global = slim.batch_norm(net_global)
    net_global = slim.conv2d(net_global, 16, [1, 1], activation_fn=tf.nn.relu)
    #net_global = slim.conv2d(net_global,depth(16), [1, 1],stride=1,weights_initializer=trunc_normal(1.0),scope=end_point)
    net_global = slim.batch_norm(net_global)

    # Combine Global Local Attributes and Calculating Weights
    net = net_local + net_global
    #net = tf.concat(axis=concat_dim, values=[net_global + net_local])
    weights = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.sigmoid)

    # Weighted Fusion
    fused_net = 2*branch_0*weights + 2*branch_1*(1 - weights)

    return fused_net




