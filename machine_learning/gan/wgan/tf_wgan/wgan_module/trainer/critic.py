import tensorflow as tf

from .print_object import print_obj


def critic_network(X, params, reuse=False):
    """Creates critic network and returns logits.

    Args:
        X: tensor, image tensors of shape
            [cur_batch_size, height, width, depth].
        params: dict, user passed parameters.
        reuse: bool, whether to reuse variables or not.

    Returns:
        Logits tensor of shape [cur_batch_size, 1].
    """
    # Create the input layer to our DNN.
    # shape = (cur_batch_size, height * width * depth)
    network = X
    print_obj("\ncritic_network", "network", network)

    # Create regularizer for dense layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["critic_l1_regularization_scale"],
        scale_l2=params["critic_l2_regularization_scale"]
    )

    with tf.variable_scope("critic", reuse=reuse):
        # Iteratively build downsampling layers.
        for i in range(len(params["critic_num_filters"])):
            # Add convolutional transpose layers with given params per layer.
            # shape = (
            #     cur_batch_size,
            #     critic_kernel_sizes[i - 1] / critic_strides[i],
            #     critic_kernel_sizes[i - 1] / critic_strides[i],
            #     critic_num_filters[i]
            # )
            network = tf.layers.conv2d(
                inputs=network,
                filters=params["critic_num_filters"][i],
                kernel_size=params["critic_kernel_sizes"][i],
                strides=params["critic_strides"][i],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_regularizer=regularizer,
                name="layers_conv2d_{}".format(i)
            )
            print_obj("critic_network", "network", network)

            # Add some dropout for better regularization and stability.
            network = tf.layers.dropout(
                inputs=network,
                rate=params["critic_dropout_rates"][i],
                name="layers_dropout_{}".format(i)
            )
            print_obj("critic_network", "network", network)

        # Flatten network output.
        # shape = (
        #     cur_batch_size,
        #     (critic_kernel_sizes[-2] / critic_strides[-1]) ** 2 * critic_num_filters[-1]
        # )
        network_flat = tf.layers.Flatten()(inputs=network)
        print_obj("critic_network", "network_flat", network_flat)

        # Final linear layer for logits.
        # shape = (cur_batch_size, 1)
        logits = tf.layers.dense(
            inputs=network_flat,
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_logits"
        )
        print_obj("critic_network", "logits", logits)

    return logits


def get_critic_loss(generated_logits, real_logits):
    """Gets critic loss.

    Args:
        generated_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        real_logits: tensor, shape of
            [cur_batch_size, height * width * depth].

    Returns:
        Tensor of critic's total loss of shape [].
    """
    # Calculate base critic loss.
    critic_real_loss = tf.reduce_mean(
        input_tensor=real_logits,
        name="critic_real_loss"
    )
    print_obj(
        "\nget_critic_loss",
        "critic_real_loss",
        critic_real_loss
    )

    critic_generated_loss = tf.reduce_mean(
        input_tensor=generated_logits,
        name="critic_generated_loss"
    )
    print_obj(
        "get_critic_loss",
        "critic_generated_loss",
        critic_generated_loss
    )

    critic_loss = tf.add(
        x=critic_real_loss, y=-critic_generated_loss,
        name="critic_loss"
    )
    print_obj(
        "get_critic_loss",
        "critic_loss",
        critic_loss
    )

    # Get regularization losses.
    critic_regularization_loss = tf.losses.get_regularization_loss(
        scope="critic",
        name="critic_regularization_loss"
    )
    print_obj(
        "get_critic_loss",
        "critic_regularization_loss",
        critic_regularization_loss
    )

    # Combine losses for total losses.
    critic_total_loss = tf.math.add(
        x=critic_loss,
        y=critic_regularization_loss,
        name="critic_total_loss"
    )
    print_obj(
        "get_critic_loss",
        "critic_total_loss",
        critic_total_loss
    )

    return critic_total_loss
