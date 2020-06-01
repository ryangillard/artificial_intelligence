import tensorflow as tf

from .print_object import print_obj


def discriminator_network(X, params, reuse=False):
    """Creates discriminator network and returns logits.

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
    print_obj("\ndiscriminator_network", "network", network)

    # Create regularizer for dense layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["discriminator_l1_regularization_scale"],
        scale_l2=params["discriminator_l2_regularization_scale"]
    )

    with tf.variable_scope("discriminator", reuse=reuse):
        # Iteratively build downsampling layers.
        for i in range(len(params["discriminator_num_filters"])):
            # Add convolutional transpose layers with given params per layer.
            # shape = (
            #     cur_batch_size,
            #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
            #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
            #     discriminator_num_filters[i]
            # )
            network = tf.layers.conv2d(
                inputs=network,
                filters=params["discriminator_num_filters"][i],
                kernel_size=params["discriminator_kernel_sizes"][i],
                strides=params["discriminator_strides"][i],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_regularizer=regularizer,
                name="layers_conv2d_{}".format(i)
            )
            print_obj("discriminator_network", "network", network)

            # Add some dropout for better regularization and stability.
            network = tf.layers.dropout(
                inputs=network,
                rate=params["discriminator_dropout_rates"][i],
                name="layers_dropout_{}".format(i)
            )
            print_obj("discriminator_network", "network", network)

        # Flatten network output.
        # shape = (
        #     cur_batch_size,
        #     (discriminator_kernel_sizes[-2] / discriminator_strides[-1]) ** 2 * discriminator_num_filters[-1]
        # )
        network_flat = tf.layers.Flatten()(inputs=network)
        print_obj("discriminator_network", "network_flat", network_flat)

        # Final linear layer for logits.
        # shape = (cur_batch_size, 1)
        logits = tf.layers.dense(
            inputs=network_flat,
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_logits"
        )
        print_obj("discriminator_network", "logits", logits)

    return logits


def get_discriminator_loss(generated_logits, real_logits):
    """Gets discriminator loss.

    Args:
        generated_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        real_logits: tensor, shape of
            [cur_batch_size, height * width * depth].

    Returns:
        Tensor of discriminator's total loss of shape [].
    """
    # Calculate base discriminator loss.
    discriminator_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_logits,
        labels=tf.ones_like(tensor=real_logits),
        name="discriminator_real_loss"
    )
    print_obj(
        "\nget_discriminator_loss",
        "discriminator_real_loss",
        discriminator_real_loss
    )

    discriminator_generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=generated_logits,
        labels=tf.zeros_like(tensor=generated_logits),
        name="discriminator_generated_loss"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_generated_loss",
        discriminator_generated_loss
    )

    discriminator_loss = tf.reduce_mean(
        input_tensor=tf.add(
            x=discriminator_real_loss, y=discriminator_generated_loss
        ),
        name="discriminator_loss"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_loss",
        discriminator_loss
    )

    # Get regularization losses.
    discriminator_regularization_loss = tf.losses.get_regularization_loss(
        scope="discriminator",
        name="discriminator_regularization_loss"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_regularization_loss",
        discriminator_regularization_loss
    )

    # Combine losses for total losses.
    discriminator_total_loss = tf.math.add(
        x=discriminator_loss,
        y=discriminator_regularization_loss,
        name="discriminator_total_loss"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_total_loss",
        discriminator_total_loss
    )

    return discriminator_total_loss
