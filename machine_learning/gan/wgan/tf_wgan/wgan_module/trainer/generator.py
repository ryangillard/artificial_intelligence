import tensorflow as tf

from .print_object import print_obj


def generator_network(Z, mode, params, reuse=False):
    """Creates generator network and returns generated output.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.
        reuse: bool, whether to reuse variables or not.

    Returns:
        Generated outputs tensor of shape
            [cur_batch_size, height * width * depth].
    """
    # Create regularizer for dense layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["generator_l1_regularization_scale"],
        scale_l2=params["generator_l2_regularization_scale"]
    )

    with tf.variable_scope("generator", reuse=reuse):
        # Project latent vectors.
        projection_height = params["generator_projection_dims"][0]
        projection_width = params["generator_projection_dims"][1]
        projection_depth = params["generator_projection_dims"][2]

        # shape = (
        #     cur_batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection = tf.layers.dense(
            inputs=Z,
            units=projection_height * projection_width * projection_depth,
            activation=tf.nn.leaky_relu,
            name="projection_layer"
        )
        print_obj("generator_network", "projection", projection)

        # shape = (
        #     cur_batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection_batch_norm = tf.layers.batch_normalization(
            inputs=projection,
            training=(mode == tf.estimator.ModeKeys.TRAIN),
            name="projection_batch_norm"
        )
        print_obj(
            "generator_network",
            "projection_batch_norm",
            projection_batch_norm
        )

        # Reshape projection into "image".
        # shape = (
        #     cur_batch_size,
        #     projection_height,
        #     projection_width,
        #     projection_depth
        # )
        network = tf.reshape(
            tensor=projection_batch_norm,
            shape=[-1, projection_height, projection_width, projection_depth],
            name="projection_reshaped"
        )
        print_obj("generator_network", "network", network)

        # Iteratively build upsampling layers.
        for i in range(len(params["generator_num_filters"])):
            # Add convolutional transpose layers with given params per layer.
            # shape = (
            #     cur_batch_size,
            #     generator_kernel_sizes[i - 1] * generator_strides[i],
            #     generator_kernel_sizes[i - 1] * generator_strides[i],
            #     generator_num_filters[i]
            # )
            network = tf.layers.conv2d_transpose(
                inputs=network,
                filters=params["generator_num_filters"][i],
                kernel_size=params["generator_kernel_sizes"][i],
                strides=params["generator_strides"][i],
                padding="same",
                activation=tf.nn.leaky_relu,
                use_bias=False,
                kernel_regularizer=regularizer,
                name="layers_conv2d_tranpose_{}".format(i)
            )
            print_obj("generator_network", "network", network)

            # Add batch normalization to keep the inputs from blowing up.
            network = tf.layers.batch_normalization(
                inputs=network,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name="layers_batch_norm_{}".format(i)
            )
            print_obj("generator_network", "network", network)

        # Final conv2d transpose layer for image output.
        # shape = (cur_batch_size, height * width * depth)
        generated_outputs = tf.layers.conv2d_transpose(
                inputs=network,
                filters=params["generator_final_num_filters"],
                kernel_size=params["generator_final_kernel_size"],
                strides=params["generator_final_stride"],
                padding="same",
                activation=tf.nn.tanh,
                use_bias=False,
                kernel_regularizer=regularizer,
                name="layers_conv2d_tranpose_generated_outputs"
        )
        print_obj("generator_network", "generated_outputs", generated_outputs)

    return generated_outputs


def get_generator_loss(generated_logits):
    """Gets generator loss.

    Args:
        generated_logits: tensor, shape of
            [cur_batch_size, height * width * depth].

    Returns:
        Tensor of generator's total loss of shape [].
    """
    # Calculate base generator loss.
    generator_loss = -tf.reduce_mean(
        input_tensor=generated_logits,
        name="generator_loss"
    )
    print_obj(
        "\nget_generator_loss",
        "generator_loss",
        generator_loss
    )

    # Get regularization losses.
    generator_regularization_loss = tf.losses.get_regularization_loss(
        scope="generator",
        name="generator_regularization_loss"
    )
    print_obj(
        "get_generator_loss",
        "generator_regularization_loss",
        generator_regularization_loss
    )

    # Combine losses for total losses.
    generator_total_loss = tf.math.add(
        x=generator_loss,
        y=generator_regularization_loss,
        name="generator_total_loss"
    )
    print_obj(
        "get_generator_loss", "generator_total_loss", generator_total_loss
    )

    return generator_total_loss
