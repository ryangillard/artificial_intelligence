import tensorflow as tf

from .print_object import print_obj


def generator_network(Z, params, reuse=False):
    """Creates generator network and returns generated output.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        params: dict, user passed parameters.
        reuse: bool, whether to reuse variables or not.

    Returns:
        Generated outputs tensor of shape
            [cur_batch_size, height * width * depth].
    """
    # Create the input layer to our DNN.
    # shape = (cur_batch_size, latent_size)
    network = Z
    print_obj("\ngenerator_network", "network", network)

    # Create regularizer for dense layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["generator_l1_regularization_scale"],
        scale_l2=params["generator_l2_regularization_scale"]
    )

    with tf.variable_scope("generator", reuse=reuse):
        # Add hidden layers with the given number of units/neurons per layer.
        for i, units in enumerate(params["generator_hidden_units"]):
            # shape = (cur_batch_size, generator_hidden_units[i])
            network = tf.layers.dense(
                inputs=network,
                units=units,
                activation=tf.nn.leaky_relu,
                kernel_regularizer=regularizer,
                name="layers_dense_{}".format(i)
            )
            print_obj("generator_network", "network", network)

        # Final linear layer for outputs.
        # shape = (cur_batch_size, height * width * depth)
        generated_outputs = tf.layers.dense(
            inputs=network,
            units=params["height"] * params["width"] * params["depth"],
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_generated_outputs"
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
    generator_loss = tf.reduce_mean(
        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generated_logits,
            labels=tf.ones_like(tensor=generated_logits)
        ),
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
