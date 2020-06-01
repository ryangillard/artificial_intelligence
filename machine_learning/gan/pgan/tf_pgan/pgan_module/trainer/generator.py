import tensorflow as tf

from .print_object import print_obj


def create_generator_base_conv_layer_block(regularizer, params):
    """Creates generator base conv layer block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of base conv layers.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Create list of base conv layers.
        base_conv_layers = [
            tf.layers.Conv2D(
                filters=params["base_num_filters"][i],
                kernel_size=params["base_kernel_sizes"][i],
                strides=params["base_strides"][i],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="base_layers_conv2d_{}".format(i)
            )
            for i in range(len(params["base_num_filters"]))
        ]
        print_obj(
            "\ncreate_generator_base_conv_layer_block",
            "base_conv_layers",
            base_conv_layers
        )

    return base_conv_layers


def create_generator_growth_layer_block(block_idx, regularizer, params):
    """Creates generator growth block.

    Args:
        block_idx: int, the current growth block's index.
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of growth block layers.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Create new inner convolutional layers.
        conv_layers = [
            tf.layers.Conv2D(
                filters=params["growth_num_filters"][block_idx][layer_idx],
                kernel_size=params["growth_kernel_sizes"][block_idx][layer_idx],
                strides=params["growth_strides"][block_idx][layer_idx],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="growth_layers_conv2d_{}_{}".format(
                    block_idx, layer_idx
                )
            )
            for layer_idx in range(
                len(params["growth_num_filters"][block_idx])
            )
        ]
        print_obj(
            "\ncreate_generator_growth_layer_block", "conv_layers", conv_layers
        )

    return conv_layers


def create_generator_to_rgb_layers(regularizer, params):
    """Creates generator toRGB layers of 1x1 convs.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of toRGB 1x1 conv layers.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Create list to hold toRGB 1x1 convs.
        to_rgb_conv_layers = [
            # Create base fromRGB conv 1x1.
            tf.layers.Conv2D(
                filters=params["depth"],
                kernel_size=1,
                strides=1,
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="base_to_rgb_layers_conv2d"
            )
        ]

        # Create toRGB 1x1 convs for growth.
        growth_to_rgb_conv_layers = [
            tf.layers.Conv2D(
                filters=params["depth"],
                kernel_size=1,
                strides=1,
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="growth_to_rgb_layers_conv2d_{}".format(i)
            )
            for i in range(len(params["growth_num_filters"]))
        ]

        # Combine base and growth toRGB 1x1 convs.
        to_rgb_conv_layers.extend(growth_to_rgb_conv_layers)
        print_obj(
            "\ncreate_generator_to_rgb_layers",
            "to_rgb_conv_layers",
            to_rgb_conv_layers
        )

    return to_rgb_conv_layers


def upsample_generator_image(image, block_idx):
    """Upsamples generator image.

    Args:
        image: tensor, image created by generator conv block.
        block_idx: int, index of the current generator growth block.

    Returns:
        Upsampled image tensor.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Upsample from s X s to 2s X 2s image.
        upsampled_image = tf.image.resize(
            images=image,
            size=tf.shape(input=image)[1:3] * 2,
            method="nearest",
            name="growth_upsampled_image_{}".format(
                block_idx
            )
        )
        print_obj(
            "\nupsample_generator_image",
            "upsampled_image",
            upsampled_image
        )

    return upsampled_image


def create_base_generator_network(X, to_rgb_conv_layers, blocks):
    """Creates base generator network.

    Args:
        X: tensor, input image to generator.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.

    Returns:
        Final network block conv tensor.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Only need the first block and toRGB conv layer for base network.
        block_layers = blocks[0]
        to_rgb_conv_layer = to_rgb_conv_layers[0]

        # Pass inputs through layer chain.
        block_conv = block_layers[0](inputs=X)
        for i in range(1, len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
        to_rgb_conv = to_rgb_conv_layer(inputs=block_conv)
        print_obj(
            "\ncreate_base_generator_network",
            "to_rgb_conv",
            to_rgb_conv
        )

    return to_rgb_conv


def create_growth_transition_generator_network(
        X,
        to_rgb_conv_layers,
        blocks,
        alpha_var,
        trans_idx):
    """Creates base generator network.

    Args:
        X: tensor, input image to generator.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        trans_idx: int, index of current growth transition.

    Returns:
        Final network block conv tensor.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Permanent blocks.
        permanent_blocks = blocks[0:trans_idx + 1]

        # Base block doesn't need any upsampling so it's handled differently.
        base_block_conv_layers = permanent_blocks[0]

        # Pass inputs through layer chain.
        block_conv = base_block_conv_layers[0](inputs=X)
        for i in range(1, len(base_block_conv_layers)):
            block_conv = base_block_conv_layers[i](inputs=block_conv)

        # Growth blocks require first the prev conv layer's image upsampled.
        for i in range(1, len(permanent_blocks)):
            # Upsample previous block's image.
            block_conv = upsample_generator_image(
                image=block_conv, block_idx=i
            )

            block_conv_layers = permanent_blocks[i]
            for i in range(1, len(block_conv_layers)):
                block_conv = block_conv_layers[i](inputs=block_conv)

        # Upsample most recent block conv image for both side chains.
        upsampled_block_conv = upsample_generator_image(
            image=block_conv, block_idx=len(permanent_blocks)
        )

        # Growing side chain.
        growing_block_layers = blocks[trans_idx + 1]
        growing_to_rgb_conv_layer = to_rgb_conv_layers[trans_idx + 1]

        # Pass inputs through layer chain.
        block_conv = growing_block_layers[0](inputs=upsampled_block_conv)
        for i in range(1, len(growing_block_layers)):
            block_conv = growing_block_layers[i](inputs=block_conv)
        block_conv = growing_to_rgb_conv_layer(inputs=block_conv)

        # Shrinking side chain.
        shrinking_to_rgb_conv_layer = to_rgb_conv_layers[trans_idx]

        # Pass inputs through layer chain.
        shrinking_to_rgb_conv = shrinking_to_rgb_conv_layer(
            inputs=upsampled_block_conv
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=block_conv * alpha_var,
            y=shrinking_to_rgb_conv * (1.0 - alpha_var),
            name="growth_transition_weighted_sum_{}".format(trans_idx)
        )
        print_obj(
            "\ncreate_base_generator_network",
            "weighted_sum",
            weighted_sum
        )

    return weighted_sum


def create_final_generator_network(X, to_rgb_conv_layers, blocks):
    """Creates base generator network.

    Args:
        X: tensor, input image to generator.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.

    Returns:
        Final network block conv tensor.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Only need the last toRGB conv layer.
        to_rgb_conv_layer = to_rgb_conv_layers[-1]

        # Flatten blocks.
        block_layers = [item for sublist in blocks for item in sublist]

        # Pass inputs through layer chain.
        block_conv = block_layers[0](inputs=X)
        for i in range(1, len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
        to_rgb_conv = to_rgb_conv_layer(inputs=block_conv)
        print_obj(
            "\ncreate_final_generator_network",
            "to_rgb_conv",
            to_rgb_conv
        )

    return to_rgb_conv


def generator_network(Z, alpha_var, mode, params):
    """Creates generator network and returns generated output.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Generated outputs tensor of shape
            [cur_batch_size, height * width * depth].
    """
    print_obj("\ngenerator_network", "Z", Z)

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Create regularizer for dense layer kernel weights.
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["generator_l1_regularization_scale"],
            scale_l2=params["generator_l2_regularization_scale"]
        )

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
            kernel_initializer="he_normal",
            name="projection_layer"
        )
        print_obj("generator_network", "projection", projection)

        # Reshape projection into "image".
        # shape = (
        #     cur_batch_size,
        #     projection_height,
        #     projection_width,
        #     projection_depth
        # )
        base_network = tf.reshape(
            tensor=projection,
            shape=[-1, projection_height, projection_width, projection_depth],
            name="projection_reshaped"
        )
        print_obj("generator_network", "base_network", base_network)

        # Create empty list to hold generator convolutional layer blocks.
        blocks = []

        # Create base convolutional layers, for post-growth.
        blocks.append(
            create_generator_base_conv_layer_block(regularizer, params)
        )

        # Create growth layer blocks.
        for block_idx in range(len(params["growth_num_filters"])):
            blocks.append(
                create_generator_growth_layer_block(
                    block_idx, regularizer, params
                )
            )
        print_obj("generator_network", "blocks", blocks)

        # Create list of toRGB 1x1 conv layers.
        to_rgb_conv_layers = create_generator_to_rgb_layers(
            regularizer, params
        )
        print_obj(
            "generator_network", "to_rgb_conv_layers", to_rgb_conv_layers
        )

        # Switch to case based on number of steps for network creation.
        generated_outputs = tf.switch_case(
            branch_index=tf.cast(
                x=tf.floordiv(
                    x=tf.train.get_or_create_global_step(),
                    y=params["num_steps_until_growth"]
                ),
                dtype=tf.int32),
            branch_fns=[
                lambda: create_base_generator_network(
                    base_network, to_rgb_conv_layers, blocks
                ),
                lambda: create_growth_transition_generator_network(
                    base_network, to_rgb_conv_layers, blocks, alpha_var, 0
                ),
                lambda: create_growth_transition_generator_network(
                    base_network, to_rgb_conv_layers, blocks, alpha_var, 1
                ),
                lambda: create_final_generator_network(
                    base_network, to_rgb_conv_layers, blocks
                ),
            ],
            name="generator_switch_case_generated_outputs"
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
#     generator_regularization_loss = tf.losses.get_regularization_loss(
#         scope="generator",
#         name="generator_regularization_loss"
#     )
    generator_regularization_loss = 0
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
