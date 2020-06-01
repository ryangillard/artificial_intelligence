import tensorflow as tf

from .print_object import print_obj


def create_discriminator_from_rgb_layers(regularizer, params):
    """Creates discriminator fromRGB layers of 1x1 convs.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of fromRGB 1x1 conv layers.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Create list to hold fromRGB 1x1 convs.
        from_rgb_conv_layers = [
            # Create base fromRGB conv 1x1.
            tf.layers.Conv2D(
                filters=params["base_num_filters"][-1],
                kernel_size=1,
                strides=1,
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="base_from_rgb_layers_conv2d"
            )
        ]

        # Create fromRGB 1x1 convs to match generator growth.
        growth_from_rgb_conv_layers = [
            tf.layers.Conv2D(
                filters=params["growth_num_filters"][i][-1],
                kernel_size=1,
                strides=1,
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="growth_from_rgb_layers_conv2d_{}".format(i)
            )
            for i in range(len(params["growth_num_filters"]))
        ]

        # Combine base and growth fromRGB 1x1 convs.
        from_rgb_conv_layers.extend(growth_from_rgb_conv_layers)
        print_obj(
            "\ncreate_discriminator_from_rgb_layers",
            "from_rgb_conv_layers",
            from_rgb_conv_layers
        )

    return from_rgb_conv_layers


def create_discriminator_base_conv_layer_block(regularizer, params):
    """Creates discriminator base conv layer block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of base conv layers.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Create list of base conv layers.
        # Note this is in reverse order of the generator.
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
            for i in range(len(params["base_num_filters"]) - 1, -1, -1)
        ]
        print_obj(
            "\ncreate_discriminator_base_conv_layer_block",
            "base_conv_layers",
            base_conv_layers
        )

    return base_conv_layers


def create_discriminator_growth_layer_block(
        block_idx, regularizer, params):
    """Creates discriminator growth block.

    Args:
        block_idx: int, the current growth block's index.
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of growth block layers.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Create new inner convolutional layers.
        # Note this is in reverse order of the generator.
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
                len(params["growth_num_filters"][block_idx]) - 1, -1, -1
            )
        ]
        print_obj(
            "\ncreate_discriminator_growth_layer_block",
            "conv_layers",
            conv_layers
        )

        # Down sample from 2s X 2s to s X s image.
        downsampled_image_layer = tf.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="growth_downsampled_image_{}".format(
                block_idx
            )
        )
        print_obj(
            "create_discriminator_growth_layer_block",
            "downsampled_image_layer",
            downsampled_image_layer
        )

    return conv_layers + [downsampled_image_layer]


def create_discriminator_growth_transition_downsample_layers(params):
    """Creates discriminator growth transition downsample layers.

    Args:
        params: dict, user passed parameters.

    Returns:
        List of growth transition downsample layers.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Down sample from 2s X 2s to s X s image.
        downsample_layers = [
            tf.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="growth_transition_downsample_layer_{}".format(
                    layer_idx
                )
            )
            for layer_idx in range(1 + len(params["growth_num_filters"]))
        ]
        print_obj(
            "\ncreate_discriminator_growth_transition_downsample_layers",
            "downsample_layers",
            downsample_layers
        )

    return downsample_layers


def create_base_discriminator_network(
        X, from_rgb_conv_layers, blocks, regularizer, params):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        Logits tensor.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Only need the first fromRGB conv layer and block for base network.
        from_rgb_conv_layer = from_rgb_conv_layers[0]
        block_layers = blocks[0]

        # Pass inputs through layer chain.
        block_conv = from_rgb_conv_layer(inputs=X)
        for i in range(len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
        print_obj(
            "\ncreate_base_discriminator_network",
            "block_conv",
            block_conv
        )

        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0],
                params["generator_projection_dims"][1],
                block_conv.get_shape()[-1]]
        )
        print_obj(
            "create_base_discriminator_network",
            "block_conv",
            block_conv
        )

        # Flatten final block conv tensor.
        block_conv_flat = tf.layers.Flatten()(inputs=block_conv)
        print_obj(
            "create_base_discriminator_network",
            "block_conv_flat",
            block_conv_flat
        )

        # Final linear layer for logits.
        logits = tf.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_logits_base"
        )(inputs=block_conv_flat)
        print_obj("create_base_discriminator_network", "logits", logits)

    return logits


def create_growth_transition_discriminator_network(
        X,
        from_rgb_conv_layers,
        blocks,
        transition_downsample_layers,
        alpha_var,
        regularizer,
        params,
        trans_idx):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        transition_downsample_layers: list, downsample layers for transition.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.
        trans_idx: int, index of current growth transition.

    Returns:
        Logits tensor.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Growing side chain.
        growing_from_rgb_conv_layer = from_rgb_conv_layers[trans_idx + 1]
        growing_block_layers = blocks[trans_idx + 1]

        # Pass inputs through layer chain.
        block_conv = growing_from_rgb_conv_layer(inputs=X)
        for i in range(len(growing_block_layers)):
            block_conv = growing_block_layers[i](inputs=block_conv)

        # Shrinking side chain.
        transition_downsample_layer = transition_downsample_layers[trans_idx]
        shrinking_from_rgb_conv_layer = from_rgb_conv_layers[trans_idx]

        # Pass inputs through layer chain.
        transition_downsample = transition_downsample_layer(inputs=X)
        shrinking_from_rgb_conv = shrinking_from_rgb_conv_layer(
            inputs=transition_downsample
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=block_conv * alpha_var,
            y=shrinking_from_rgb_conv * (1.0 - alpha_var),
            name="growth_transition_weighted_sum_{}".format(trans_idx)
        )

        # Permanent blocks.
        permanent_blocks = blocks[0:trans_idx + 1]

        # Reverse order of blocks and flatten.
        permanent_block_layers = [
            item for sublist in permanent_blocks[::-1] for item in sublist
        ]

        # Pass inputs through layer chain.
        block_conv = weighted_sum
        for i in range(len(permanent_block_layers)):
            block_conv = permanent_block_layers[i](inputs=block_conv)
        print_obj(
            "\ncreate_growth_transition_discriminator_network",
            "block_conv",
            block_conv
        )

        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0] * 2 ** (trans_idx + 1),
                params["generator_projection_dims"][1] * 2 ** (trans_idx + 1),
                block_conv.get_shape()[-1]]
        )
        print_obj(
            "create_growth_transition_discriminator_network",
            "block_conv",
            block_conv
        )

        # Flatten final block conv tensor.
        block_conv_flat = tf.layers.Flatten()(inputs=block_conv)
        print_obj(
            "create_growth_transition_discriminator_network",
            "block_conv_flat",
            block_conv_flat
        )

        # Final linear layer for logits.
        logits = tf.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_logits_transition_{}".format(
                trans_idx
            )
        )(inputs=block_conv_flat)
        print_obj(
            "create_growth_transition_discriminator_network", "logits", logits
        )

    return logits


def create_final_discriminator_network(
        X, from_rgb_conv_layers, blocks, regularizer, params):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        Logits tensor.
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Only need the last fromRGB conv layer.
        from_rgb_conv_layer = from_rgb_conv_layers[-1]

        # Reverse order of blocks and flatten.
        block_layers = [item for sublist in blocks[::-1] for item in sublist]

        # Pass inputs through layer chain.
        block_conv = from_rgb_conv_layer(inputs=X)
        for i in range(len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
        print_obj(
            "\ncreate_final_discriminator_network",
            "block_conv",
            block_conv
        )

        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0] * (2 ** len(params["growth_num_filters"])),
                params["generator_projection_dims"][1] * (2 ** len(params["growth_num_filters"])),
                block_conv.get_shape()[-1]]
        )
        print_obj(
            "create_final_discriminator_network",
            "block_conv",
            block_conv
        )

        # Flatten final block conv tensor.
        block_conv_flat = tf.layers.Flatten()(inputs=block_conv)
        print_obj(
            "create_final_discriminator_network",
            "block_conv_flat",
            block_conv_flat
        )

        # Final linear layer for logits.
        logits = tf.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="layers_dense_logits_final"
        )(inputs=block_conv_flat)
        print_obj("create_final_discriminator_network", "logits", logits)

    return logits


def discriminator_network(X, alpha_var, params):
    """Creates discriminator network and returns logits.

    Args:
        X: tensor, image tensors of shape
            [cur_batch_size, height, width, depth].
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Logits tensor of shape [cur_batch_size, 1].
    """
    print_obj("\ndiscriminator_network", "X", X)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Create regularizer for dense layer kernel weights.
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["discriminator_l1_regularization_scale"],
            scale_l2=params["discriminator_l2_regularization_scale"]
        )

        # Create list of fromRGB 1x1 conv layers.
        from_rgb_conv_layers = create_discriminator_from_rgb_layers(
            regularizer, params
        )
        print_obj(
            "discriminator_network",
            "from_rgb_conv_layers",
            from_rgb_conv_layers
        )

        # Create empty list to hold discriminator convolutional layer blocks.
        blocks = []

        # Create base convolutional layers, for post-growth.
        blocks.append(
            create_discriminator_base_conv_layer_block(regularizer, params)
        )

        # Create growth layer blocks.
        for block_idx in range(len(params["growth_num_filters"])):
            blocks.append(
                create_discriminator_growth_layer_block(
                    block_idx, regularizer, params
                )
            )
        print_obj("discriminator_network", "blocks", blocks)

        # Create list of transition downsample layers.
        transition_downsample_layers = (
            create_discriminator_growth_transition_downsample_layers(params)
        )
        print_obj(
            "discriminator_network",
            "transition_downsample_layers",
            transition_downsample_layers
        )

        # Switch to case based on number of steps for network creation.
        logits = tf.switch_case(
            branch_index=tf.cast(
                x=tf.floordiv(
                    x=tf.train.get_or_create_global_step(),
                    y=params["num_steps_until_growth"]
                ),
                dtype=tf.int32),
            branch_fns=[
                lambda: create_base_discriminator_network(
                    X, from_rgb_conv_layers, blocks, regularizer, params
                ),
                lambda: create_growth_transition_discriminator_network(
                    X,
                    from_rgb_conv_layers,
                    blocks,
                    transition_downsample_layers,
                    alpha_var,
                    regularizer,
                    params,
                    0
                ),
                lambda: create_growth_transition_discriminator_network(
                    X,
                    from_rgb_conv_layers,
                    blocks,
                    transition_downsample_layers,
                    alpha_var,
                    regularizer,
                    params,
                    1
                ),
                lambda: create_final_discriminator_network(
                    X, from_rgb_conv_layers, blocks, regularizer, params
                )
            ],
            name="discriminator_switch_case_block_conv"
        )

    return logits


def get_discriminator_loss(generated_logits, real_logits, params):
    """Gets discriminator loss.

    Args:
        generated_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        real_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        params: dict, user passed parameters.

    Returns:
        Tensor of discriminator's total loss of shape [].
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Calculate base discriminator loss.
        discriminator_real_loss = tf.reduce_mean(
            input_tensor=real_logits,
            name="discriminator_real_loss"
        )
        print_obj(
            "\nget_discriminator_loss",
            "discriminator_real_loss",
            discriminator_real_loss
        )

        discriminator_generated_loss = tf.reduce_mean(
            input_tensor=generated_logits,
            name="discriminator_generated_loss"
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_generated_loss",
            discriminator_generated_loss
        )

        discriminator_loss = tf.add(
            x=discriminator_real_loss, y=-discriminator_generated_loss,
            name="discriminator_loss"
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_loss",
            discriminator_loss
        )

        # Get discriminator gradient penalty.
        discriminator_gradients = tf.gradients(
            ys=discriminator_loss,
            xs=tf.trainable_variables(scope="discriminator"),
            name="discriminator_gradients_for_penalty"
        )

        discriminator_gradient_penalty = tf.square(
            x=tf.multiply(
                x=params["discriminator_gradient_penalty_coefficient"],
                y=tf.linalg.global_norm(
                    t_list=discriminator_gradients,
                    name="discriminator_gradients_global_norm"
                ) - 1.0
            ),
            name="discriminator_gradient_penalty"
        )

        discriminator_wasserstein_gp_loss = tf.add(
            x=discriminator_loss,
            y=discriminator_gradient_penalty,
            name="discriminator_wasserstein_gp_loss"
        )

        # Get regularization losses.
#         discriminator_regularization_loss = tf.losses.get_regularization_loss(
#             scope="discriminator",
#             name="discriminator_regularization_loss"
#         )
        discriminator_regularization_loss = 0
        print_obj(
            "get_discriminator_loss",
            "discriminator_regularization_loss",
            discriminator_regularization_loss
        )

        # Combine losses for total losses.
        discriminator_total_loss = tf.math.add(
            x=discriminator_wasserstein_gp_loss,
            y=discriminator_regularization_loss,
            name="discriminator_total_loss"
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_total_loss",
            discriminator_total_loss
        )

    return discriminator_total_loss
