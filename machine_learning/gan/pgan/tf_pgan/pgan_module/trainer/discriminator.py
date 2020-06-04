import tensorflow as tf

from . import regularization
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
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get fromRGB layer properties.
        from_rgb = [
            params["discriminator_from_rgb_layers"][i][0][:]
            for i in range(len(params["discriminator_from_rgb_layers"]))
        ]

        # Create list to hold toRGB 1x1 convs.
        from_rgb_conv_layers = [
            tf.layers.Conv2D(
                filters=from_rgb[i][3],
                kernel_size=from_rgb[i][0:2],
                strides=from_rgb[i][4:6],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="discriminator_from_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                    i,
                    from_rgb[i][0],
                    from_rgb[i][1],
                    from_rgb[i][2],
                    from_rgb[i][3]
                )
            )
            for i in range(len(from_rgb))
        ]
        print_obj(
            "\ncreate_discriminator_from_rgb_layers",
            "from_rgb_conv_layers",
            from_rgb_conv_layers
        )

    return from_rgb_conv_layers


def build_discriminator_from_rgb_layers(from_rgb_conv_layers, params):
    """Creates discriminator fromRGB layers of 1x1 convs.

    Args:
        from_rgb_conv_layers: list, fromGRB con layers.
        params: dict, user passed parameters.

    Returns:
        List of fromRGB 1x1 conv tensors.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get fromRGB layer properties.
        from_rgb = [
            params["discriminator_from_rgb_layers"][i][0][:]
            for i in range(len(params["discriminator_from_rgb_layers"]))
        ]

        # Create list to hold toRGB 1x1 convs.
        from_rgb_conv_tensors = [
            from_rgb_conv_layers[i](
                inputs=tf.zeros(
                    shape=[1] + from_rgb[i][0:3], dtype=tf.float32
                )
            )
            for i in range(len(from_rgb))
        ]
        print_obj(
            "\nbuild_discriminator_from_rgb_layers",
            "from_rgb_conv_tensors",
            from_rgb_conv_tensors
        )

    return from_rgb_conv_tensors


def create_discriminator_base_conv_layer_block(regularizer, params):
    """Creates discriminator base conv layer block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of base conv layers.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["discriminator_base_conv_blocks"][0]

        # Create list of base conv layers.
        base_conv_layers = [
            tf.layers.Conv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="discriminator_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                    i,
                    conv_block[i][0],
                    conv_block[i][1],
                    conv_block[i][2],
                    conv_block[i][3]
                )
            )
            for i in range(len(conv_block) - 1)
        ]

        # Have valid padding for layer just before flatten and logits.
        base_conv_layers.append(
            tf.layers.Conv2D(
                filters=conv_block[-1][3],
                kernel_size=conv_block[-1][0:2],
                strides=conv_block[-1][4:6],
                padding="valid",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="discriminator_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                    len(conv_block) - 1,
                    conv_block[-1][0],
                    conv_block[-1][1],
                    conv_block[-1][2],
                    conv_block[-1][3]
                )
            )
        )
        print_obj(
            "\ncreate_discriminator_base_conv_layer_block",
            "base_conv_layers",
            base_conv_layers
        )

    return base_conv_layers


def build_discriminator_base_conv_layer_block(base_conv_layers, params):
    """Creates discriminator base conv layer block.

    Args:
        base_conv_layers: list, base conv block's layers.
        params: dict, user passed parameters.

    Returns:
        List of base conv tensors.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["discriminator_base_conv_blocks"][0]

        # Create list of base conv layer tensors.
        base_conv_tensors = [
            base_conv_layers[i](
                inputs=tf.zeros(
                    shape=[1] + conv_block[i][0:3], dtype=tf.float32
                )
            )
            for i in range(len(conv_block))
        ]
        print_obj(
            "\nbase_conv_layers",
            "base_conv_tensors",
            base_conv_tensors
        )

    return base_conv_tensors


def create_discriminator_growth_layer_block(regularizer, params, block_idx):
    """Creates discriminator growth block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.
        block_idx: int, the current growth block's index.

    Returns:
        List of growth block layers.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["discriminator_growth_conv_blocks"][block_idx]

        # Create new inner convolutional layers.
        conv_layers = [
            tf.layers.Conv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="discriminator_growth_layers_conv2d_{}_{}_{}x{}_{}_{}".format(
                    block_idx,
                    i,
                    conv_block[i][0],
                    conv_block[i][1],
                    conv_block[i][2],
                    conv_block[i][3]
                )
            )
            for i in range(len(conv_block))
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
            name="discriminator_growth_downsampled_image_{}".format(
                block_idx
            )
        )
        print_obj(
            "create_discriminator_growth_layer_block",
            "downsampled_image_layer",
            downsampled_image_layer
        )

    return conv_layers + [downsampled_image_layer]


def build_discriminator_growth_layer_block(conv_layers, params, block_idx):
    """Creates discriminator growth block.

    Args:
        list, the current growth block's conv layers.
        params: dict, user passed parameters.
        block_idx: int, the current growth block's index.

    Returns:
        List of growth block layers.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["discriminator_growth_conv_blocks"][block_idx]

        # Create new inner convolutional layers.
        conv_tensors = [
            conv_layers[i](
                inputs=tf.zeros(
                    shape=[1] + conv_block[i][0:3], dtype=tf.float32
                )
            )
            for i in range(len(conv_block))
        ]
        print_obj(
            "\nbuild_discriminator_growth_layer_block",
            "conv_tensors",
            conv_tensors
        )

        # Down sample from 2s X 2s to s X s image.
        downsampled_image_tensor = tf.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="discriminator_growth_downsampled_image_{}".format(
                block_idx
            )
        )(inputs=conv_tensors[-1])
        print_obj(
            "build_discriminator_growth_layer_block",
            "downsampled_image_tensor",
            downsampled_image_tensor
        )

    return conv_tensors + [downsampled_image_tensor]


def create_discriminator_growth_transition_downsample_layers(params):
    """Creates discriminator growth transition downsample layers.

    Args:
        params: dict, user passed parameters.

    Returns:
        List of growth transition downsample layers.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Down sample from 2s X 2s to s X s image.
        downsample_layers = [
            tf.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="discriminator_growth_transition_downsample_layer_{}".format(
                    layer_idx
                )
            )
            for layer_idx in range(
                1 + len(params["discriminator_growth_conv_blocks"])
            )
        ]
        print_obj(
            "\ncreate_discriminator_growth_transition_downsample_layers",
            "downsample_layers",
            downsample_layers
        )

    return downsample_layers


def create_discriminator_logits_layer(regularizer):
    """Creates discriminator flatten and logits layer.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.

    Returns:
        Flatten and logits layers of discriminator.
    """
    with tf.variable_scope(
        name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Flatten layer to get final block conv tensor ready for dense layer.
        flatten_layer = tf.layers.Flatten(name="discriminator_flatten_layer")
        print_obj(
            "\ncreate_discriminator_logits_layer",
            "flatten_layer",
            flatten_layer
        )

        # Final linear layer for logits.
        logits_layer = tf.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=regularizer,
            name="discriminator_layers_dense_logits"
        )
        print_obj(
            "create_growth_transition_discriminator_network",
            "logits_layer",
            logits_layer
        )

    return flatten_layer, logits_layer


def build_discriminator_logits_layer(flatten_layer, logits_layer, params):
    """Builds flatten and logits layer internals using call.

    Args:
        flatten_layer: `Flatten` layer.
        logits_layer: `Dense` layer for logits.
        params: dict, user passed parameters.

    Returns:
        Final logits tensor of discriminator.
    """
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        block_conv_size = params["discriminator_base_conv_blocks"][-1][-1][3]

        # Flatten final block conv tensor.
        block_conv_flat = flatten_layer(
            inputs=tf.zeros(
                shape=[1, 1, 1, block_conv_size],
                dtype=tf.float32
            )
        )
        print_obj(
            "build_discriminator_logits_layer",
            "block_conv_flat",
            block_conv_flat
        )

        # Final linear layer for logits.
        logits = logits_layer(inputs=block_conv_flat)
        print_obj("build_discriminator_logits_layer", "logits", logits)

    return logits


def use_discriminator_logits_layer(
        block_conv, flatten_layer, logits_layer, params):
    """Uses flatten and logits layers to get logits tensor.

    Args:
        block_conv: tensor, output of last conv layer of discriminator.
        flatten_layer: `Flatten` layer.
        logits_layer: `Dense` layer for logits.
        params: dict, user passed parameters.

    Returns:
        Final logits tensor of discriminator.
    """
    print_obj("\nuse_discriminator_logits_layer", "block_conv", block_conv)
    # Set shape to remove ambiguity for dense layer.
    block_conv.set_shape(
        [
            block_conv.get_shape()[0],
            params["generator_projection_dims"][0] / 4,
            params["generator_projection_dims"][1] / 4,
            block_conv.get_shape()[-1]]
    )
    print_obj("use_discriminator_logits_layer", "block_conv", block_conv)

    with tf.variable_scope(name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Flatten final block conv tensor.
        block_conv_flat = flatten_layer(inputs=block_conv)
        print_obj(
            "use_discriminator_logits_layer",
            "block_conv_flat",
            block_conv_flat
        )

        # Final linear layer for logits.
        logits = logits_layer(inputs=block_conv_flat)
        print_obj("use_discriminator_logits_layer", "logits", logits)

    return logits


def create_base_discriminator_network(
        X, from_rgb_conv_layers, blocks, flatten_layer, logits_layer, params):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        flatten_layer: `Flatten` layer.
        logits_layer: `Dense` layer for logits.
        params: dict, user passed parameters.

    Returns:
        Final logits tensor of discriminator.
    """
    print_obj("\ncreate_base_discriminator_network", "X", X)
    with tf.variable_scope(name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Only need the first fromRGB conv layer and block for base network.
        from_rgb_conv_layer = from_rgb_conv_layers[0]
        block_layers = blocks[0]

        # Pass inputs through layer chain.
        from_rgb_conv = from_rgb_conv_layer(inputs=X)
        print_obj(
            "create_base_discriminator_network",
            "from_rgb_conv",
            from_rgb_conv
        )

        block_conv = from_rgb_conv
        for i in range(len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
            print_obj(
                "create_base_discriminator_network", "block_conv", block_conv
            )

        # Get logits now.
        logits = use_discriminator_logits_layer(
            block_conv=block_conv,
            flatten_layer=flatten_layer,
            logits_layer=logits_layer,
            params=params
        )
        print_obj("create_base_discriminator_network", "logits", logits)

    return logits


def create_growth_transition_discriminator_network(
        X,
        from_rgb_conv_layers,
        blocks,
        transition_downsample_layers,
        flatten_layer,
        logits_layer,
        alpha_var,
        params,
        trans_idx):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        transition_downsample_layers: list, downsample layers for transition.
        flatten_layer: `Flatten` layer.
        logits_layer: `Dense` layer for logits.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        trans_idx: int, index of current growth transition.

    Returns:
        Final logits tensor of discriminator.
    """
    print_obj(
        "\nEntered create_growth_transition_discriminator_network",
        "trans_idx",
        trans_idx
    )
    print_obj("create_growth_transition_discriminator_network", "X", X)
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Growing side chain.
        growing_from_rgb_conv_layer = from_rgb_conv_layers[trans_idx + 1]
        growing_block_layers = blocks[trans_idx + 1]

        # Pass inputs through layer chain.
        growing_block_conv = growing_from_rgb_conv_layer(inputs=X)
        print_obj(
            "\ncreate_growth_transition_discriminator_network",
            "growing_block_conv",
            growing_block_conv
        )
        for i in range(len(growing_block_layers)):
            growing_block_conv = growing_block_layers[i](
                inputs=growing_block_conv
            )
            print_obj(
                "create_growth_transition_discriminator_network",
                "growing_block_conv",
                growing_block_conv
            )

        # Shrinking side chain.
        transition_downsample_layer = transition_downsample_layers[trans_idx]
        shrinking_from_rgb_conv_layer = from_rgb_conv_layers[trans_idx]

        # Pass inputs through layer chain.
        transition_downsample = transition_downsample_layer(inputs=X)
        print_obj(
            "create_growth_transition_discriminator_network",
            "transition_downsample",
            transition_downsample
        )
        shrinking_from_rgb_conv = shrinking_from_rgb_conv_layer(
            inputs=transition_downsample
        )
        print_obj(
            "create_growth_transition_discriminator_network",
            "shrinking_from_rgb_conv",
            shrinking_from_rgb_conv
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=growing_block_conv * alpha_var,
            y=shrinking_from_rgb_conv * (1.0 - alpha_var),
            name="growth_transition_weighted_sum_{}".format(trans_idx)
        )
        print_obj(
            "create_growth_transition_discriminator_network",
            "weighted_sum",
            weighted_sum
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
                "create_growth_transition_discriminator_network",
                "block_conv",
                block_conv
            )

        # Get logits now.
        logits = use_discriminator_logits_layer(
            block_conv, flatten_layer, logits_layer, params
        )
        print_obj(
            "create_growth_transition_discriminator_network", "logits", logits
        )

    return logits


def create_final_discriminator_network(
        X, from_rgb_conv_layers, blocks, flatten_layer, logits_layer, params):
    """Creates base discriminator network.

    Args:
        X: tensor, input image to discriminator.
        from_rgb_conv_layers: list, fromRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        flatten_layer: `Flatten` layer.
        logits_layer: `Dense` layer for logits.
        params: dict, user passed parameters.

    Returns:
        Final logits tensor of discriminator.
    """
    print_obj("\ncreate_final_discriminator_network", "X", X)
    with tf.variable_scope(
            name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        # Only need the last fromRGB conv layer.
        from_rgb_conv_layer = from_rgb_conv_layers[-1]

        # Reverse order of blocks and flatten.
        block_layers = [item for sublist in blocks[::-1] for item in sublist]

        # Pass inputs through layer chain.
        block_conv = from_rgb_conv_layer(inputs=X)
        print_obj(
            "\ncreate_final_discriminator_network",
            "block_conv",
            block_conv
        )

        for i in range(len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
            print_obj(
                "create_final_discriminator_network", "block_conv", block_conv
            )

        # Get logits now.
        logits = use_discriminator_logits_layer(
            block_conv=block_conv,
            flatten_layer=flatten_layer,
            logits_layer=logits_layer,
            params=params
        )
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

    # Create regularizer for layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["discriminator_l1_regularization_scale"],
        scale_l2=params["discriminator_l2_regularization_scale"]
    )

    # Create fromRGB 1x1 conv layers.
    from_rgb_conv_layers = create_discriminator_from_rgb_layers(
        regularizer=regularizer, params=params
    )
    print_obj(
        "discriminator_network",
        "from_rgb_conv_layers",
        from_rgb_conv_layers
    )

    # Build fromRGB 1x1 conv layers internals through call.
    from_rgb_conv_tensors = build_discriminator_from_rgb_layers(
        from_rgb_conv_layers=from_rgb_conv_layers, params=params
    )
    print_obj(
        "discriminator_network",
        "from_rgb_conv_tensors",
        from_rgb_conv_tensors
    )

    with tf.control_dependencies(control_inputs=from_rgb_conv_tensors):
        # Create empty list to hold discriminator convolutional layer blocks.
        block_layers = []
        block_tensors = []

        # Create base convolutional block's layers, for post-growth.
        block_layers.append(
            create_discriminator_base_conv_layer_block(
                regularizer=regularizer, params=params
            )
        )

        # Create base convolutional block's layer internals using call.
        block_tensors.append(
            build_discriminator_base_conv_layer_block(
                base_conv_layers=block_layers[0], params=params
            )
        )

        # Create growth layer blocks.
        for block_idx in range(
           len(params["discriminator_growth_conv_blocks"])):
            block_layers.append(
                create_discriminator_growth_layer_block(
                    regularizer=regularizer,
                    params=params,
                    block_idx=block_idx
                )
            )
        print_obj("discriminator_network", "block_layers", block_layers)

        # Build growth layer block internals through call.
        for block_idx in range(
           len(params["discriminator_growth_conv_blocks"])):
            block_tensors.append(
                build_discriminator_growth_layer_block(
                    conv_layers=block_layers[block_idx + 1],
                    params=params,
                    block_idx=block_idx
                )
            )

        # Flatten block tensor lists of lists into list.
        block_tensors = [item for sublist in block_tensors for item in sublist]
        print_obj("discriminator_network", "block_tensors", block_tensors)

        with tf.control_dependencies(control_inputs=block_tensors):
            # Create list of transition downsample layers.
            transition_downsample_layers = (
                create_discriminator_growth_transition_downsample_layers(
                    params=params
                )
            )
            print_obj(
                "discriminator_network",
                "transition_downsample_layers",
                transition_downsample_layers
            )

            # Create flatten and logits layers.
            flatten_layer, logits_layer = create_discriminator_logits_layer(
                regularizer=regularizer
            )

            # Build logits layer internals using call.
            logits_tensor = build_discriminator_logits_layer(
                flatten_layer=flatten_layer,
                logits_layer=logits_layer,
                params=params
            )

            with tf.control_dependencies(control_inputs=[logits_tensor]):
                # Get discriminator's logits output tensor.
                train_steps = params["train_steps"]
                num_steps_until_growth = params["num_steps_until_growth"]
                num_stages = train_steps // num_steps_until_growth
                if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
                    print(
                        "\ndiscriminator_network: NOT GOING TO GROW, SKIP SWITCH CASE!"
                    )
                    # If never going to grow, no sense using the switch case.
                    # 4x4
                    logits = create_base_discriminator_network(
                        X=X,
                        from_rgb_conv_layers=from_rgb_conv_layers,
                        blocks=block_layers,
                        flatten_layer=flatten_layer,
                        logits_layer=logits_layer,
                        params=params
                    )
                else:
                    # Find index based on global step and growth frequency.
                    growth_index = tf.cast(
                        x=tf.floordiv(
                            x=tf.train.get_or_create_global_step(),
                            y=params["num_steps_until_growth"],
                            name="discriminator_global_step_floordiv"
                        ),
                        dtype=tf.int32,
                        name="discriminator_growth_index"
                    )

                    # Switch to case based on number of steps to get logits.
                    logits = tf.switch_case(
                        branch_index=growth_index,
                        branch_fns=[
                            # 4x4
                            lambda: create_base_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                params=params
                            ),
                            # 8x8
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=0
                            ),
                            # 16x16
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=1
                            ),
                            # 32x32
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=2
                            ),
                            # 64x64
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=3
                            ),
                            # 128x128
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=4
                            ),
                            # 256x256
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=5
                            ),
                            # 512x512
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=6
                            ),
                            # 1024x1024
                            lambda: create_growth_transition_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                transition_downsample_layers=transition_downsample_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                alpha_var=alpha_var,
                                params=params,
                                trans_idx=7
                            ),
                            # 1024x1024
                            lambda: create_final_discriminator_network(
                                X=X,
                                from_rgb_conv_layers=from_rgb_conv_layers,
                                blocks=block_layers,
                                flatten_layer=flatten_layer,
                                logits_layer=logits_layer,
                                params=params
                            )
                        ],
                        name="discriminator_switch_case_logits"
                    )

    return logits


def get_discriminator_loss(fake_logits, real_logits, params):
    """Gets discriminator loss.

    Args:
        fake_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        real_logits: tensor, shape of
            [cur_batch_size, height * width * depth].
        params: dict, user passed parameters.

    Returns:
        Tensor of discriminator's total loss of shape [].
    """
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
        input_tensor=fake_logits,
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

    # Get discriminator gradient penalty loss.
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

    # Get discriminator Wasserstein GP loss.
    discriminator_wasserstein_gp_loss = tf.add(
        x=discriminator_loss,
        y=discriminator_gradient_penalty,
        name="discriminator_wasserstein_gp_loss"
    )

    # Get discriminator regularization losses.
    discriminator_reg_loss = regularization.get_regularization_loss(
        params=params, scope="discriminator"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_reg_loss",
        discriminator_reg_loss
    )

    # Combine losses for total losses.
    discriminator_total_loss = tf.math.add(
        x=discriminator_wasserstein_gp_loss,
        y=discriminator_reg_loss,
        name="discriminator_total_loss"
    )
    print_obj(
        "get_discriminator_loss",
        "discriminator_total_loss",
        discriminator_total_loss
    )

    return discriminator_total_loss
