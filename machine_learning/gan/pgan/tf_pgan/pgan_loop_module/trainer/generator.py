import tensorflow as tf

from . import regularization
from . import utils
from .print_object import print_obj


def create_generator_projection_layer(regularizer, params):
    """Creates generator projection from noise latent vector.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        Latent vector projection `Dense` layer.
    """
    # Project latent vectors.
    projection_height = params["generator_projection_dims"][0]
    projection_width = params["generator_projection_dims"][1]
    projection_depth = params["generator_projection_dims"][2]

    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # shape = (
        #     cur_batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection_layer = tf.layers.Dense(
            units=projection_height * projection_width * projection_depth,
            activation=tf.nn.leaky_relu,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name="generator_projection_layer"
        )
        print_obj(
            "create_generator_projection_layer",
            "projection_layer",
            projection_layer
        )

    return projection_layer


def build_generator_projection_layer(projection_layer, params):
    """Builds generator projection layer internals using call.

    Args:
        projection_layer: `Dense` layer for projection of noise into image.
        params: dict, user passed parameters.

    Returns:
        Latent vector projection tensor.
    """
    # Project latent vectors.
    projection_height = params["generator_projection_dims"][0]
    projection_width = params["generator_projection_dims"][1]
    projection_depth = params["generator_projection_dims"][2]

    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # shape = (
        #     cur_batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection_tensor = projection_layer(
            inputs=tf.zeros(
                shape=[1, params["latent_size"]], dtype=tf.float32
            )
        )
        print_obj(
            "\nbuild_generator_projection_layer",
            "projection_tensor",
            projection_tensor
        )

    return projection_tensor

def use_generator_projection_layer(Z, projection_layer, params):
    """Uses projection layer to convert random noise vector into an image.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        projection_layer: `Dense` layer for projection of noise into image.
        params: dict, user passed parameters.

    Returns:
        Latent vector projection tensor.
    """
    # Project latent vectors.
    projection_height = params["generator_projection_dims"][0]
    projection_width = params["generator_projection_dims"][1]
    projection_depth = params["generator_projection_dims"][2]

    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # shape = (
        #     cur_batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection_tensor = projection_layer(inputs=Z)
        print_obj(
            "\nuse_generator_projection_layer", "projection_tensor", projection_tensor
        )

    # Reshape projection into "image".
    # shape = (
    #     cur_batch_size,
    #     projection_height,
    #     projection_width,
    #     projection_depth
    # )
    projection_tensor_reshaped = tf.reshape(
        tensor=projection_tensor,
        shape=[-1, projection_height, projection_width, projection_depth],
        name="generator_projection_reshaped"
    )
    print_obj(
        "use_generator_projection_layer",
        "projection_tensor_reshaped",
        projection_tensor_reshaped
    )

    return projection_tensor_reshaped


def create_generator_base_conv_layer_block(regularizer, params):
    """Creates generator base conv layer block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of base conv layers.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["generator_base_conv_blocks"][0]

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
                name="generator_base_layers_conv2d_{}_{}x{}_{}_{}".format(
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
            "\ncreate_generator_base_conv_layer_block",
            "base_conv_layers",
            base_conv_layers
        )

    return base_conv_layers


def build_generator_base_conv_layer_block(base_conv_layers, params):
    """Builds generator base conv layer block internals using call.

    Args:
        base_conv_layers: list, the base block's conv layers.
        params: dict, user passed parameters.

    Returns:
        List of base conv tensors.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["generator_base_conv_blocks"][0]

        # Create list of base conv layers.
        base_conv_tensors = [
            base_conv_layers[i](
                inputs=tf.zeros(
                    shape=[1] + conv_block[i][0:3], dtype=tf.float32
                )
            )
            for i in range(len(conv_block))
        ]
        print_obj(
            "\nbuild_generator_base_conv_layer_block",
            "base_conv_tensors",
            base_conv_tensors
        )

    return base_conv_tensors


def create_generator_growth_layer_block(regularizer, params, block_idx):
    """Creates generator growth block.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.
        block_idx: int, the current growth block's index.

    Returns:
        List of growth block layers.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["generator_growth_conv_blocks"][block_idx]

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
                name="generator_growth_layers_conv2d_{}_{}_{}x{}_{}_{}".format(
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
            "\ncreate_generator_growth_layer_block", "conv_layers", conv_layers
        )

    return conv_layers


def build_generator_growth_layer_block(conv_layers, params, block_idx):
    """Builds generator growth block internals through call.

    Args:
        conv_layers: list, the current growth block's conv layers.
        params: dict, user passed parameters.
        block_idx: int, the current growth block's index.

    Returns:
        List of growth block tensors.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get conv block layer properties.
        conv_block = params["generator_growth_conv_blocks"][block_idx]

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
            "\nbuild_generator_growth_layer_block",
            "conv_tensors",
            conv_tensors
        )

    return conv_tensors


def create_generator_to_rgb_layers(regularizer, params):
    """Creates generator toRGB layers of 1x1 convs.

    Args:
        regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        params: dict, user passed parameters.

    Returns:
        List of toRGB 1x1 conv layers.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get toRGB layer properties.
        to_rgb = [
            params["generator_to_rgb_layers"][i][0][:]
            for i in range(len(params["generator_to_rgb_layers"]))
        ]

        # Create list to hold toRGB 1x1 convs.
        to_rgb_conv_layers = [
            tf.layers.Conv2D(
                filters=to_rgb[i][3],
                kernel_size=to_rgb[i][0:2],
                strides=to_rgb[i][4:6],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name="generator_to_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                    i, to_rgb[i][0], to_rgb[i][1], to_rgb[i][2], to_rgb[i][3]
                )
            )
            for i in range(len(to_rgb))
        ]
        print_obj(
            "\ncreate_generator_to_rgb_layers",
            "to_rgb_conv_layers",
            to_rgb_conv_layers
        )

    return to_rgb_conv_layers


def build_generator_to_rgb_layers(to_rgb_conv_layers, params):
    """Builds generator toRGB layers of 1x1 convs internals through call.

    Args:
        to_rgb_conv_layers: list, toRGB conv layers.
        params: dict, user passed parameters.

    Returns:
        List of toRGB 1x1 conv tensors.
    """
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Get toRGB layer properties.
        to_rgb = [
            params["generator_to_rgb_layers"][i][0][:]
            for i in range(len(params["generator_to_rgb_layers"]))
        ]

        # Create list to hold toRGB 1x1 convs.
        to_rgb_conv_tensors = [
            to_rgb_conv_layers[i](
                inputs=tf.zeros(shape=[1] + to_rgb[i][0:3], dtype=tf.float32))
            for i in range(len(to_rgb))
        ]
        print_obj(
            "\nbuild_generator_to_rgb_layers",
            "to_rgb_conv_tensors",
            to_rgb_conv_tensors
        )

    return to_rgb_conv_tensors


def upsample_generator_image(image, original_image_size, block_idx):
    """Upsamples generator image.

    Args:
        image: tensor, image created by generator conv block.
        original_image_size: list, the height and width dimensions of the
            original image before any growth.
        block_idx: int, index of the current generator growth block.

    Returns:
        Upsampled image tensor.
    """
    # Upsample from s X s to 2s X 2s image.
    upsampled_image = tf.image.resize(
        images=image,
        size=tf.convert_to_tensor(
            value=original_image_size,
            dtype=tf.int32,
            name="upsample_generator_image_original_image_size"
        ) * 2 ** block_idx,
        method="nearest",
        name="generator_growth_upsampled_image_{}_{}x{}_{}x{}".format(
            block_idx,
            original_image_size[0] * 2 ** (block_idx - 1),
            original_image_size[1] * 2 ** (block_idx - 1),
            original_image_size[0] * 2 ** block_idx,
            original_image_size[1] * 2 ** block_idx
        )
    )
    print_obj(
        "\nupsample_generator_image",
        "upsampled_image",
        upsampled_image
    )

    return upsampled_image


def create_base_generator_network(
        Z, projection_layer, to_rgb_conv_layers, blocks, params):
    """Creates base generator network.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        projection_layer: `Dense` layer for projection of noise into image.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        params: dict, user passed parameters.

    Returns:
        Final network block conv tensor.
    """
    print_obj("\ncreate_base_generator_network", "Z", Z)
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Project latent noise vectors into image.
        projection = use_generator_projection_layer(
            Z=Z, projection_layer=projection_layer, params=params
        )
        print_obj("create_base_generator_network", "projection", projection)

        # Only need the first block and toRGB conv layer for base network.
        block_layers = blocks[0]
        to_rgb_conv_layer = to_rgb_conv_layers[0]

        # Pass inputs through layer chain.
        block_conv = block_layers[0](inputs=projection)
        print_obj("create_base_generator_network", "block_conv_0", block_conv)

        for i in range(1, len(block_layers)):
            block_conv = block_layers[i](inputs=block_conv)
            print_obj(
                "create_base_generator_network",
                "block_conv_{}".format(i),
                block_conv
            )
        to_rgb_conv = to_rgb_conv_layer(inputs=block_conv)
        print_obj("create_base_generator_network", "to_rgb_conv", to_rgb_conv)

    return to_rgb_conv


def create_growth_transition_generator_network(
        Z,
        projection_layer,
        to_rgb_conv_layers,
        blocks,
        original_image_size,
        alpha_var,
        params,
        trans_idx):
    """Creates base generator network.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        projection_layer: `Dense` layer for projection of noise into image.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        original_image_size: list, the height and width dimensions of the
            original image before any growth.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        trans_idx: int, index of current growth transition.

    Returns:
        Final network block conv tensor.
    """
    print_obj(
        "\nEntered create_growth_transition_generator_network",
        "trans_idx",
        trans_idx
    )
    print_obj("create_growth_transition_generator_network", "Z", Z)
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Project latent noise vectors into image.
        projection = use_generator_projection_layer(
            Z=Z, projection_layer=projection_layer, params=params
        )
        print_obj(
            "create_growth_transition_generator_network",
            "projection",
            projection
        )

        # Permanent blocks.
        permanent_blocks = blocks[0:trans_idx + 1]

        # Base block doesn't need any upsampling so it's handled differently.
        base_block_conv_layers = permanent_blocks[0]

        # Pass inputs through layer chain.
        block_conv = base_block_conv_layers[0](inputs=projection)
        print_obj(
            "create_growth_transition_generator_network",
            "base_block_conv_{}_0".format(trans_idx),
            block_conv
        )
        for i in range(1, len(base_block_conv_layers)):
            block_conv = base_block_conv_layers[i](inputs=block_conv)
            print_obj(
                "create_growth_transition_generator_network",
                "base_block_conv_{}_{}".format(trans_idx, i),
                block_conv
            )

        # Growth blocks require first the prev conv layer's image upsampled.
        for i in range(1, len(permanent_blocks)):
            # Upsample previous block's image.
            block_conv = upsample_generator_image(
                image=block_conv,
                original_image_size=original_image_size,
                block_idx=i
            )
            print_obj(
                "create_growth_transition_generator_network",
                "upsample_generator_image_block_conv_{}_{}".format(
                    trans_idx, i
                ),
                block_conv
            )

            block_conv_layers = permanent_blocks[i]
            for j in range(0, len(block_conv_layers)):
                block_conv = block_conv_layers[j](inputs=block_conv)
                print_obj(
                    "create_growth_transition_generator_network",
                    "block_conv_{}_{}_{}".format(trans_idx, i, j),
                    block_conv
                )

        # Upsample most recent block conv image for both side chains.
        upsampled_block_conv = upsample_generator_image(
            image=block_conv,
            original_image_size=original_image_size,
            block_idx=len(permanent_blocks)
        )
        print_obj(
            "create_growth_transition_generator_network",
            "upsampled_block_conv_{}".format(trans_idx),
            upsampled_block_conv
        )

        # Growing side chain.
        growing_block_layers = blocks[trans_idx + 1]
        growing_to_rgb_conv_layer = to_rgb_conv_layers[trans_idx + 1]

        # Pass inputs through layer chain.
        block_conv = growing_block_layers[0](inputs=upsampled_block_conv)
        print_obj(
            "create_growth_transition_generator_network",
            "growing_block_conv_{}_0".format(trans_idx),
            block_conv
        )
        for i in range(1, len(growing_block_layers)):
            block_conv = growing_block_layers[i](inputs=block_conv)
            print_obj(
                "create_growth_transition_generator_network",
                "growing_block_conv_{}_{}".format(trans_idx, i),
                block_conv
            )
        growing_to_rgb_conv = growing_to_rgb_conv_layer(inputs=block_conv)
        print_obj(
            "create_growth_transition_generator_network",
            "growing_to_rgb_conv_{}".format(trans_idx),
            growing_to_rgb_conv
        )

        # Shrinking side chain.
        shrinking_to_rgb_conv_layer = to_rgb_conv_layers[trans_idx]

        # Pass inputs through layer chain.
        shrinking_to_rgb_conv = shrinking_to_rgb_conv_layer(
            inputs=upsampled_block_conv
        )
        print_obj(
            "create_growth_transition_generator_network",
            "shrinking_to_rgb_conv_{}".format(trans_idx),
            shrinking_to_rgb_conv
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=growing_to_rgb_conv * alpha_var,
            y=shrinking_to_rgb_conv * (1.0 - alpha_var),
            name="growth_transition_weighted_sum_{}".format(trans_idx)
        )
        print_obj(
            "create_growth_transition_generator_network",
            "weighted_sum_{}".format(trans_idx),
            weighted_sum
        )

    return weighted_sum


def create_final_generator_network(
        Z, projection_layer, to_rgb_conv_layers, blocks, original_image_size, params):
    """Creates base generator network.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        projection_layer: `Dense` layer for projection of noise into image.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        blocks: list, lists of block layers for each block.
        original_image_size: list, the height and width dimensions of the
            original image before any growth.
        params: dict, user passed parameters.

    Returns:
        Final network block conv tensor.
    """
    print_obj("\ncreate_final_generator_network", "Z", Z)
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        # Project latent noise vectors into image.
        projection = use_generator_projection_layer(
            Z=Z, projection_layer=projection_layer, params=params
        )
        print_obj("create_final_generator_network", "projection", projection)

        # Base block doesn't need any upsampling so it's handled differently.
        base_block_conv_layers = blocks[0]

        # Pass inputs through layer chain.
        block_conv = base_block_conv_layers[0](inputs=projection)
        print_obj(
            "\ncreate_final_generator_network",
            "base_block_conv",
            block_conv
        )

        for i in range(1, len(base_block_conv_layers)):
            block_conv = base_block_conv_layers[i](inputs=block_conv)
            print_obj(
                "create_final_generator_network",
                "base_block_conv_{}".format(i),
                block_conv
            )

        # Growth blocks require first the prev conv layer's image upsampled.
        for i in range(1, len(blocks)):
            # Upsample previous block's image.
            block_conv = upsample_generator_image(
                image=block_conv,
                original_image_size=original_image_size,
                block_idx=i
            )
            print_obj(
                "create_final_generator_network",
                "upsample_generator_image_block_conv_{}".format(i),
                block_conv
            )

            block_conv_layers = blocks[i]
            for j in range(0, len(block_conv_layers)):
                block_conv = block_conv_layers[j](inputs=block_conv)
                print_obj(
                    "create_final_generator_network",
                    "block_conv_{}_{}".format(i, j),
                    block_conv
                )

        # Only need the last toRGB conv layer.
        to_rgb_conv_layer = to_rgb_conv_layers[-1]

        # Pass inputs through layer chain.
        to_rgb_conv = to_rgb_conv_layer(inputs=block_conv)
        print_obj(
            "create_final_generator_network", "to_rgb_conv", to_rgb_conv
        )

    return to_rgb_conv


def generator_network(Z, alpha_var, params):
    """Creates generator network and returns generated output.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Generated outputs tensor of shape
            [cur_batch_size, height * width * depth].
    """
    print_obj("\ngenerator_network", "Z", Z)

    # Create regularizer for layer kernel weights.
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=params["generator_l1_regularization_scale"],
        scale_l2=params["generator_l2_regularization_scale"]
    )

    # Create projection dense layer to turn random noise vector into image.
    projection_layer = create_generator_projection_layer(
        regularizer=regularizer, params=params
    )

    # Build projection layer internals using call.
    projection_tensor = build_generator_projection_layer(
        projection_layer=projection_layer, params=params
    )

    with tf.control_dependencies(control_inputs=[projection_tensor]):
        # Create empty lists to hold generator convolutional layer/tensor blocks.
        block_layers = []
        block_tensors = []

        # Create base convolutional layers, for post-growth.
        block_layers.append(
            create_generator_base_conv_layer_block(
                regularizer=regularizer, params=params
            )
        )

        # Build base convolutional layer block's internals using call.
        block_tensors.append(
            build_generator_base_conv_layer_block(
                base_conv_layers=block_layers[0], params=params
            )
        )

        # Create growth block layers.
        for block_idx in range(len(params["generator_growth_conv_blocks"])):
            block_layers.append(
                create_generator_growth_layer_block(
                    regularizer=regularizer, params=params, block_idx=block_idx
                )
            )
        print_obj("generator_network", "block_layers", block_layers)

        # Build growth block layer internals through call.
        for block_idx in range(len(params["generator_growth_conv_blocks"])):
            block_tensors.append(
                build_generator_growth_layer_block(
                    conv_layers=block_layers[block_idx + 1],
                    params=params,
                    block_idx=block_idx
                )
            )

        # Flatten block tensor lists of lists into list.
        block_tensors = [item for sublist in block_tensors for item in sublist]
        print_obj("generator_network", "block_tensors", block_tensors)

        with tf.control_dependencies(control_inputs=block_tensors):
            # Create toRGB 1x1 conv layers.
            to_rgb_conv_layers = create_generator_to_rgb_layers(
                regularizer=regularizer, params=params
            )
            print_obj(
                "generator_network", "to_rgb_conv_layers", to_rgb_conv_layers
            )

            # Build toRGB 1x1 conv layer internals through call.
            to_rgb_conv_tensors = build_generator_to_rgb_layers(
                to_rgb_conv_layers=to_rgb_conv_layers, params=params
            )
            print_obj(
                "generator_network", "to_rgb_conv_tensors", to_rgb_conv_tensors
            )

            with tf.control_dependencies(control_inputs=to_rgb_conv_tensors):
                # Get original image size to use for setting image shape.
                original_image_size = params["generator_projection_dims"][0:2]

                # Create list of function calls for each training stage.
                generated_outputs_list = utils.LazyList(
                    [
                        # 4x4
                        lambda: create_base_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            params=params
                        ),
                        # 8x8
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=0
                        ),
                        # 16x16
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=1
                        ),
                        # 32x32
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=2
                        ),
                        # 64x64
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=3
                        ),
                        # 128x128
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=4
                        ),
                        # 256x256
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=5
                        ),
                        # 512x512
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=6
                        ),
                        # 1024x1024
                        lambda: create_growth_transition_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            alpha_var=alpha_var,
                            params=params,
                            trans_idx=7
                        ),
                        # 1024x1024
                        lambda: create_final_generator_network(
                            Z=Z,
                            projection_layer=projection_layer,
                            to_rgb_conv_layers=to_rgb_conv_layers,
                            blocks=block_layers,
                            original_image_size=original_image_size,
                            params=params
                        )
                    ]
                )

            # Call function from list for generated outputs at growth index.
            generated_outputs = generated_outputs_list[params["growth_index"]]

            print_obj(
                "generator_network", "generated_outputs", generated_outputs
            )

    return generated_outputs


def get_generator_loss(fake_logits, params):
    """Gets generator loss.

    Args:
        fake_logits: tensor, shape of [cur_batch_size, 1] that came from
            discriminator having processed generator's output image.
        params: dict, user passed parameters.

    Returns:
        Tensor of generator's total loss of shape [].
    """
    # Calculate base generator loss.
    generator_loss = -tf.reduce_mean(
        input_tensor=fake_logits,
        name="generator_loss"
    )
    print_obj("\nget_generator_loss", "generator_loss", generator_loss)

    # Get generator regularization losses.
    generator_reg_loss = regularization.get_regularization_loss(
        params=params, scope="generator"
    )
    print_obj(
        "get_generator_loss",
        "generator_reg_loss",
        generator_reg_loss
    )

    # Combine losses for total losses.
    generator_total_loss = tf.math.add(
        x=generator_loss,
        y=generator_reg_loss,
        name="generator_total_loss"
    )
    print_obj(
        "get_generator_loss", "generator_total_loss", generator_total_loss
    )

    return generator_total_loss
