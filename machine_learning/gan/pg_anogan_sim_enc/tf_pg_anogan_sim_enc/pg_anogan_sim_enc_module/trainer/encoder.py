import tensorflow as tf

from . import regularization
from .print_object import print_obj


class Encoder(object):
    """Encoder that takes image input and outputs logits.

    Fields:
        name: str, name of `Encoder`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        from_rgb_conv_layers: list, fromRGB 1x1 `Conv2D` layers.
        conv_layer_blocks: list, lists of `Conv2D` block layers for each
            block.
        transition_downsample_layers: list, `AveragePooling2D` layers for
            downsampling shrinking transition paths.
        flatten_layer: `Flatten` layer prior to logits layer.
        logits_layer: `Dense` layer for logits.
        build_encoder_tensors: list, tensors used to build layer
            internals.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Instantiates and builds encoder network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of encoder.
        """
        # Set name of encoder.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

        # Instantiate encoder layers.
        (self.from_rgb_conv_layers,
         self.conv_layer_blocks,
         self.transition_downsample_layers,
         self.flatten_layer,
         self.logits_layer) = self.instantiate_encoder_layers(
            params
        )

        # Build encoder layer internals.
        self.build_encoder_tensors = self.build_encoder_layers(
            params
        )

    def instantiate_encoder_from_rgb_layers(self, params):
        """Instantiates encoder fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of fromRGB 1x1 Conv2D layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get fromRGB layer properties.
            from_rgb = [
                params["encoder_from_rgb_layers"][i][0][:]
                for i in range(len(params["encoder_from_rgb_layers"]))
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
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="{}_from_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                        self.name,
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
                "\ninstantiate_encoder_from_rgb_layers",
                "from_rgb_conv_layers",
                from_rgb_conv_layers
            )

        return from_rgb_conv_layers

    def instantiate_encoder_base_conv_layer_block(self, params):
        """Instantiates encoder base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of base conv layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["encoder_base_conv_blocks"][0]

            # Create list of base conv layers.
            base_conv_layers = [
                tf.layers.Conv2D(
                    filters=conv_block[i][3],
                    kernel_size=conv_block[i][0:2],
                    strides=conv_block[i][4:6],
                    padding="same",
                    activation=tf.nn.leaky_relu,
                    kernel_initializer="he_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="{}_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                        self.name,
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
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="{}_base_layers_conv2d_{}_{}x{}_{}_{}".format(
                        self.name,
                        len(conv_block) - 1,
                        conv_block[-1][0],
                        conv_block[-1][1],
                        conv_block[-1][2],
                        conv_block[-1][3]
                    )
                )
            )
            print_obj(
                "\ninstantiate_encoder_base_conv_layer_block",
                "base_conv_layers",
                base_conv_layers
            )

        return base_conv_layers

    def instantiate_encoder_growth_layer_block(self, params, block_idx):
        """Instantiates encoder growth block layers.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["encoder_growth_conv_blocks"][block_idx]

            # Create new inner convolutional layers.
            conv_layers = [
                tf.layers.Conv2D(
                    filters=conv_block[i][3],
                    kernel_size=conv_block[i][0:2],
                    strides=conv_block[i][4:6],
                    padding="same",
                    activation=tf.nn.leaky_relu,
                    kernel_initializer="he_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="{}_growth_layers_conv2d_{}_{}_{}x{}_{}_{}".format(
                        self.name,
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
                "\ninstantiate_encoder_growth_layer_block",
                "conv_layers",
                conv_layers
            )

            # Down sample from 2s X 2s to s X s image.
            downsampled_image_layer = tf.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_growth_downsampled_image_{}".format(
                    self.name,
                    block_idx
                )
            )
            print_obj(
                "instantiate_encoder_growth_layer_block",
                "downsampled_image_layer",
                downsampled_image_layer
            )

        return conv_layers + [downsampled_image_layer]

    def instantiate_encoder_growth_transition_downsample_layers(
            self, params):
        """Instantiates encoder growth transition downsample layers.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of growth transition downsample layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Down sample from 2s X 2s to s X s image.
            downsample_layers = [
                tf.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    name="{}_growth_transition_downsample_layer_{}".format(
                        self.name,
                        layer_idx
                    )
                )
                for layer_idx in range(
                    1 + len(params["encoder_growth_conv_blocks"])
                )
            ]
            print_obj(
                "\ninstantiate_encoder_growth_transition_downsample_layers",
                "downsample_layers",
                downsample_layers
            )

        return downsample_layers

    def instantiate_encoder_logits_layer(self, params):
        """Instantiates encoder flatten and logits layers.

        Args:
            params: dict, user passed parameters.
        Returns:
            Flatten and logits layers of encoder.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten layer to ready final block conv tensor for dense layer.
            flatten_layer = tf.layers.Flatten(
                name="{}_flatten_layer".format(self.name)
            )
            print_obj(
                "\ncreate_encoder_logits_layer",
                "flatten_layer",
                flatten_layer
            )

            # Final linear layer for logits with same shape as latent vector.
            logits_layer = tf.layers.Dense(
                units=params["latent_size"],
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_layers_dense_logits".format(self.name)
            )
            print_obj(
                "create_growth_transition_encoder_network",
                "logits_layer",
                logits_layer
            )

        return flatten_layer, logits_layer

    def instantiate_encoder_layers(self, params):
        """Instantiates layers of encoder network.

        Args:
            params: dict, user passed parameters.

        Returns:
            from_rgb_conv_layers: list, fromRGB 1x1 `Conv2D` layers.
            conv_layer_blocks: list, lists of `Conv2D` block layers for each
                block.
            transition_downsample_layers: list, `AveragePooling2D` layers for
                downsampling shrinking transition paths.
            flatten_layer: `Flatten` layer prior to logits layer.
            logits_layer: `Dense` layer for logits.
        """
        # Instantiate fromRGB 1x1 `Conv2D` layers.
        from_rgb_conv_layers = self.instantiate_encoder_from_rgb_layers(
            params=params
        )
        print_obj(
            "instantiate_encoder_layers",
            "from_rgb_conv_layers",
            from_rgb_conv_layers
        )

        # Instantiate base conv block's `Conv2D` layers, for post-growth.
        conv_layer_blocks = [
            self.instantiate_encoder_base_conv_layer_block(
                params=params
            )
        ]

        # Instantiate growth `Conv2D` layer blocks.
        conv_layer_blocks.extend(
            [
                self.instantiate_encoder_growth_layer_block(
                    params=params,
                    block_idx=block_idx
                )
                for block_idx in range(
                    len(params["encoder_growth_conv_blocks"])
                )
            ]
        )
        print_obj(
            "instantiate_encoder_layers",
            "conv_layer_blocks",
            conv_layer_blocks
        )

        # Instantiate transition downsample `AveragePooling2D` layers.
        transition_downsample_layers = (
            self.instantiate_encoder_growth_transition_downsample_layers(
                params=params
            )
        )
        print_obj(
            "instantiate_encoder_layers",
            "transition_downsample_layers",
            transition_downsample_layers
        )

        # Instantiate `Flatten` and `Dense` logits layers.
        (flatten_layer,
         logits_layer) = self.instantiate_encoder_logits_layer(params=params)
        print_obj(
            "instantiate_encoder_layers",
            "flatten_layer",
            flatten_layer
        )
        print_obj(
            "instantiate_encoder_layers",
            "logits_layer",
            logits_layer
        )

        return (from_rgb_conv_layers,
                conv_layer_blocks,
                transition_downsample_layers,
                flatten_layer,
                logits_layer)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def build_encoder_from_rgb_layers(self, params):
        """Creates encoder fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from fromRGB 1x1 `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get fromRGB layer properties.
            from_rgb = [
                params["encoder_from_rgb_layers"][i][0][:]
                for i in range(len(params["encoder_from_rgb_layers"]))
            ]

            # Create list to hold fromRGB 1x1 convs.
            from_rgb_conv_tensors = [
                self.from_rgb_conv_layers[i](
                    inputs=tf.zeros(
                        shape=[1] + from_rgb[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(from_rgb))
            ]
            print_obj(
                "\nbuild_encoder_from_rgb_layers",
                "from_rgb_conv_tensors",
                from_rgb_conv_tensors
            )

        return from_rgb_conv_tensors

    def build_encoder_base_conv_layer_block(self, params):
        """Creates encoder base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from base `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["encoder_base_conv_blocks"][0]

            # The base conv block is always the 0th one.
            base_conv_layer_block = self.conv_layer_blocks[0]

            # Build base conv block layers, store in list.
            base_conv_tensors = [
                base_conv_layer_block[i](
                    inputs=tf.zeros(
                        shape=[1] + conv_block[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(conv_block))
            ]
            print_obj(
                "\nbuild_encoder_base_conv_layer_block",
                "base_conv_tensors",
                base_conv_tensors
            )

        return base_conv_tensors

    def build_encoder_growth_layer_block(self, params, block_idx):
        """Creates encoder growth block.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of tensors from growth block `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["encoder_growth_conv_blocks"][block_idx]

            # Create new inner convolutional layers.
            conv_tensors = [
                self.conv_layer_blocks[1 + block_idx][i](
                    inputs=tf.zeros(
                        shape=[1] + conv_block[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(conv_block))
            ]
            print_obj(
                "\nbuild_encoder_growth_layer_block",
                "conv_tensors",
                conv_tensors
            )

        return conv_tensors

    def build_encoder_logits_layer(self, params):
        """Builds flatten and logits layer internals using call.

        Args:
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            block_conv_size = params["encoder_base_conv_blocks"][-1][-1][3]

            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(
                inputs=tf.zeros(
                    shape=[1, 1, 1, block_conv_size],
                    dtype=tf.float32
                )
            )
            print_obj(
                "\nbuild_encoder_logits_layer",
                "block_conv_flat",
                block_conv_flat
            )

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj("build_encoder_logits_layer", "logits", logits)

        return logits

    def build_encoder_layers(self, params):
        """Builds encoder layer internals.

        Args:
            params: dict, user passed parameters.

        Returns:
            Logits tensor.
        """
        # Build fromRGB 1x1 `Conv2D` layers internals through call.
        from_rgb_conv_tensors = self.build_encoder_from_rgb_layers(
            params=params
        )
        print_obj(
            "\nbuild_encoder_layers",
            "from_rgb_conv_tensors",
            from_rgb_conv_tensors
        )

        with tf.control_dependencies(control_inputs=from_rgb_conv_tensors):
            # Create base convolutional block's layer internals using call.
            conv_block_tensors = [
                self.build_encoder_base_conv_layer_block(
                    params=params
                )
            ]

            # Build growth `Conv2D` layer block internals through call.
            conv_block_tensors.extend(
                [
                    self.build_encoder_growth_layer_block(
                        params=params, block_idx=block_idx
                    )
                    for block_idx in range(
                       len(params["encoder_growth_conv_blocks"])
                    )
                ]
            )

            # Flatten conv block tensor lists of lists into list.
            conv_block_tensors = [
                item for sublist in conv_block_tensors for item in sublist
            ]
            print_obj(
                "build_encoder_layers",
                "conv_block_tensors",
                conv_block_tensors
            )

            with tf.control_dependencies(control_inputs=conv_block_tensors):
                # Build logits layer internals using call.
                logits_tensor = self.build_encoder_logits_layer(
                    params=params
                )
                print_obj(
                    "build_encoder_layers",
                    "logits_tensor",
                    logits_tensor
                )

        return logits_tensor

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def use_encoder_logits_layer(self, block_conv, params):
        """Uses flatten and logits layers to get logits tensor.

        Args:
            block_conv: tensor, output of last conv layer of encoder.
            flatten_layer: `Flatten` layer.
            logits_layer: `Dense` layer for logits.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        print_obj(
            "\nuse_encoder_logits_layer", "block_conv", block_conv
        )
        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0] / 4,
                params["generator_projection_dims"][1] / 4,
                block_conv.get_shape()[-1]]
        )
        print_obj("use_encoder_logits_layer", "block_conv", block_conv)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(inputs=block_conv)
            print_obj(
                "use_encoder_logits_layer",
                "block_conv_flat",
                block_conv_flat
            )

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj("use_encoder_logits_layer", "logits", logits)

        return logits

    def create_base_encoder_network(self, X, params):
        """Creates base encoder network.

        Args:
            X: tensor, input image to encoder.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        print_obj("\ncreate_base_encoder_network", "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the first fromRGB conv layer & block for base network.
            from_rgb_conv_layer = self.from_rgb_conv_layers[0]
            block_layers = self.conv_layer_blocks[0]

            # Pass inputs through layer chain.
            from_rgb_conv = from_rgb_conv_layer(inputs=X)
            print_obj(
                "create_base_encoder_network",
                "from_rgb_conv",
                from_rgb_conv
            )

            block_conv = from_rgb_conv
            for i in range(len(block_layers)):
                block_conv = block_layers[i](inputs=block_conv)
                print_obj(
                    "create_base_encoder_network",
                    "block_conv",
                    block_conv
                )

            # Get logits now.
            logits = self.use_encoder_logits_layer(
                block_conv=block_conv,
                params=params
            )
            print_obj("create_base_encoder_network", "logits", logits)

        return logits

    def create_growth_transition_encoder_network(
            self, X, alpha_var, params, trans_idx):
        """Creates growth transition encoder network.

        Args:
            X: tensor, input image to encoder.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Final logits tensor of encoder.
        """
        print_obj(
            "\nEntered create_growth_transition_encoder_network",
            "trans_idx",
            trans_idx
        )
        print_obj("create_growth_transition_encoder_network", "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Growing side chain.
            growing_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx + 1]
            growing_block_layers = self.conv_layer_blocks[trans_idx + 1]

            # Pass inputs through layer chain.
            growing_block_conv = growing_from_rgb_conv_layer(inputs=X)
            print_obj(
                "\ncreate_growth_transition_encoder_network",
                "growing_block_conv",
                growing_block_conv
            )
            for i in range(len(growing_block_layers)):
                growing_block_conv = growing_block_layers[i](
                    inputs=growing_block_conv
                )
                print_obj(
                    "create_growth_transition_encoder_network",
                    "growing_block_conv",
                    growing_block_conv
                )

            # Shrinking side chain.
            transition_downsample_layer = self.transition_downsample_layers[trans_idx]
            shrinking_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx]

            # Pass inputs through layer chain.
            transition_downsample = transition_downsample_layer(inputs=X)
            print_obj(
                "create_growth_transition_encoder_network",
                "transition_downsample",
                transition_downsample
            )
            shrinking_from_rgb_conv = shrinking_from_rgb_conv_layer(
                inputs=transition_downsample
            )
            print_obj(
                "create_growth_transition_encoder_network",
                "shrinking_from_rgb_conv",
                shrinking_from_rgb_conv
            )

            # Weighted sum.
            weighted_sum = tf.add(
                x=growing_block_conv * alpha_var,
                y=shrinking_from_rgb_conv * (1.0 - alpha_var),
                name="{}_growth_transition_weighted_sum_{}".format(
                    self.name, trans_idx
                )
            )
            print_obj(
                "create_growth_transition_encoder_network",
                "weighted_sum",
                weighted_sum
            )

            # Permanent blocks.
            permanent_blocks = self.conv_layer_blocks[0:trans_idx + 1]

            # Reverse order of blocks and flatten.
            permanent_block_layers = [
                item for sublist in permanent_blocks[::-1] for item in sublist
            ]

            # Pass inputs through layer chain.
            block_conv = weighted_sum

            # Find number of permanent growth conv layers.
            num_perm_growth_conv_layers = len(permanent_block_layers)
            num_perm_growth_conv_layers -= len(params["conv_num_filters"][0])

            # Loop through only the permanent growth conv layers.
            for i in range(num_perm_growth_conv_layers):
                block_conv = permanent_block_layers[i](inputs=block_conv)
                print_obj(
                    "create_growth_transition_encoder_network",
                    "block_conv_{}".format(i),
                    block_conv
                )

            # Loop through only the permanent base conv layers now.
            for i in range(
                    num_perm_growth_conv_layers, len(permanent_block_layers)):
                block_conv = permanent_block_layers[i](inputs=block_conv)
                print_obj(
                    "create_growth_transition_encoder_network",
                    "block_conv_{}".format(i),
                    block_conv
                )

            # Get logits now.
            logits = self.use_encoder_logits_layer(
                block_conv=block_conv, params=params
            )
            print_obj(
                "create_growth_transition_encoder_network",
                "logits",
                logits
            )

        return logits

    def create_final_encoder_network(self, X, params):
        """Creates final encoder network.

        Args:
            X: tensor, input image to encoder.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        print_obj("\ncreate_final_encoder_network", "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the last fromRGB conv layer.
            from_rgb_conv_layer = self.from_rgb_conv_layers[-1]

            # Reverse order of blocks.
            reversed_blocks = self.conv_layer_blocks[::-1]

            # Flatten list of lists block layers into list.
            block_layers = [
                item for sublist in reversed_blocks for item in sublist
            ]

            # Pass inputs through layer chain.
            block_conv = from_rgb_conv_layer(inputs=X)
            print_obj(
                "\ncreate_final_encoder_network",
                "block_conv",
                block_conv
            )

            # Find number of permanent growth conv layers.
            num_growth_conv_layers = len(block_layers)
            num_growth_conv_layers -= len(params["conv_num_filters"][0])

            # Loop through only the permanent growth conv layers.
            for i in range(num_growth_conv_layers):
                block_conv = block_layers[i](inputs=block_conv)
                print_obj(
                    "create_final_encoder_network",
                    "block_conv_{}".format(i),
                    block_conv
                )

            # Loop through only the permanent base conv layers now.
            for i in range(num_growth_conv_layers, len(block_layers)):
                block_conv = block_layers[i](inputs=block_conv)
                print_obj(
                    "create_final_encoder_network",
                    "block_conv_{}".format(i),
                    block_conv
                )

            # Get logits now.
            logits = self.use_encoder_logits_layer(
                block_conv=block_conv,
                params=params
            )
            print_obj("create_final_encoder_network", "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def switch_case_encoder_logits(
            self, X, alpha_var, params, growth_index):
        """Uses switch case to use the correct network to get logits.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            growth_index: int, current growth stage.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        # Switch to case based on number of steps to get logits.
        logits = tf.switch_case(
            branch_index=growth_index,
            branch_fns=[
                # 4x4
                lambda: self.create_base_encoder_network(
                    X=X, params=params
                ),
                # 8x8
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(0, len(params["conv_num_filters"]) - 2)
                ),
                # 16x16
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(1, len(params["conv_num_filters"]) - 2)
                ),
                # 32x32
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(2, len(params["conv_num_filters"]) - 2)
                ),
                # 64x64
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(3, len(params["conv_num_filters"]) - 2)
                ),
                # 128x128
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(4, len(params["conv_num_filters"]) - 2)
                ),
                # 256x256
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(5, len(params["conv_num_filters"]) - 2)
                ),
                # 512x512
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(6, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_growth_transition_encoder_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(7, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_final_encoder_network(
                    X=X, params=params
                )
            ],
            name="{}_switch_case_logits".format(self.name)
        )

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_train_eval_encoder_logits(self, X, alpha_var, params):
        """Uses encoder network and returns encoded logits for train/eval.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, latent_size].
        """
        print_obj("\nget_train_eval_encoder_logits", "X", X)

        # Get encoder's logits tensor.
        train_steps = params["train_steps"] + params["prev_train_steps"]
        num_steps_until_growth = params["num_steps_until_growth"]
        num_stages = train_steps // num_steps_until_growth
        if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
            print(
                "\nget_train_eval_encoder_logits: NOT GOING TO GROW, SKIP SWITCH CASE!"
            )
            # If never going to grow, no sense using the switch case.
            # 4x4
            logits = self.create_base_encoder_network(
                X=X, params=params
            )
        else:
            # Find growth index based on global step and growth frequency.
            growth_index = tf.cast(
                x=tf.floordiv(
                    x=tf.train.get_or_create_global_step(),
                    y=params["num_steps_until_growth"],
                    name="{}_global_step_floordiv".format(self.name)
                ),
                dtype=tf.int32,
                name="{}_growth_index".format(self.name)
            )

            # Switch to case based on number of steps for logits.
            logits = self.switch_case_encoder_logits(
                X=X,
                alpha_var=alpha_var,
                params=params,
                growth_index=growth_index
            )

        print_obj(
            "\nget_train_eval_encoder_logits", "logits", logits
        )

        # Wrap logits in a control dependency for the build encoder
        # tensors to ensure encoder internals are built.
        with tf.control_dependencies(
                control_inputs=[self.build_encoder_tensors]):
            logits = tf.identity(
                input=logits, name="{}_logits_identity".format(self.name)
            )

        return logits

    def get_predict_encoder_logits(self, X, params, block_idx):
        """Uses encoder network and returns encoded logits for predict.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            params: dict, user passed parameters.
            block_idx: int, current conv layer block's index.

        Returns:
            Logits tensor of shape [cur_batch_size, latent_size].
        """
        print_obj("\nget_predict_encoder_logits", "X", X)

        # Get encoder's logits tensor.
        if block_idx == 0:
            # 4x4
            logits = self.create_base_encoder_network(X=X, params=params)
        elif block_idx < len(params["conv_num_filters"]) - 1:
            # 8x8 through 512x512
            logits = self.create_growth_transition_encoder_network(
                X=X,
                alpha_var=tf.ones(shape=[], dtype=tf.float32),
                params=params,
                trans_idx=block_idx - 1
            )
        else:
            # 1024x1024
            logits = self.create_final_encoder_network(X=X, params=params)

        print_obj("\nget_predict_encoder_logits", "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_encoder_loss(
            self,
            fake_images,
            encoded_images,
            params):
        """Gets encoder loss.

        Args:
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            encoded_images: tensor, images generated by the generator from
                encoder's vector output of shape
                [cur_batch_size, image_size, image_size, 3].
            params: dict, user passed parameters.

        Returns:
            Encoder's total loss tensor of shape [].
        """
        # Get difference between fake images and encoder images.
        generator_encoder_image_diff = tf.subtract(
            x=fake_images,
            y=encoded_images,
            name="generator_encoder_image_diff"
        )

        # Get L1 norm of image difference.
        image_diff_l1_norm = tf.reduce_sum(
            input_tensor=tf.abs(x=generator_encoder_image_diff),
            axis=[1, 2, 3]
        )

        # Calculate base encoder loss.
        encoder_loss = tf.reduce_mean(
            input_tensor=image_diff_l1_norm,
            name="{}_loss".format(self.name)
        )
        print_obj(
            "get_encoder_loss",
            "encoder_loss",
            encoder_loss
        )

        # Get encoder regularization losses.
        encoder_reg_loss = regularization.get_regularization_loss(
            lambda1=params["encoder_l1_regularization_scale"],
            lambda2=params["encoder_l2_regularization_scale"],
            scope=self.name
        )
        print_obj(
            "get_encoder_loss",
            "encoder_reg_loss",
            encoder_reg_loss
        )

        # Combine losses for total losses.
        encoder_total_loss = tf.add(
            x=encoder_loss,
            y=encoder_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            "get_encoder_loss",
            "encoder_total_loss",
            encoder_total_loss
        )

        return encoder_total_loss
