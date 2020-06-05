import tensorflow as tf

from . import regularization
from .print_object import print_obj


class Discriminator(object):
    """
    Fields:
        name: str, name of `Discriminator`.
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
        build_discriminator_tensors: list, tensors used to build layer
            internals.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Creates generator network and returns generated output.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of discriminator.
        """
        # Set name of discriminator.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

        # Instantiate discriminator layers.
        (self.from_rgb_conv_layers,
         self.conv_layer_blocks,
         self.transition_downsample_layers,
         self.flatten_layer,
         self.logits_layer) = self.instantiate_discriminator_layers(
            params
        )

        # Build discriminator layer internals.
        self.build_discriminator_tensors = self.build_discriminator_layers(
            params
        )

    def instantiate_discriminator_from_rgb_layers(self, params):
        """Instantiates discriminator fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of fromRGB 1x1 Conv2D layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                "\ninstantiate_discriminator_from_rgb_layers",
                "from_rgb_conv_layers",
                from_rgb_conv_layers
            )

        return from_rgb_conv_layers

    def instantiate_discriminator_base_conv_layer_block(self, params):
        """Instantiates discriminator base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of base conv layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                "\ninstantiate_discriminator_base_conv_layer_block",
                "base_conv_layers",
                base_conv_layers
            )

        return base_conv_layers

    def instantiate_discriminator_growth_layer_block(self, params, block_idx):
        """Instantiates discriminator growth block layers.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                "\ninstantiate_discriminator_growth_layer_block",
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
                "instantiate_discriminator_growth_layer_block",
                "downsampled_image_layer",
                downsampled_image_layer
            )

        return conv_layers + [downsampled_image_layer]

    def instantiate_discriminator_growth_transition_downsample_layers(
            self, params):
        """Instantiates discriminator growth transition downsample layers.

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
                    1 + len(params["discriminator_growth_conv_blocks"])
                )
            ]
            print_obj(
                "\ninstantiate_discriminator_growth_transition_downsample_layers",
                "downsample_layers",
                downsample_layers
            )

        return downsample_layers

    def instantiate_discriminator_logits_layer(self):
        """Instantiates discriminator flatten and logits layers.

        Returns:
            Flatten and logits layers of discriminator.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten layer to ready final block conv tensor for dense layer.
            flatten_layer = tf.layers.Flatten(
                name="{}_flatten_layer".format(self.name)
            )
            print_obj(
                "\ncreate_discriminator_logits_layer",
                "flatten_layer",
                flatten_layer
            )

            # Final linear layer for logits.
            logits_layer = tf.layers.Dense(
                units=1,
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_layers_dense_logits".format(self.name)
            )
            print_obj(
                "create_growth_transition_discriminator_network",
                "logits_layer",
                logits_layer
            )

        return flatten_layer, logits_layer

    def instantiate_discriminator_layers(self, params):
        """Instantiates layers of discriminator network.

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
        from_rgb_conv_layers = self.instantiate_discriminator_from_rgb_layers(
            params=params
        )
        print_obj(
            "instantiate_discriminator_layers",
            "from_rgb_conv_layers",
            from_rgb_conv_layers
        )

        # Instantiate base conv block's `Conv2D` layers, for post-growth.
        conv_layer_blocks = [
            self.instantiate_discriminator_base_conv_layer_block(
                params=params
            )
        ]

        # Instantiate growth `Conv2D` layer blocks.
        conv_layer_blocks.extend(
            [
                self.instantiate_discriminator_growth_layer_block(
                    params=params,
                    block_idx=block_idx
                )
                for block_idx in range(
                    len(params["discriminator_growth_conv_blocks"])
                )
            ]
        )
        print_obj(
            "instantiate_discriminator_layers",
            "conv_layer_blocks",
            conv_layer_blocks
        )

        # Instantiate transition downsample `AveragePooling2D` layers.
        transition_downsample_layers = (
            self.instantiate_discriminator_growth_transition_downsample_layers(
                params=params
            )
        )
        print_obj(
            "instantiate_discriminator_layers",
            "transition_downsample_layers",
            transition_downsample_layers
        )

        # Instantiate `Flatten` and `Dense` logits layers.
        (flatten_layer,
         logits_layer) = self.instantiate_discriminator_logits_layer()
        print_obj(
            "instantiate_discriminator_layers",
            "flatten_layer",
            flatten_layer
        )
        print_obj(
            "instantiate_discriminator_layers",
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

    def build_discriminator_from_rgb_layers(self, params):
        """Creates discriminator fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from fromRGB 1x1 `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get fromRGB layer properties.
            from_rgb = [
                params["discriminator_from_rgb_layers"][i][0][:]
                for i in range(len(params["discriminator_from_rgb_layers"]))
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
                "\nbuild_discriminator_from_rgb_layers",
                "from_rgb_conv_tensors",
                from_rgb_conv_tensors
            )

        return from_rgb_conv_tensors

    def build_discriminator_base_conv_layer_block(self, params):
        """Creates discriminator base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from base `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["discriminator_base_conv_blocks"][0]

            # Create list of base conv layer tensors.
            base_conv_tensors = [
                # The base conv block is always the 0th one.
                self.conv_layer_blocks[0][i](
                    inputs=tf.zeros(
                        shape=[1] + conv_block[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(conv_block))
            ]
            print_obj(
                "\nbuild_discriminator_base_conv_layer_block",
                "base_conv_tensors",
                base_conv_tensors
            )

        return base_conv_tensors

    def build_discriminator_growth_layer_block(self, params, block_idx):
        """Creates discriminator growth block.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of tensors from growth block `Conv2D` layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["discriminator_growth_conv_blocks"][block_idx]

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
                "\nbuild_discriminator_growth_layer_block",
                "conv_tensors",
                conv_tensors
            )

        return conv_tensors

    def build_discriminator_logits_layer(self, params):
        """Builds flatten and logits layer internals using call.

        Args:
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of discriminator.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            block_conv_size = params["discriminator_base_conv_blocks"][-1][-1][3]

            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(
                inputs=tf.zeros(
                    shape=[1, 1, 1, block_conv_size],
                    dtype=tf.float32
                )
            )
            print_obj(
                "\nbuild_discriminator_logits_layer",
                "block_conv_flat",
                block_conv_flat
            )

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj("build_discriminator_logits_layer", "logits", logits)

        return logits

    def build_discriminator_layers(self, params):
        """Builds discriminator layer internals.

        Args:
            params: dict, user passed parameters.

        Returns:
            Logits tensor.
        """
        # Build fromRGB 1x1 `Conv2D` layers internals through call.
        from_rgb_conv_tensors = self.build_discriminator_from_rgb_layers(
            params=params
        )
        print_obj(
            "\nbuild_discriminator_layers",
            "from_rgb_conv_tensors",
            from_rgb_conv_tensors
        )

        with tf.control_dependencies(control_inputs=from_rgb_conv_tensors):
            # Create base convolutional block's layer internals using call.
            conv_block_tensors = [
                self.build_discriminator_base_conv_layer_block(
                    params=params
                )
            ]

            # Build growth `Conv2D` layer block internals through call.
            conv_block_tensors.extend(
                [
                    self.build_discriminator_growth_layer_block(
                        params=params, block_idx=block_idx
                    )
                    for block_idx in range(
                       len(params["discriminator_growth_conv_blocks"])
                    )
                ]
            )

            # Flatten conv block tensor lists of lists into list.
            conv_block_tensors = [
                item for sublist in conv_block_tensors for item in sublist
            ]
            print_obj(
                "build_discriminator_layers",
                "conv_block_tensors",
                conv_block_tensors
            )

            with tf.control_dependencies(control_inputs=conv_block_tensors):
                # Build logits layer internals using call.
                logits_tensor = self.build_discriminator_logits_layer(
                    params=params
                )
                print_obj(
                    "build_discriminator_layers",
                    "logits_tensor",
                    logits_tensor
                )

        return logits_tensor

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def use_discriminator_logits_layer(self, block_conv, params):
        """Uses flatten and logits layers to get logits tensor.

        Args:
            block_conv: tensor, output of last conv layer of discriminator.
            flatten_layer: `Flatten` layer.
            logits_layer: `Dense` layer for logits.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of discriminator.
        """
        print_obj(
            "\nuse_discriminator_logits_layer", "block_conv", block_conv
        )
        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0] / 4,
                params["generator_projection_dims"][1] / 4,
                block_conv.get_shape()[-1]]
        )
        print_obj("use_discriminator_logits_layer", "block_conv", block_conv)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(inputs=block_conv)
            print_obj(
                "use_discriminator_logits_layer",
                "block_conv_flat",
                block_conv_flat
            )

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj("use_discriminator_logits_layer", "logits", logits)

        return logits

    def create_base_discriminator_network(self, X, params):
        """Creates base discriminator network.

        Args:
            X: tensor, input image to discriminator.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of discriminator.
        """
        print_obj("\ncreate_base_discriminator_network", "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the first fromRGB conv layer & block for base network.
            from_rgb_conv_layer = self.from_rgb_conv_layers[0]
            block_layers = self.conv_layer_blocks[0]

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
                    "create_base_discriminator_network",
                    "block_conv",
                    block_conv
                )

            # Get logits now.
            logits = self.use_discriminator_logits_layer(
                block_conv=block_conv,
                params=params
            )
            print_obj("create_base_discriminator_network", "logits", logits)

        return logits

    def create_growth_transition_discriminator_network(
            self, X, alpha_var, params, trans_idx):
        """Creates growth transition discriminator network.

        Args:
            X: tensor, input image to discriminator.
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
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Growing side chain.
            growing_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx + 1]
            growing_block_layers = self.conv_layer_blocks[trans_idx + 1]

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
            transition_downsample_layer = self.transition_downsample_layers[trans_idx]
            shrinking_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx]

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
                name="{}_growth_transition_weighted_sum_{}".format(
                    self.name, trans_idx
                )
            )
            print_obj(
                "create_growth_transition_discriminator_network",
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
            for i in range(len(permanent_block_layers)):
                block_conv = permanent_block_layers[i](inputs=block_conv)
                print_obj(
                    "create_growth_transition_discriminator_network",
                    "block_conv",
                    block_conv
                )

            # Get logits now.
            logits = self.use_discriminator_logits_layer(
                block_conv=block_conv, params=params
            )
            print_obj(
                "create_growth_transition_discriminator_network",
                "logits",
                logits
            )

        return logits

    def create_final_discriminator_network(self, X, params):
        """Creates final discriminator network.

        Args:
            X: tensor, input image to discriminator.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of discriminator.
        """
        print_obj("\ncreate_final_discriminator_network", "X", X)
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
                "\ncreate_final_discriminator_network",
                "block_conv",
                block_conv
            )

            for i in range(len(block_layers)):
                block_conv = block_layers[i](inputs=block_conv)
                print_obj(
                    "create_final_discriminator_network",
                    "block_conv",
                    block_conv
                )

            # Get logits now.
            logits = self.use_discriminator_logits_layer(
                block_conv=block_conv,
                params=params
            )
            print_obj("create_final_discriminator_network", "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def switch_case_discriminator_logits(
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
                lambda: self.create_base_discriminator_network(
                    X=X, params=params
                ),
                # 8x8
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(0, len(params["conv_num_filters"]) - 2)
                ),
                # 16x16
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(1, len(params["conv_num_filters"]) - 2)
                ),
                # 32x32
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(2, len(params["conv_num_filters"]) - 2)
                ),
                # 64x64
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(3, len(params["conv_num_filters"]) - 2)
                ),
                # 128x128
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(4, len(params["conv_num_filters"]) - 2)
                ),
                # 256x256
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(5, len(params["conv_num_filters"]) - 2)
                ),
                # 512x512
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(6, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_growth_transition_discriminator_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(7, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_final_discriminator_network(
                    X=X, params=params
                )
            ],
            name="{}_switch_case_logits".format(self.name)
        )

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_discriminator_logits(self, X, alpha_var, params):
        """Uses generator network and returns generated output for train/eval.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        print_obj("\nget_discriminator_logits", "X", X)

        # Get discriminator's logits tensor.
        train_steps = params["train_steps"]
        num_steps_until_growth = params["num_steps_until_growth"]
        num_stages = train_steps // num_steps_until_growth
        if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
            print(
                "\nget_discriminator_logits: NOT GOING TO GROW, SKIP SWITCH CASE!"
            )
            # If never going to grow, no sense using the switch case.
            # 4x4
            logits = self.create_base_discriminator_network(
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
            logits = self.switch_case_discriminator_logits(
                X=X,
                alpha_var=alpha_var,
                params=params,
                growth_index=growth_index
            )

        print_obj(
            "\nget_discriminator_logits", "logits", logits
        )

        # Wrap logits in a control dependency for the build discriminator
        # tensors to ensure discriminator internals are built.
        with tf.control_dependencies(
                control_inputs=[self.build_discriminator_tensors]):
            logits = tf.identity(
                input=logits, name="{}_logits_identity".format(self.name)
            )

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_gradient_penalty_loss(
            self,
            cur_batch_size,
            fake_images,
            real_images,
            alpha_var,
            params):
        """Gets discriminator gradient penalty loss.

        Args:
            cur_batch_size: tensor, in case of a partial batch instead of
                using the user passed int.
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [cur_batch_size, image_size, image_size, 3].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Discriminator's gradient penalty loss of shape [].
        """
        with tf.name_scope(name="{}/gradient_penalty".format(self.name)):
            # Get a random uniform number rank 4 tensor.
            random_uniform_num = tf.random.uniform(
                shape=[cur_batch_size, 1, 1, 1],
                minval=0., maxval=1.,
                dtype=tf.float32,
                name="random_uniform_num"
            )
            print_obj(
                "\nget_gradient_penalty_loss",
                "random_uniform_num",
                random_uniform_num
            )

            # Find the element-wise difference between images.
            image_difference = real_images - fake_images
            print_obj(
                "get_gradient_penalty_loss",
                "image_difference",
                image_difference
            )

            # Get random samples from this mixed image distribution.
            mixed_images = random_uniform_num * image_difference
            mixed_images += fake_images
            print_obj(
                "get_gradient_penalty_loss",
                "mixed_images",
                mixed_images
            )

            # Send to the discriminator to get logits.
            mixed_logits = self.get_discriminator_logits(
                X=mixed_images, alpha_var=alpha_var, params=params
            )
            print_obj(
                "get_gradient_penalty_loss",
                "mixed_logits",
                mixed_logits
            )

            # Get the mixed loss.
            mixed_loss = tf.reduce_sum(
                input_tensor=mixed_images,
                name="mixed_loss"
            )
            print_obj(
                "get_gradient_penalty_loss",
                "mixed_loss",
                mixed_loss
            )

            # Get gradient from returned list of length 1.
            mixed_gradients = tf.gradients(
                ys=mixed_loss,
                xs=[mixed_images],
                name="gradients"
            )[0]
            print_obj(
                "get_gradient_penalty_loss",
                "mixed_gradients",
                mixed_gradients
            )

            # Get gradient's L2 norm.
            mixed_norms = tf.sqrt(
                x=tf.reduce_sum(
                    input_tensor=tf.square(
                        x=mixed_gradients,
                        name="squared_grads"
                    ),
                    axis=[1, 2, 3]
                )
            )
            print_obj(
                "get_gradient_penalty_loss",
                "mixed_norms",
                mixed_norms
            )

            # Get squared difference from target of 1.0.
            squared_difference = tf.square(
                x=mixed_norms - 1.0,
                name="{}_squared_difference".format(self.name)
            )
            print_obj(
                "get_gradient_penalty_loss",
                "squared_difference",
                squared_difference
            )

            # Get gradient penalty scalar.
            gradient_penalty = tf.reduce_mean(
                input_tensor=squared_difference, name="gradient_penalty"
            )
            print_obj(
                "get_gradient_penalty_loss",
                "gradient_penalty",
                gradient_penalty
            )

            # Multiply with lambda to get gradient penalty loss.
            gradient_penalty_loss = tf.multiply(
                x=params["discriminator_gradient_penalty_coefficient"],
                y=gradient_penalty,
                name="gradient_penalty_loss"
            )

            return gradient_penalty_loss

    def get_discriminator_loss(
            self,
            cur_batch_size,
            fake_images,
            real_images,
            fake_logits,
            real_logits,
            alpha_var,
            params):
        """Gets discriminator loss.

        Args:
            cur_batch_size: tensor, in case of a partial batch instead of
                using the user passed int.
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [cur_batch_size, image_size, image_size, 3].
            fake_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed generator's output image.
            fake_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed real image.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Discriminator's total loss tensor of shape [].
        """
        # Calculate base discriminator loss.
        discriminator_real_loss = tf.reduce_mean(
            input_tensor=real_logits,
            name="{}_real_loss".format(self.name)
        )
        print_obj(
            "\nget_discriminator_loss",
            "discriminator_real_loss",
            discriminator_real_loss
        )

        discriminator_generated_loss = tf.reduce_mean(
            input_tensor=fake_logits,
            name="{}_generated_loss".format(self.name)
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_generated_loss",
            discriminator_generated_loss
        )

        discriminator_loss = tf.add(
            x=discriminator_real_loss, y=-discriminator_generated_loss,
            name="{}_loss".format(self.name)
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_loss",
            discriminator_loss
        )

        # Get discriminator gradient penalty loss.
        discriminator_gradient_penalty = self.get_gradient_penalty_loss(
            cur_batch_size=cur_batch_size,
            fake_images=fake_images,
            real_images=real_images,
            alpha_var=alpha_var,
            params=params
        )

        # Get discriminator Wasserstein GP loss.
        discriminator_wasserstein_gp_loss = tf.add(
            x=discriminator_loss,
            y=discriminator_gradient_penalty,
            name="{}_wasserstein_gp_loss".format(self.name)
        )

        # Get discriminator regularization losses.
        discriminator_reg_loss = regularization.get_regularization_loss(
            params=params, scope=self.name
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
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            "get_discriminator_loss",
            "discriminator_total_loss",
            discriminator_total_loss
        )

        return discriminator_total_loss
