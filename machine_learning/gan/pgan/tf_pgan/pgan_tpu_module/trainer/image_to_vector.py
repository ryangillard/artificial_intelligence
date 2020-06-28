import tensorflow as tf

from . import equalized_learning_rate_layers
from .print_object import print_obj


class ImageToVector(object):
    """Convolutional network takes image input and outputs a vector.

    Fields:
        kind: str, kind of `ImageToVector` instance.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        projection_layer: `Dense` layer for projection of noise to image.
        conv_layer_blocks: list, lists of block layers for each block.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        build_vector_to_image_tensors: list, tensors used to build layer
            internals.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, kind):
        """Instantiates and builds vec_to_img network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            kind: str, kind of `ImageToVector` instance.
        """
        # Set kind of image to vector network.
        self.kind = kind

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

        # Instantiate image to vector layers.
        (self.from_rgb_conv_layers,
         self.conv_layer_blocks,
         self.flatten_layer,
         self.logits_layer) = self.instantiate_img_to_vec_layers(
            params
        )

        # Build image to vector layer internals.
        self.build_img_to_vec_tensors = self.build_img_to_vec_layers(
            params
        )

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def instantiate_img_to_vec_from_rgb_layers(self, params):
        """Instantiates img_to_vec fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of fromRGB 1x1 Conv2D layers.
        """
        func_name = "instantiate_{}_from_rgb_layers".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get fromRGB layer properties.
            from_rgb = [
                params["{}_from_rgb_layers".format(self.kind)][i][0][:]
                for i in range(
                    len(params["{}_from_rgb_layers".format(self.kind)])
                )
            ]

            # Create list to hold toRGB 1x1 convs.
            from_rgb_conv_layers = [
                equalized_learning_rate_layers.Conv2D(
                    filters=from_rgb[i][3],
                    kernel_size=from_rgb[i][0:2],
                    strides=from_rgb[i][4:6],
                    padding="same",
                    activation=None,
                    kernel_initializer=(
                        tf.random_normal_initializer(mean=0., stddev=1.0)
                        if params["use_equalized_learning_rate"]
                        else "he_normal"
                    ),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    equalized_learning_rate=params["use_equalized_learning_rate"],
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
                "\n" + func_name, "from_rgb_conv_layers", from_rgb_conv_layers
            )

        return from_rgb_conv_layers

    def instantiate_img_to_vec_base_conv_layer_block(self, params):
        """Instantiates img_to_vec base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of base conv layers.
        """
        func_name = "instantiate_{}_base_conv_layer_block".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["{}_base_conv_blocks".format(self.kind)][0]

            # Create list of base conv layers.
            base_conv_layers = [
                equalized_learning_rate_layers.Conv2D(
                    filters=conv_block[i][3],
                    kernel_size=conv_block[i][0:2],
                    strides=conv_block[i][4:6],
                    padding="same",
                    activation=None,
                    kernel_initializer=(
                        tf.random_normal_initializer(mean=0., stddev=1.0)
                        if params["use_equalized_learning_rate"]
                        else "he_normal"
                    ),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    equalized_learning_rate=params["use_equalized_learning_rate"],
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
                equalized_learning_rate_layers.Conv2D(
                    filters=conv_block[-1][3],
                    kernel_size=conv_block[-1][0:2],
                    strides=conv_block[-1][4:6],
                    padding="valid",
                    activation=None,
                    kernel_initializer=(
                        tf.random_normal_initializer(mean=0., stddev=1.0)
                        if params["use_equalized_learning_rate"]
                        else "he_normal"
                    ),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    equalized_learning_rate=params["use_equalized_learning_rate"],
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
                "\n" + func_name, "base_conv_layers", base_conv_layers
            )

        return base_conv_layers

    def instantiate_img_to_vec_growth_layer_block(self, params, block_idx):
        """Instantiates img_to_vec growth block layers.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block layers.
        """
        func_name = "instantiate_{}_growth_layer_block".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["{}_growth_conv_blocks".format(self.kind)][block_idx]

            # Create new inner convolutional layers.
            conv_layers = [
                equalized_learning_rate_layers.Conv2D(
                    filters=conv_block[i][3],
                    kernel_size=conv_block[i][0:2],
                    strides=conv_block[i][4:6],
                    padding="same",
                    activation=None,
                    kernel_initializer=(
                        tf.random_normal_initializer(mean=0., stddev=1.0)
                        if params["use_equalized_learning_rate"]
                        else "he_normal"
                    ),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    equalized_learning_rate=params["use_equalized_learning_rate"],
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
            print_obj("\n" + func_name, "conv_layers", conv_layers)

        return conv_layers

    def instantiate_img_to_vec_layers(self, params):
        """Instantiates layers of img_to_vec network.

        Args:
            params: dict, user passed parameters.

        Returns:
            from_rgb_conv_layers: list, fromRGB 1x1 `Conv2D` layers.
            conv_layer_blocks: list, lists of `Conv2D` block layers for each
                block.
            flatten_layer: `Flatten` layer prior to logits layer.
            logits_layer: `Dense` layer for logits.
        """
        func_name = "instantiate_{}_layers".format(self.kind)

        # Instantiate fromRGB 1x1 `Conv2D` layers.
        from_rgb_conv_layers = self.instantiate_img_to_vec_from_rgb_layers(
            params=params
        )
        print_obj(
            "\n" + func_name, "from_rgb_conv_layers", from_rgb_conv_layers
        )

        # Instantiate base conv block's `Conv2D` layers, for post-growth.
        conv_layer_blocks = [
            self.instantiate_img_to_vec_base_conv_layer_block(
                params=params
            )
        ]

        # Instantiate growth `Conv2D` layer blocks.
        conv_layer_blocks.extend(
            [
                self.instantiate_img_to_vec_growth_layer_block(
                    params=params,
                    block_idx=block_idx
                )
                for block_idx in range(
                    len(params["{}_growth_conv_blocks".format(self.kind)])
                )
            ]
        )
        print_obj(
            func_name, "conv_layer_blocks", conv_layer_blocks
        )

        # Instantiate `Flatten` and `Dense` logits layers.
        (flatten_layer,
         logits_layer) = self.instantiate_img_to_vec_logits_layer(
            params=params
        )
        print_obj(func_name, "flatten_layer", flatten_layer)
        print_obj(func_name, "logits_layer", logits_layer)

        return (from_rgb_conv_layers,
                conv_layer_blocks,
                flatten_layer,
                logits_layer)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def build_img_to_vec_from_rgb_layers(self, params):
        """Creates img_to_vec fromRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from fromRGB 1x1 `Conv2D` layers.
        """
        func_name = "build_{}_from_rgb_layers".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get fromRGB layer properties.
            from_rgb = [
                params["{}_from_rgb_layers".format(self.kind)][i][0][:]
                for i in range(
                    len(params["{}_from_rgb_layers".format(self.kind)])
                )
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
                "\n" + func_name,
                "from_rgb_conv_tensors",
                from_rgb_conv_tensors
            )

        return from_rgb_conv_tensors

    def build_img_to_vec_growth_layer_block(self, params, block_idx):
        """Creates img_to_vec growth block.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of tensors from growth block `Conv2D` layers.
        """
        func_name = "build_{}_growth_layer_block".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["{}_growth_conv_blocks".format(self.kind)][block_idx]

            # Create new inner convolutional layers.
            conv_tensors = [
                self.conv_layer_blocks[1 + block_idx][i](
                    inputs=tf.zeros(
                        shape=[1] + conv_block[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(conv_block))
            ]
            print_obj("\n" + func_name, "conv_tensors", conv_tensors)

        return conv_tensors

    def build_img_to_vec_logits_layer(self, params):
        """Builds flatten and logits layer internals using call.

        Args:
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of img_to_vec.
        """
        func_name = "build_{}_logits_layer".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            block_conv_size = params["{}_base_conv_blocks".format(self.kind)][-1][-1][3]

            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(
                inputs=tf.zeros(
                    shape=[1, 1, 1, block_conv_size],
                    dtype=tf.float32
                )
            )
            print_obj("\n" + func_name, "block_conv_flat", block_conv_flat)

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj(func_name, "logits", logits)

        return logits

    def build_img_to_vec_layers(self, params):
        """Builds img_to_vec layer internals.

        Args:
            params: dict, user passed parameters.

        Returns:
            Logits tensor.
        """
        func_name = "build_{}_layers".format(self.kind)

        # Build fromRGB 1x1 `Conv2D` layers internals through call.
        from_rgb_conv_tensors = self.build_img_to_vec_from_rgb_layers(
            params=params
        )
        print_obj(
            "\n" + func_name, "from_rgb_conv_tensors", from_rgb_conv_tensors
        )

        with tf.control_dependencies(control_inputs=from_rgb_conv_tensors):
            # Create base convolutional block's layer internals using call.
            conv_block_tensors = [
                self.build_img_to_vec_base_conv_layer_block(
                    params=params
                )
            ]

            # Build growth `Conv2D` layer block internals through call.
            conv_block_tensors.extend(
                [
                    self.build_img_to_vec_growth_layer_block(
                        params=params, block_idx=block_idx
                    )
                    for block_idx in range(
                       len(params["{}_growth_conv_blocks".format(self.kind)])
                    )
                ]
            )

            # Flatten conv block tensor lists of lists into list.
            conv_block_tensors = [
                item for sublist in conv_block_tensors for item in sublist
            ]
            print_obj(func_name, "conv_block_tensors", conv_block_tensors)

            with tf.control_dependencies(control_inputs=conv_block_tensors):
                # Build logits layer internals using call.
                logits_tensor = self.build_img_to_vec_logits_layer(
                    params=params
                )
                print_obj(func_name, "logits_tensor", logits_tensor)

        return logits_tensor

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def use_img_to_vec_logits_layer(self, block_conv, params):
        """Uses flatten and logits layers to get logits tensor.

        Args:
            block_conv: tensor, output of last conv layer of img_to_vec.
            flatten_layer: `Flatten` layer.
            logits_layer: `Dense` layer for logits.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of img_to_vec.
        """
        func_name = "use_{}_logits_layer".format(self.kind)

        print_obj("\n" + func_name, "block_conv", block_conv)
        # Set shape to remove ambiguity for dense layer.
        block_conv.set_shape(
            [
                block_conv.get_shape()[0],
                params["generator_projection_dims"][0] / 4,
                params["generator_projection_dims"][1] / 4,
                block_conv.get_shape()[-1]]
        )
        print_obj(func_name, "block_conv", block_conv)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten final block conv tensor.
            block_conv_flat = self.flatten_layer(inputs=block_conv)
            print_obj(func_name, "block_conv_flat", block_conv_flat)

            # Final linear layer for logits.
            logits = self.logits_layer(inputs=block_conv_flat)
            print_obj(func_name, "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def create_base_img_to_vec_block_and_logits(self, block_conv, params):
        """Creates base img_to_vec block and logits.

        Args:
            block_conv: tensor, output of previous `Conv2D` block's layer.
            params: dict, user passed parameters.
        Returns:
            Final logits tensor of img_to_vec.
        """
        func_name = "create_base_{}_block_and_logits".format(self.kind)
        print_obj("\n" + func_name, "block_conv", block_conv)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the first conv layer block for base network.
            block_layers = self.conv_layer_blocks[0]

            for i in range(len(block_layers)):
                block_conv = block_layers[i](inputs=block_conv)
                print_obj(func_name, "block_conv", block_conv)

                block_conv = tf.nn.leaky_relu(
                    features=block_conv,
                    alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                    name="{}_base_layers_conv2d_{}_leaky_relu".format(
                        self.kind, i
                    )
                )
                print_obj(func_name, "block_conv_leaky", block_conv)

            # Get logits now.
            logits = self.use_img_to_vec_logits_layer(
                block_conv=block_conv,
                params=params
            )
            print_obj(func_name, "logits", logits)

        return logits

    def create_growth_transition_img_to_vec_weighted_sum(
            self, X, alpha_var, params, trans_idx):
        """Creates growth transition img_to_vec weighted_sum

        Args:
            X: tensor, input image to img_to_vec.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Tensor of weighted sum between shrinking and growing block paths.
        """
        func_name = "create_growth_transition_{}_weighted_sum".format(
            self.kind
        )

        print_obj("\nEntered {}".format(func_name), "trans_idx", trans_idx)
        print_obj(func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Growing side chain.
            growing_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx + 1]
            growing_block_layers = self.conv_layer_blocks[trans_idx + 1]

            # Pass inputs through layer chain.
            growing_block_conv = growing_from_rgb_conv_layer(inputs=X)
            print_obj(
                "\n" + func_name, "growing_block_conv", growing_block_conv
            )

            growing_block_conv = tf.nn.leaky_relu(
                features=growing_block_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_growth_growing_from_rgb_{}_leaky_relu".format(
                    self.kind, trans_idx
                )
            )
            print_obj(func_name, "growing_block_conv_leaky", growing_block_conv)

            for i in range(len(growing_block_layers)):
                growing_block_conv = growing_block_layers[i](
                    inputs=growing_block_conv
                )
                print_obj(
                    func_name, "growing_block_conv", growing_block_conv
                )

                growing_block_conv = tf.nn.leaky_relu(
                    features=growing_block_conv,
                    alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                    name="{}_growth_conv_2d_{}_{}_leaky_relu".format(
                        self.kind, trans_idx, i
                    )
                )
                print_obj(func_name, "growing_block_conv_leaky", growing_block_conv)

            # Down sample from 2s X 2s to s X s image.
            growing_block_conv_downsampled = tf.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_growing_downsampled_image_{}".format(
                    self.name,
                    trans_idx
                )
            )(inputs=growing_block_conv)
            print_obj(
                func_name,
                "growing_block_conv_downsampled",
                growing_block_conv_downsampled
            )

            # Shrinking side chain.
            shrinking_from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx]

            # Pass inputs through layer chain.
            # Down sample from 2s X 2s to s X s image.
            X_downsampled = tf.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_shrinking_downsampled_image_{}".format(
                    self.name,
                    trans_idx
                )
            )(inputs=X)
            print_obj(func_name, "X_downsampled", X_downsampled)

            shrinking_from_rgb_conv = shrinking_from_rgb_conv_layer(
                inputs=X_downsampled
            )
            print_obj(
                func_name, "shrinking_from_rgb_conv", shrinking_from_rgb_conv
            )

            shrinking_from_rgb_conv = tf.nn.leaky_relu(
                features=shrinking_from_rgb_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_growth_shrinking_from_rgb_{}_leaky_relu".format(
                    self.kind, trans_idx
                )
            )
            print_obj(
                func_name,
                "shrinking_from_rgb_conv_leaky",
                shrinking_from_rgb_conv
            )

            # Weighted sum.
            weighted_sum = tf.add(
                x=growing_block_conv_downsampled * alpha_var,
                y=shrinking_from_rgb_conv * (1.0 - alpha_var),
                name="{}_growth_transition_weighted_sum_{}".format(
                    self.name, trans_idx
                )
            )
            print_obj(func_name, "weighted_sum", weighted_sum)

        return weighted_sum

    def create_img_to_vec_perm_growth_block_network(
            self, block_conv, params, trans_idx):
        """Creates img_to_vec permanent block network.

        Args:
            block_conv: tensor, output of previous block's layer.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Tensor from final permanant block `Conv2D` layer.
        """
        func_name = "create_{}_perm_block_network".format(self.kind)

        print_obj("\nEntered {}".format(func_name), "trans_idx", trans_idx)
        print_obj(func_name, "block_conv", block_conv)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get permanent growth blocks, so skip the base block.
            permanent_blocks = self.conv_layer_blocks[1:trans_idx + 1]

            # Reverse order of blocks.
            permanent_blocks = permanent_blocks[::-1]

            # Pass inputs through layer chain.

            # Loop through the permanent growth blocks.
            for i in range(len(permanent_blocks)):
                # Get layers from ith permanent block.
                permanent_block_layers = permanent_blocks[i]

                # Loop through layers of ith permanent block.
                for j in range(len(permanent_block_layers)):
                    block_conv = permanent_block_layers[j](inputs=block_conv)
                    print_obj(func_name, "block_conv_{}".format(i), block_conv)

                    block_conv = tf.nn.leaky_relu(
                        features=block_conv,
                        alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                        name="{}_perm_conv_2d_{}_{}_{}_leaky_relu".format(
                            self.kind, trans_idx, i, j
                        )
                    )
                    print_obj(func_name, "block_conv_leaky", block_conv)

                # Down sample from 2s X 2s to s X s image.
                block_conv = tf.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    name="{}_perm_conv_downsample_{}_{}".format(
                        self.name, trans_idx, i
                    )
                )(inputs=block_conv)
                print_obj(func_name, "block_conv_downsampled", block_conv)

        return block_conv

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def unknown_switch_case_img_to_vec_logits(
            self, X, alpha_var, params, growth_index):
        """Uses switch case to use the correct network to get logits.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            growth_index: tensor, current growth stage.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        func_name = "unknown_switch_case_{}_logits".format(self.kind)
        # Switch to case based on number of steps to get logits.
        logits = tf.switch_case(
            branch_index=growth_index,
            branch_fns=[
                # 4x4
                lambda: self.create_base_img_to_vec_network(
                    X=X, params=params
                ),
                # 8x8
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(0, len(params["conv_num_filters"]) - 2)
                ),
                # 8x8
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(0, len(params["conv_num_filters"]) - 2)
                ),
                # 16x16
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(1, len(params["conv_num_filters"]) - 2)
                ),
                # 16x16
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(1, len(params["conv_num_filters"]) - 2)
                ),
                # 32x32
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(2, len(params["conv_num_filters"]) - 2)
                ),
                # 32x32
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(2, len(params["conv_num_filters"]) - 2)
                ),
                # 64x64
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(3, len(params["conv_num_filters"]) - 2)
                ),
                # 64x64
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(3, len(params["conv_num_filters"]) - 2)
                ),
                # 128x128
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(4, len(params["conv_num_filters"]) - 2)
                ),
                # 128x128
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(4, len(params["conv_num_filters"]) - 2)
                ),
                # 256x256
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(5, len(params["conv_num_filters"]) - 2)
                ),
                # 256x256
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(5, len(params["conv_num_filters"]) - 2)
                ),
                # 512x512
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(6, len(params["conv_num_filters"]) - 2)
                ),
                # 512x512
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(6, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(7, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_growth_stable_img_to_vec_network(
                    X=X,
                    params=params,
                    trans_idx=min(7, len(params["conv_num_filters"]) - 2)
                )
            ],
            name="{}_switch_case_logits".format(self.name)
        )
        print_obj("\n" + func_name, "logits", logits)

        return logits

    def known_switch_case_img_to_vec_logits(self, X, alpha_var, params):
        """Uses switch case to use the correct network to get logits.

        Args:
            X: tensor, image tensors of shape
                [batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [batch_size, 1].
        """
        func_name = "switch_case_{}_logits".format(self.kind)

        # Switch to case based on number of steps to get logits.
        if params["growth_idx"] == 0:
            # No growth yet, just base block.
            logits = self.create_base_img_to_vec_network(X=X, params=params)
        else:
            # Determine which growth transition we're in.
            trans_idx = (params["growth_idx"] - 1) // 2

            # If there is more room to grow.
            if params["growth_idx"] % 2 == 1:
                # Grow network using weighted sum with smaller network.
                logits = self.create_growth_transition_img_to_vec_network(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=trans_idx
                )
            else:
                # Stablize bigger network without weighted sum.
                logits = self.create_growth_stable_img_to_vec_network(
                    X=X, params=params, trans_idx=trans_idx
                )
        print_obj("\n" + func_name, "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_train_eval_img_to_vec_logits(self, X, alpha_var, params):
        """Uses generator network and returns generated output for train/eval.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        func_name = "get_train_eval_{}_logits".format(self.kind)

        print_obj("\n" + func_name, "X", X)

        # Get img_to_vec's logits tensor.
        if len(params["conv_num_filters"]) == 1:
            print(
                "\n {}: NOT GOING TO GROW, SKIP SWITCH CASE!".format(
                    func_name
                )
            )
            # If never going to grow, no sense using the switch case.
            # 4x4
            logits = self.create_base_img_to_vec_network(X=X, params=params)
        else:
            if params["growth_idx"] is not None:
                logits = self.known_switch_case_img_to_vec_logits(
                    X=X, alpha_var=alpha_var, params=params
                )
            else:
                # Find growth index based on global step and growth frequency.
                growth_index = tf.minimum(
                    x=tf.cast(
                        x=tf.floordiv(
                            x=tf.train.get_or_create_global_step() - 1,
                            y=params["num_steps_until_growth"],
                            name="{}_global_step_floordiv".format(self.name)
                        ),
                        dtype=tf.int32),
                    y=(len(params["conv_num_filters"]) - 1) * 2,
                    name="{}_growth_index".format(self.name)
                )

                # Switch to case based on number of steps for logits.
                logits = self.unknown_switch_case_img_to_vec_logits(
                    X=X,
                    alpha_var=alpha_var,
                    params=params,
                    growth_index=growth_index
                )
        print_obj("\n" + func_name, "logits", logits)

        # Wrap logits in a control dependency for the build img_to_vec
        # tensors to ensure img_to_vec internals are built.
        with tf.control_dependencies(
                control_inputs=[self.build_img_to_vec_tensors]):
            logits = tf.identity(
                input=logits, name="{}_logits_identity".format(self.name)
            )

        return logits
