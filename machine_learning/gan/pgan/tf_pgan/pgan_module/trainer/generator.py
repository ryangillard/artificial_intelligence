import tensorflow as tf

from . import regularization
from .print_object import print_obj


class Generator(object):
    """
    Fields:
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        projection_layer: `Dense` layer for projection of noise to image.
        conv_layer_blocks: list, lists of block layers for each block.
        to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        build_generator_tensors: list, tensors used to build layer internals.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Creates generator network and returns generated output.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of generator.
        """
        # Set name of generator.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

        # Instantiate generator layers.
        (self.projection_layer,
         self.conv_layer_blocks,
         self.to_rgb_conv_layers) = self.instantiate_generator_layers(params)

        # Build generator layer internals.
        self.build_generator_tensors = self.build_generator_layers(
            params
        )

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def instantiate_generator_projection_layer(self, params):
        """Instantiates generator projection layer.

        Projection layer projects latent noise vector into an image.

        Args:
            params: dict, user passed parameters.

        Returns:
            Latent vector projection `Dense` layer.
        """
        # Project latent vectors.
        projection_height = params["generator_projection_dims"][0]
        projection_width = params["generator_projection_dims"][1]
        projection_depth = params["generator_projection_dims"][2]

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # shape = (
            #     cur_batch_size,
            #     projection_height * projection_width * projection_depth
            # )
            projection_layer = tf.layers.Dense(
                units=projection_height * projection_width * projection_depth,
                activation=tf.nn.leaky_relu,
                kernel_initializer="he_normal",
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_projection_layer".format(self.name)
            )
            print_obj(
                "\ninstantiate_generator_projection_layer",
                "projection_layer",
                projection_layer
            )

        return projection_layer

    def instantiate_generator_base_conv_layer_block(self, params):
        """Instantiates generator base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of base block conv layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                for i in range(len(conv_block))
            ]
            print_obj(
                "\ninstantiate_generator_base_conv_layer_block",
                "base_conv_layers",
                base_conv_layers
            )

        return base_conv_layers

    def instantiate_generator_growth_layer_block(self, params, block_idx):
        """Instantiates generator growth layer block.

        Args:
            params: dict, user passed parameters.
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block conv layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                "\ninstantiate_generator_growth_layer_block",
                "conv_layers",
                conv_layers
            )

        return conv_layers

    def instantiate_generator_to_rgb_layers(self, params):
        """Instantiates generator toRGB layers of 1x1 convs.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of toRGB 1x1 conv layers.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
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
                    # Notice there is no activation for toRGB conv layers.
                    activation=None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="{}_to_rgb_layers_conv2d_{}_{}x{}_{}_{}".format(
                        self.name,
                        i,
                        to_rgb[i][0],
                        to_rgb[i][1],
                        to_rgb[i][2],
                        to_rgb[i][3]
                    )
                )
                for i in range(len(to_rgb))
            ]
            print_obj(
                "\ninstantiate_generator_to_rgb_layers",
                "to_rgb_conv_layers",
                to_rgb_conv_layers
            )

        return to_rgb_conv_layers

    def instantiate_generator_layers(self, params):
        """Instantiates layers of generator network.

        Args:
            params: dict, user passed parameters.

        Returns:
            projection_layer: `Dense` layer for projection of noise to image.
            conv_layer_blocks: list, lists of block layers for each block.
            to_rgb_conv_layers: list, toRGB 1x1 conv layers.
        """
        # Instantiate noise-image projection `Dense` layer.
        projection_layer = self.instantiate_generator_projection_layer(
            params=params
        )
        print_obj(
            "\ninstantiate_generator_layers",
            "projection_layer",
            projection_layer
        )

        # Instantiate base convolutional `Conv2D` layers, for post-growth.
        conv_layer_blocks = [
            self.instantiate_generator_base_conv_layer_block(
                params=params
            )
        ]

        # Instantiate growth block `Conv2D` layers.
        conv_layer_blocks.extend(
            [
                self.instantiate_generator_growth_layer_block(
                    params=params, block_idx=block_idx
                )
                for block_idx in range(
                    len(params["generator_growth_conv_blocks"])
                )
            ]
        )
        print_obj(
            "instantiate_generator_layers",
            "conv_layer_blocks",
            conv_layer_blocks
        )

        # Instantiate toRGB 1x1 `Conv2D` layers.
        to_rgb_conv_layers = self.instantiate_generator_to_rgb_layers(
            params=params
        )
        print_obj(
            "instantiate_generator_layers",
            "to_rgb_conv_layers",
            to_rgb_conv_layers
        )

        return projection_layer, conv_layer_blocks, to_rgb_conv_layers

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def build_generator_projection_layer(self, params):
        """Builds generator projection layer internals using call.

        Args:
            params: dict, user passed parameters.

        Returns:
            Latent vector projection tensor.
        """
        # Project latent vectors.
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # shape = (
            #     cur_batch_size,
            #     projection_height * projection_width * projection_depth
            # )
            projection_tensor = self.projection_layer(
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

    def build_generator_base_conv_layer_block(self, params):
        """Builds generator base conv layer block internals using call.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of base conv tensors.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["generator_base_conv_blocks"][0]

            # Create list of base conv layers.
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
                "\nbuild_generator_base_conv_layer_block",
                "base_conv_tensors",
                base_conv_tensors
            )

        return base_conv_tensors

    def build_generator_growth_layer_block(
            self, params, growth_block_idx):
        """Builds generator growth block internals through call.

        Args:
            params: dict, user passed parameters.
            growth_block_idx: int, the current growth block's index.

        Returns:
            List of growth block tensors.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["generator_growth_conv_blocks"][growth_block_idx]

            # Create new inner convolutional layers.
            conv_tensors = [
                self.conv_layer_blocks[1 + growth_block_idx][i](
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

    def build_generator_to_rgb_layers(self, params):
        """Builds generator toRGB layers of 1x1 convs internals through call.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of toRGB 1x1 conv tensors.
        """
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get toRGB layer properties.
            to_rgb = [
                params["generator_to_rgb_layers"][i][0][:]
                for i in range(len(params["generator_to_rgb_layers"]))
            ]

            # Create list to hold toRGB 1x1 convs.
            to_rgb_conv_tensors = [
                self.to_rgb_conv_layers[i](
                    inputs=tf.zeros(
                        shape=[1] + to_rgb[i][0:3], dtype=tf.float32)
                    )
                for i in range(len(to_rgb))
            ]
            print_obj(
                "\nbuild_generator_to_rgb_layers",
                "to_rgb_conv_tensors",
                to_rgb_conv_tensors
            )

        return to_rgb_conv_tensors

    def build_generator_layers(self, params):
        """Builds generator layer internals.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of toRGB tensors.
        """
        # Build projection layer internals using call.
        projection_tensor = self.build_generator_projection_layer(
            params=params
        )
        print_obj(
            "\nbuild_generator_layers",
            "projection_tensor",
            projection_tensor
        )

        with tf.control_dependencies(control_inputs=[projection_tensor]):
            # Build base convolutional layer block's internals using call.
            conv_block_tensors = [
                self.build_generator_base_conv_layer_block(
                    params=params
                )
            ]

            # Build growth block layer internals through call.
            conv_block_tensors.extend(
                [
                    self.build_generator_growth_layer_block(
                        params=params,
                        growth_block_idx=growth_block_idx
                    )
                    for growth_block_idx in range(
                        len(params["generator_growth_conv_blocks"])
                    )
                ]
            )
            print_obj(
                "build_generator_layers",
                "conv_block_tensors",
                conv_block_tensors
            )

            # Flatten block tensor lists of lists into list.
            conv_block_tensors = [
                item for sublist in conv_block_tensors for item in sublist
            ]
            print_obj(
                "build_generator_layers",
                "conv_block_tensors",
                conv_block_tensors
            )

            with tf.control_dependencies(
                    control_inputs=conv_block_tensors):
                # Build toRGB 1x1 conv layer internals through call.
                to_rgb_conv_tensors = self.build_generator_to_rgb_layers(
                    params=params
                )
                print_obj(
                    "build_generator_layers",
                    "to_rgb_conv_tensors",
                    to_rgb_conv_tensors
                )

        return to_rgb_conv_tensors

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def pixel_norm(self, X, epsilon=1e-8):
        """Normalizes the feature vector in each pixel to unit length.

        Args:
            X: tensor, image feature vectors.
            epsilon: float, small value to add to denominator for numerical
                stability.

        Returns:
            Pixel normalized feature vectors.
        """
        with tf.variable_scope("{}/pixel_norm".format(self.name)):
            return X * tf.rsqrt(
                x=tf.add(
                    x=tf.reduce_mean(
                        input_tensor=tf.square(x=X), axis=1, keepdims=True
                    ),
                    y=epsilon
                )
            )

    def use_pixel_norm(self, X, params, epsilon=1e-8):
        """Decides based on user parameter whether to use pixel norm or not.

        Args:
            X: tensor, image feature vectors.
            params: dict, user passed parameters.
            epsilon: float, small value to add to denominator for numerical
                stability.

        Returns:
            Pixel normalized feature vectors if using pixel norm, else
                original feature vectors.
        """
        if params["use_pixel_norm"]:
            return self.pixel_norm(X=X, epsilon=epsilon)
        else:
            return X

    def use_generator_projection_layer(self, Z, params):
        """Uses projection layer to convert random noise vector into an image.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            params: dict, user passed parameters.

        Returns:
            Latent vector projection tensor.
        """
        # Project latent vectors.
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            if params["normalize_latent"]:
                # shape = (cur_batch_size, latent_size)
                Z = self.pixel_norm(X=Z, epsilon=params["pixel_norm_epsilon"])

            # shape = (
            #     cur_batch_size,
            #     projection_height * projection_width * projection_depth
            # )
            projection_tensor = self.projection_layer(inputs=Z)
            print_obj(
                "\nuse_generator_projection_layer",
                "projection_tensor",
                projection_tensor
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
            shape=[-1] + params["generator_projection_dims"],
            name="{}_projection_reshaped".format(self.name)
        )
        print_obj(
            "use_generator_projection_layer",
            "projection_tensor_reshaped",
            projection_tensor_reshaped
        )

        return projection_tensor_reshaped

    def fused_conv2d_pixel_norm(self, input_image, conv2d_layer, params):
        """Fused `Conv2D` layer and pixel norm operation.

        Args:
            input_image: tensor, input image of rank 4.
            conv2d_layer: `Conv2D` layer.
            params: dict, user passed parameters.

        Returns:
            New image tensor of rank 4.
        """
        conv_output = conv2d_layer(inputs=input_image)
        print_obj("\nfused_conv2d_pixel_norm", "conv_output", conv_output)

        pixel_norm_output = self.use_pixel_norm(
            X=conv_output,
            params=params,
            epsilon=params["pixel_norm_epsilon"]
        )
        print_obj(
            "fused_conv2d_pixel_norm", "pixel_norm_output", pixel_norm_output
        )

        return pixel_norm_output

    def upsample_generator_image(self, image, original_image_size, block_idx):
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
                name="{}_upsample_generator_image_original_image_size".format(
                    self.name
                )
            ) * 2 ** block_idx,
            method="nearest",
            name="{}_growth_upsampled_image_{}_{}x{}_{}x{}".format(
                self.name,
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

    def create_base_generator_network(self, Z, params):
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
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Project latent noise vectors into image.
            projection = self.use_generator_projection_layer(
                Z=Z, params=params
            )
            print_obj(
                "create_base_generator_network", "projection", projection
            )

            # Only need the first block and toRGB conv layer for base network.
            block_layers = self.conv_layer_blocks[0]
            to_rgb_conv_layer = self.to_rgb_conv_layers[0]

            # Pass inputs through layer chain.
            block_conv = projection
            for i in range(0, len(block_layers)):
                block_conv = self.fused_conv2d_pixel_norm(
                    input_image=block_conv,
                    conv2d_layer=block_layers[i],
                    params=params
                )
                print_obj(
                    "create_base_generator_network",
                    "block_conv_{}".format(i),
                    block_conv
                )

            # Convert convolution to RGB image.
            to_rgb_conv = self.fused_conv2d_pixel_norm(
                input_image=block_conv,
                conv2d_layer=to_rgb_conv_layer,
                params=params
            )
            print_obj(
                "create_base_generator_network", "to_rgb_conv", to_rgb_conv
            )

        return to_rgb_conv

    def create_growth_transition_generator_network(
            self, Z, original_image_size, alpha_var, params, trans_idx):
        """Creates growth transition generator network.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            original_image_size: list, the height and width dimensions of the
                original image before any growth.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Weighted sum tensor of growing and shrinking network paths.
        """
        print_obj(
            "\nEntered create_growth_transition_generator_network",
            "trans_idx",
            trans_idx
        )
        print_obj("create_growth_transition_generator_network", "Z", Z)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Project latent noise vectors into image.
            projection = self.use_generator_projection_layer(
                Z=Z, params=params
            )
            print_obj(
                "create_growth_transition_generator_network",
                "projection",
                projection
            )

            # Permanent blocks.
            permanent_blocks = self.conv_layer_blocks[0:trans_idx + 1]

            # Base block doesn't need any upsampling so handle differently.
            base_block_conv_layers = permanent_blocks[0]

            # Pass inputs through layer chain.
            block_conv = projection
            for i in range(0, len(base_block_conv_layers)):
                block_conv = self.fused_conv2d_pixel_norm(
                    input_image=block_conv,
                    conv2d_layer=base_block_conv_layers[i],
                    params=params
                )
                print_obj(
                    "create_growth_transition_generator_network",
                    "base_block_conv_{}_{}".format(trans_idx, i),
                    block_conv
                )

            # Growth blocks require first prev conv layer's image upsampled.
            for i in range(1, len(permanent_blocks)):
                # Upsample previous block's image.
                block_conv = self.upsample_generator_image(
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
                    block_conv = self.fused_conv2d_pixel_norm(
                        input_image=block_conv,
                        conv2d_layer=block_conv_layers[j],
                        params=params
                    )
                    print_obj(
                        "create_growth_transition_generator_network",
                        "block_conv_{}_{}_{}".format(trans_idx, i, j),
                        block_conv
                    )

            # Upsample most recent block conv image for both side chains.
            upsampled_block_conv = self.upsample_generator_image(
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
            growing_block_layers = self.conv_layer_blocks[trans_idx + 1]
            growing_to_rgb_conv_layer = self.to_rgb_conv_layers[trans_idx + 1]

            # Pass inputs through layer chain.
            block_conv = upsampled_block_conv
            for i in range(0, len(growing_block_layers)):
                block_conv = self.fused_conv2d_pixel_norm(
                    input_image=block_conv,
                    conv2d_layer=growing_block_layers[i],
                    params=params
                )
                print_obj(
                    "create_growth_transition_generator_network",
                    "growing_block_conv_{}_{}".format(trans_idx, i),
                    block_conv
                )

            growing_to_rgb_conv = self.fused_conv2d_pixel_norm(
                input_image=block_conv,
                conv2d_layer=growing_to_rgb_conv_layer,
                params=params
            )
            print_obj(
                "create_growth_transition_generator_network",
                "growing_to_rgb_conv_{}".format(trans_idx),
                growing_to_rgb_conv
            )

            # Shrinking side chain.
            shrinking_to_rgb_conv_layer = self.to_rgb_conv_layers[trans_idx]

            # Pass inputs through layer chain.
            shrinking_to_rgb_conv = self.fused_conv2d_pixel_norm(
                input_image=upsampled_block_conv,
                conv2d_layer=shrinking_to_rgb_conv_layer,
                params=params
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

    def create_final_generator_network(self, Z, original_image_size, params):
        """Creates final generator network.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            original_image_size: list, the height and width dimensions of the
                original image before any growth.
            params: dict, user passed parameters.

        Returns:
            Final network block conv tensor.
        """
        print_obj("\ncreate_final_generator_network", "Z", Z)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Project latent noise vectors into image.
            projection = self.use_generator_projection_layer(
                Z=Z, params=params
            )
            print_obj(
                "create_final_generator_network", "projection", projection
            )

            # Base block doesn't need any upsampling so handle differently.
            base_block_conv_layers = self.conv_layer_blocks[0]

            # Pass inputs through layer chain.
            block_conv = projection
            for i in range(0, len(base_block_conv_layers)):
                block_conv = self.fused_conv2d_pixel_norm(
                    input_image=block_conv,
                    conv2d_layer=base_block_conv_layers[i],
                    params=params
                )
                print_obj(
                    "create_final_generator_network",
                    "base_block_conv_{}".format(i),
                    block_conv
                )

            # Growth blocks require first prev conv layer's image upsampled.
            for i in range(1, len(self.conv_layer_blocks)):
                # Upsample previous block's image.
                block_conv = self.upsample_generator_image(
                    image=block_conv,
                    original_image_size=original_image_size,
                    block_idx=i
                )
                print_obj(
                    "create_final_generator_network",
                    "upsample_generator_image_block_conv_{}".format(i),
                    block_conv
                )

                block_conv_layers = self.conv_layer_blocks[i]
                for j in range(0, len(block_conv_layers)):
                    block_conv = self.fused_conv2d_pixel_norm(
                        input_image=block_conv,
                        conv2d_layer=block_conv_layers[j],
                        params=params
                    )
                    print_obj(
                        "create_final_generator_network",
                        "block_conv_{}_{}".format(i, j),
                        block_conv
                    )

            # Only need the last toRGB conv layer.
            to_rgb_conv_layer = self.to_rgb_conv_layers[-1]

            # Pass inputs through layer chain.
            to_rgb_conv = self.fused_conv2d_pixel_norm(
                input_image=block_conv,
                conv2d_layer=to_rgb_conv_layer,
                params=params
            )
            print_obj(
                "create_final_generator_network", "to_rgb_conv", to_rgb_conv
            )

        return to_rgb_conv

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def switch_case_generator_outputs(
            self, Z, original_image_size, alpha_var, params, growth_index):
        """Uses switch case to use the correct network to generate images.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            original_image_size: list, the height and width dimensions of the
                original image before any growth.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            growth_index: int, current growth stage.

        Returns:
            Generated image output tensor.
        """
        # Switch to case based on number of steps for gen outputs.
        generated_outputs = tf.switch_case(
            branch_index=growth_index,
            branch_fns=[
                # 4x4
                lambda: self.create_base_generator_network(
                    Z=Z, params=params
                ),
                # 8x8
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(0, len(params["conv_num_filters"]) - 2)
                ),
                # 16x16
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(1, len(params["conv_num_filters"]) - 2)
                ),
                # 32x32
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(2, len(params["conv_num_filters"]) - 2)
                ),
                # 64x64
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(3, len(params["conv_num_filters"]) - 2)
                ),
                # 128x128
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(4, len(params["conv_num_filters"]) - 2)
                ),
                # 256x256
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(5, len(params["conv_num_filters"]) - 2)
                ),
                # 512x512
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(6, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_growth_transition_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    alpha_var=alpha_var,
                    params=params,
                    trans_idx=min(7, len(params["conv_num_filters"]) - 2)
                ),
                # 1024x1024
                lambda: self.create_final_generator_network(
                    Z=Z,
                    original_image_size=original_image_size,
                    params=params
                )
            ],
            name="{}_switch_case_generated_outputs".format(self.name)
        )

        return generated_outputs

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_train_eval_generator_outputs(self, Z, alpha_var, params):
        """Uses generator network and returns generated output for train/eval.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Generated image output tensor of shape
                [cur_batch_size, image_size, image_size, depth].
        """
        print_obj("\nget_train_eval_generator_outputs", "Z", Z)

        # Get generator's output image tensor.
        train_steps = params["train_steps"] + params["prev_train_steps"]
        num_steps_until_growth = params["num_steps_until_growth"]
        num_stages = train_steps // num_steps_until_growth
        if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
            print(
                "\nget_train_eval_generator_outputs: NOT GOING TO GROW, SKIP SWITCH CASE!"
            )
            # If never going to grow, no sense using the switch case.
            # 4x4
            generated_outputs = self.create_base_generator_network(
                Z=Z, params=params
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

            # Switch to case based on number of steps for gen outputs.
            generated_outputs = self.switch_case_generator_outputs(
                Z=Z,
                original_image_size=params["generator_projection_dims"][0:2],
                alpha_var=alpha_var,
                params=params,
                growth_index=growth_index
            )

        print_obj(
            "\nget_train_eval_generator_outputs",
            "generated_outputs",
            generated_outputs
        )

        # Wrap generated outputs in a control dependency for the build
        # generator tensors to ensure generator internals are built.
        with tf.control_dependencies(
                control_inputs=self.build_generator_tensors):
            generated_outputs = tf.identity(
                input=generated_outputs,
                name="{}_generated_outputs_identity".format(self.name)
            )

        return generated_outputs

    def get_predict_generator_outputs(self, Z, params):
        """Uses generator network and returns generated output for predict.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            params: dict, user passed parameters.

        Returns:
            Generated image output tensor of shape
                [cur_batch_size, image_size, image_size, depth].
        """
        print_obj("\nget_predict_generator_outputs", "Z", Z)

        if params["predict_all_resolutions"]:
            generated_outputs = [
                # 4x4
                self.create_base_generator_network(Z=Z, params=params)
            ]

            generated_outputs.extend(
                [
                    # 8x8 through 1024x1024
                    self.create_growth_transition_generator_network(
                        Z=Z,
                        original_image_size=params["generator_projection_dims"][0:2],
                        alpha_var=tf.ones(shape=[], dtype=tf.float32),
                        params=params,
                        trans_idx=i
                    )
                    for i in range(len(params["conv_num_filters"]) - 1)
                ]
            )
        else:
            # 1024x1024
            generated_outputs = self.create_final_generator_network(
                Z=Z,
                original_image_size=params["generator_projection_dims"][0:2],
                params=params
            )
        print_obj(
            "get_predict_generator_outputs",
            "generated_outputs",
            generated_outputs
        )

        return generated_outputs

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_generator_loss(self, fake_logits, params):
        """Gets generator loss.

        Args:
            fake_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed generator's output image.
            params: dict, user passed parameters.

        Returns:
            Generator's total loss tensor of shape [].
        """
        # Calculate base generator loss.
        generator_loss = -tf.reduce_mean(
            input_tensor=fake_logits,
            name="{}_loss".format(self.name)
        )
        print_obj("\nget_generator_loss", "generator_loss", generator_loss)

        # Get generator regularization losses.
        generator_reg_loss = regularization.get_regularization_loss(
            lambda1=params["generator_l1_regularization_scale"],
            lambda2=params["generator_l2_regularization_scale"],
            scope=self.name
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
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            "get_generator_loss", "generator_total_loss", generator_total_loss
        )

        return generator_total_loss
