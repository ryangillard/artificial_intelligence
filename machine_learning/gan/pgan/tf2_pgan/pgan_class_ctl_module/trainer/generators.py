import tensorflow as tf

from . import custom_layers


class Generator(object):
    """Generator that takes latent vector input and outputs image.

    Attributes:
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        input_layer: `Input` layer of every generator model.
        projection_dense_layer: `WeightScaledDense` layer used for projecting noise
            latent vectors into an image.
        conv_layers: list, `Conv2D` layers.
        leaky_relu_layers: list, leaky relu layers that follow `Conv2D`
            layers.
        to_rgb_conv_layers: list, `Conv2D` toRGB layers.
        models: list, instances of generator `Model`s for each growth.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        num_growths,
    ):
        """Instantiates and builds generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of generator.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            num_growths: int, number of growth phases for model.
        """
        # Set name of generator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        # Store lists of layers.
        self.input_layer = None
        self.projection_dense_layer = None
        self.conv_layers = []
        self.leaky_relu_layers = []
        self.to_rgb_conv_layers = []

        # Instantiate generator layers.
        self._create_generator_layers()

        # Store list of discriminator models.
        self.models = self._create_models(num_growths)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_input_layer(self):
        """Creates `Input` layer.

        Returns:
            Instance of `Input` layer.
        """
        return tf.keras.Input(
            shape=(self.params["generator_latent_size"],),
            name="{}_inputs".format(self.name)
        )

    def _create_projection_dense_layer(self):
        """Creates projection dense layer.
        
        Dense layer converts latent vectors to flattened images.

        Returns:
            Instance of `WeightScaledDense` layer.
        """
        projection_height = self.params["generator_projection_dims"][0]
        projection_width = self.params["generator_projection_dims"][1]
        projection_depth = self.params["generator_projection_dims"][2]

        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        return custom_layers.WeightScaledDense(
            units=projection_height * projection_width * projection_depth,
            activation=None,
            kernel_initializer=(
                tf.random_normal_initializer(mean=0., stddev=1.0)
                if self.params["use_equalized_learning_rate"]
                else "he_normal"
            ),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_equalized_learning_rate=(
                self.params["use_equalized_learning_rate"]
            ),
            name="projection_dense_layer"
        )

    def _create_base_conv_layer_block(self):
        """Creates generator base conv layer block.

        Returns:
            List of base block conv layers and list of leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = self.params["generator_base_conv_blocks"][0]

        # Create list of base conv layers.
        base_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["use_equalized_learning_rate"]
                ),
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

        base_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params["generator_leaky_relu_alpha"],
                name="{}_base_layers_leaky_relu_{}".format(self.name, i)
            )
            for i in range(len(conv_block))
        ]

        return base_conv_layers, base_leaky_relu_layers

    def _create_growth_conv_layer_block(self, block_idx):
        """Creates generator growth conv layer block.

        Args:
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block's conv layers and list of growth block's
                leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = self.params["generator_growth_conv_blocks"][block_idx]

        # Create new growth convolutional layers.
        growth_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=conv_block[i][3],
                kernel_size=conv_block[i][0:2],
                strides=conv_block[i][4:6],
                padding="same",
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["use_equalized_learning_rate"]
                ),
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

        growth_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params["generator_leaky_relu_alpha"],
                name="{}_growth_layers_leaky_relu_{}_{}".format(
                    self.name, block_idx, i
                )
            )
            for i in range(len(conv_block))
        ]

        return growth_conv_layers, growth_leaky_relu_layers

    def _create_to_rgb_layers(self):
        """Creates generator toRGB layers of 1x1 convs.

        Returns:
            List of toRGB 1x1 conv layers.
        """
        # Dictionary containing possible final activations.
        final_activation_set = {"sigmoid", "relu", "tanh"}

        # Get toRGB layer properties.
        to_rgb = [
            self.params["generator_to_rgb_layers"][i][0][:]
            for i in range(
                len(self.params["generator_to_rgb_layers"])
            )
        ]

        # Create list to hold toRGB 1x1 convs.
        to_rgb_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=to_rgb[i][3],
                kernel_size=to_rgb[i][0:2],
                strides=to_rgb[i][4:6],
                padding="same",
                activation=(
                    self.params["generator_final_activation"].lower()
                    if self.params["generator_final_activation"].lower()
                    in final_activation_set
                    else None
                ),
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if self.params["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_equalized_learning_rate=(
                    self.params["use_equalized_learning_rate"]
                ),
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

        return to_rgb_conv_layers

    def _create_generator_layers(self):
        """Creates generator layers.
        """
        self.input_layer = self._create_input_layer()

        self.projection_dense_layer = self._create_projection_dense_layer()

        (base_conv_layers,
         base_leaky_relu_layers) = self._create_base_conv_layer_block()
        self.conv_layers.append(base_conv_layers)
        self.leaky_relu_layers.append(base_leaky_relu_layers)

        for block_idx in range(
            len(self.params["generator_growth_conv_blocks"])
        ):
            (growth_conv_layers,
             growth_leaky_relu_layers
             ) = self._create_growth_conv_layer_block(block_idx)

            self.conv_layers.append(growth_conv_layers)
            self.leaky_relu_layers.append(growth_leaky_relu_layers)

        self.to_rgb_conv_layers = self._create_to_rgb_layers()

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _maybe_use_pixel_norm(self, inputs, epsilon=1e-8):
        """Decides based on user parameter whether to use pixel norm or not.

        Args:
            epsilon: float, small value to add to denominator for numerical
                stability.
            inputs: tensor, image to potentially be normalized.

        Returns:
            Pixel normalized feature vectors if using pixel norm, else
                original feature vectors.
        """
        if self.params["generator_use_pixel_norm"]:
            return custom_layers.PixelNormalization(epsilon=epsilon)(
                inputs=inputs
            )
        return inputs

    def _project_latent_vectors(self, latent_vectors):
        """Defines generator network.

        Args:
            latent_vectors: tensor, latent vector inputs of shape
                [batch_size, latent_size].

        Returns:
            Projected image of latent vector inputs.
        """
        # Possibly normalize latent vectors.
        if self.params["generator_normalize_latents"]:
            latent_vectors = self._maybe_use_pixel_norm(
                inputs=latent_vectors,
                epsilon=self.params["generator_pixel_norm_epsilon"]
            )

        projection_height = self.params["generator_projection_dims"][0]
        projection_width = self.params["generator_projection_dims"][1]
        projection_depth = self.params["generator_projection_dims"][2]

        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection = self.projection_dense_layer(inputs=latent_vectors)

        projection_leaky_relu = tf.keras.layers.LeakyReLU(
            alpha=self.params["generator_leaky_relu_alpha"],
            name="projection_leaky_relu"
        )(inputs=projection)

        # Reshape projection into "image".
        # shape = (
        #     batch_size,
        #     projection_height,
        #     projection_width,
        #     projection_depth
        # )
        projected_image = tf.reshape(
            tensor=projection_leaky_relu,
            shape=[
                -1, projection_height, projection_width, projection_depth
            ],
            name="projected_image"
        )

        # Possibly add pixel normalization to image.
        projected_image = self._maybe_use_pixel_norm(
            inputs=projected_image,
            epsilon=self.params["generator_pixel_norm_epsilon"]
        )

        return projected_image

    def _upsample_generator_image(self, image, orig_img_size, block_idx):
        """Upsamples generator intermediate image.
        Args:
            image: tensor, image created by vec_to_img conv block.
            orig_img_size: list, the height and width dimensions of the
                original image before any growth.
            block_idx: int, index of the current vec_to_img growth block.
        Returns:
            Upsampled image tensor.
        """
        # Upsample from s X s to 2s X 2s image.
        upsampled_image = tf.image.resize(
            images=image,
            size=tf.convert_to_tensor(
                value=orig_img_size,
                dtype=tf.int32
            ) * 2 ** block_idx,
            method="nearest",
            name="{}_growth_upsampled_image_{}_{}x{}_{}x{}".format(
                self.name,
                block_idx,
                orig_img_size[0] * 2 ** (block_idx - 1),
                orig_img_size[1] * 2 ** (block_idx - 1),
                orig_img_size[0] * 2 ** block_idx,
                orig_img_size[1] * 2 ** block_idx
            )
        )

        return upsampled_image

    def _fused_conv2d_act_pixel_norm_block(
        self, conv_layer, activation_layer, inputs
    ):
        """Fused Conv2D, activation, and pixel norm operation block.

        Args:
            conv_layer: instance of `Conv2D` layer.
            activation_layer: instance of `Layer`, such as LeakyRelu layer.
            inputs: tensor, inputs to fused block.

        Returns:
            Output tensor of fused block.
        """
        # Perform convolution with no activation.
        network = conv_layer(inputs=inputs)

        # Now apply activation to convolved inputs.
        network = activation_layer(inputs=network)

        # Possibly add pixel normalization to image.
        network = self._maybe_use_pixel_norm(
            inputs=network,
            epsilon=self.params["generator_pixel_norm_epsilon"]
        )

        return network

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _build_base_model(self):
        """Builds generator base model.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to generator.
        # shape = (batch_size, latent_size)
        inputs = self.input_layer

        # Project latent vectors.
        network = self._project_latent_vectors(latent_vectors=inputs)

        # Get base block layers.
        base_conv_layers = self.conv_layers[0]
        base_leaky_relu_layers = self.leaky_relu_layers[0]
        base_to_rgb_conv_layer = self.to_rgb_conv_layers[0]

        # Pass inputs through layer chain.
        for i in range(len(base_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=base_conv_layers[i],
                activation_layer=base_leaky_relu_layers[i],
                inputs=network
            )

        fake_images = base_to_rgb_conv_layer(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=fake_images,
            name="{}_base".format(self.name)
        )

        return model

    def _build_growth_transition_model(self, block_idx):
        """Builds generator growth transition model.

        Args:
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to generator.
        # shape = (batch_size, latent_size)
        inputs = self.input_layer

        # Project latent vectors.
        network = self._project_latent_vectors(latent_vectors=inputs)

        # Permanent blocks.
        permanent_conv_layers = self.conv_layers[0:block_idx]
        permanent_leaky_relu_layers = self.leaky_relu_layers[0:block_idx]

        # Base block doesn't need any upsampling so handle differently.
        base_conv_layers = permanent_conv_layers[0]
        base_leaky_relu_layers = permanent_leaky_relu_layers[0]

        # Pass inputs through layer chain.
        for i in range(len(base_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=base_conv_layers[i],
                activation_layer=base_leaky_relu_layers[i],
                inputs=network
            )

        # Growth blocks require first prev conv layer's image upsampled.
        for i in range(1, len(permanent_conv_layers)):
            # Upsample previous block's image.
            network = self._upsample_generator_image(
                image=network,
                orig_img_size=self.params["generator_projection_dims"][0:2],
                block_idx=i
            )

            block_conv_layers = permanent_conv_layers[i]
            block_leaky_relu_layers = permanent_leaky_relu_layers[i]
            for j in range(0, len(block_conv_layers)):
                network = self._fused_conv2d_act_pixel_norm_block(
                    conv_layer=block_conv_layers[j],
                    activation_layer=block_leaky_relu_layers[j],
                    inputs=network
                )

        # Upsample most recent block conv image for both side chains.
        upsampled_block_conv = self._upsample_generator_image(
            image=network,
            orig_img_size=self.params["generator_projection_dims"][0:2],
            block_idx=len(permanent_conv_layers)
        )

        # Growing side chain.
        growing_conv_layers = self.conv_layers[block_idx]
        growing_leaky_relu_layers = self.leaky_relu_layers[block_idx]
        growing_to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx]

        # Pass inputs through layer chain.
        network = upsampled_block_conv
        for i in range(0, len(growing_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=growing_conv_layers[i],
                activation_layer=growing_leaky_relu_layers[i],
                inputs=network
            )

        growing_to_rgb_conv = growing_to_rgb_conv_layer(inputs=network)

        # Shrinking side chain.
        shrinking_to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx - 1]

        # Pass inputs through layer chain.
        shrinking_to_rgb_conv = shrinking_to_rgb_conv_layer(
            inputs=upsampled_block_conv
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=growing_to_rgb_conv * self.alpha_var,
            y=shrinking_to_rgb_conv * (1.0 - self.alpha_var),
            name="{}_growth_transition_weighted_sum_{}".format(
                self.name, block_idx
            )
        )

        fake_images = weighted_sum

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=fake_images,
            name="{}_growth_transition_{}".format(self.name, block_idx)
        )

        return model

    def _build_growth_stable_model(self, block_idx):
        """Builds generator growth stable model.

        Args:
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to generator.
        # shape = (batch_size, latent_size)
        inputs = self.input_layer

        # Project latent vectors.
        network = self._project_latent_vectors(latent_vectors=inputs)

        # Permanent blocks.
        permanent_conv_layers = self.conv_layers[0:block_idx + 1]
        permanent_leaky_relu_layers = self.leaky_relu_layers[0:block_idx + 1]

        # Base block doesn't need any upsampling so handle differently.
        base_conv_layers = permanent_conv_layers[0]
        base_leaky_relu_layers = permanent_leaky_relu_layers[0]

        # Pass inputs through layer chain.
        for i in range(len(base_conv_layers)):
            network = self._fused_conv2d_act_pixel_norm_block(
                conv_layer=base_conv_layers[i],
                activation_layer=base_leaky_relu_layers[i],
                inputs=network
            )

        # Growth blocks require first prev conv layer's image upsampled.
        for i in range(1, len(permanent_conv_layers)):
            # Upsample previous block's image.
            network = self._upsample_generator_image(
                image=network,
                orig_img_size=self.params["generator_projection_dims"][0:2],
                block_idx=i
            )

            block_conv_layers = permanent_conv_layers[i]
            block_leaky_relu_layers = permanent_leaky_relu_layers[i]
            for j in range(0, len(block_conv_layers)):
                network = self._fused_conv2d_act_pixel_norm_block(
                    conv_layer=block_conv_layers[j],
                    activation_layer=block_leaky_relu_layers[j],
                    inputs=network
                )

        # Get toRGB layer.
        to_rgb_conv_layer = self.to_rgb_conv_layers[block_idx]

        fake_images = to_rgb_conv_layer(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=fake_images,
            name="{}_growth_stable_{}".format(self.name, block_idx)
        )

        return model

    def _create_models(self, num_growths):
        """Creates list of generator's `Model` objects for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `Generator` `Model` objects.
        """
        models = []
        for growth_idx in range(num_growths):
            block_idx = (growth_idx + 1) // 2
            input_shape = (self.params["generator_latent_size"],)

            if growth_idx == 0:
                model = self._build_base_model()
            elif growth_idx % 2 == 1:
                model = self._build_growth_transition_model(block_idx)
            elif growth_idx % 2 == 0:
                model = self._build_growth_stable_model(block_idx)

            models.append(model)

        return models

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_generator_loss(
        self,
        global_batch_size,
        fake_logits,
        global_step,
        summary_file_writer,
        growth_idx
    ):
        """Gets generator loss.

        Args:
            global_batch_size: int, global batch size for distribution.
            fake_logits: tensor, shape of
                [batch_size, 1].
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.
            growth_idx: int, current growth index model has progressed to.

        Returns:
            Tensor of generator's total loss of shape [].
        """
        if self.params["distribution_strategy"]:
            # Calculate base generator loss.
            generator_loss = tf.nn.compute_average_loss(
                per_example_loss=-fake_logits,
                global_batch_size=global_batch_size
            )

            # Get regularization losses.
            generator_reg_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum(self.models[growth_idx].losses)
            )
        else:
            # Calculate base generator loss.
            generator_loss = -tf.reduce_mean(
                input_tensor=fake_logits,
                name="{}_loss".format(self.name)
            )

            # Get regularization losses.
            generator_reg_loss = sum(self.models[growth_idx].losses)

        # Combine losses for total losses.
        generator_total_loss = tf.math.add(
            x=generator_loss,
            y=generator_reg_loss,
            name="generator_total_loss"
        )

        if self.params["write_summaries"]:
            # Add summaries for TensorBoard.
            with summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=global_step, y=self.params["save_summary_steps"]
                        ),
                        y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/generator_loss",
                        data=generator_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/generator_reg_loss",
                        data=generator_reg_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="optimized_losses/generator_total_loss",
                        data=generator_total_loss,
                        step=global_step
                    )
                    summary_file_writer.flush()

        return generator_total_loss
