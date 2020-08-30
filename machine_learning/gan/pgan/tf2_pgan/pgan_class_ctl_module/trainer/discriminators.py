import tensorflow as tf

from . import custom_layers


class Discriminator(object):
    """Discriminator that takes image input and outputs logits.

    Attributes:
        name: str, name of `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        params: dict, user passed parameters.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        input_layers: list, `Input` layers for each resolution of image.
        from_rgb_conv_layers: list, `Conv2D` fromRGB layers.
        from_rgb_leaky_relu_layers: list, leaky relu layers that follow
            `Conv2D` fromRGB layers.
        conv_layers: list, `Conv2D` layers.
        leaky_relu_layers: list, leaky relu layers that follow `Conv2D`
            layers.
        growing_downsample_layers: list, `AveragePooling2D` layers for growing
            branch.
        shrinking_downsample_layers: list, `AveragePooling2D` layers for
            shrinking branch.
        minibatch_stddev_layer: `MiniBatchStdDev` layer, applies minibatch
            stddev to image to add an additional feature channel based on the
            sample.
        flatten_layer: `Flatten` layer, flattens image for logits layer.
        logits_layer: `Dense` layer, used for calculating logits.
        models: list, instances of discriminator `Model`s for each growth.
    """
    def __init__(
        self,
        kernel_regularizer,
        bias_regularizer,
        name,
        params,
        alpha_var,
        num_growths
    ):
        """Instantiates and builds discriminator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of discriminator.
            params: dict, user passed parameters.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            num_growths: int, number of growth phases for model.
        """
        # Set name of discriminator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Store reference to alpha variable.
        self.alpha_var = alpha_var

        # Store lists of layers.
        self.input_layers = []

        self.from_rgb_conv_layers = []
        self.from_rgb_leaky_relu_layers = []

        self.conv_layers = []
        self.leaky_relu_layers = []

        self.growing_downsample_layers = []
        self.shrinking_downsample_layers = []

        self.minibatch_stddev_layer = None

        self.flatten_layer = None
        self.logits_layer = None

        # Instantiate discriminator layers.
        self._create_discriminator_layers()
        
        # Store list of discriminator models.
        self.models = self._create_models(num_growths)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _create_input_layers(self):
        """Creates discriminator input layers for each image resolution.

        Returns:
            List of `Input` layers.
        """
        height, width = self.params["generator_projection_dims"][0:2]

        # Create list to hold `Input` layers.
        input_layers = [
            tf.keras.Input(
                shape=(height * 2 ** i, width * 2 ** i, self.params["depth"]),
                name="{}_{}x{}_inputs".format(
                    self.name, height * 2 ** i, width * 2 ** i
                )
            )
            for i in range(len(self.params["discriminator_from_rgb_layers"]))
        ]

        return input_layers

    def _create_from_rgb_layers(self):
        """Creates discriminator fromRGB layers of 1x1 convs.

        Returns:
            List of fromRGB 1x1 conv layers and leaky relu layers.
        """
        # Get fromRGB layer properties.
        from_rgb = [
            self.params["discriminator_from_rgb_layers"][i][0][:]
            for i in range(
                len(self.params["discriminator_from_rgb_layers"])
            )
        ]

        # Create list to hold toRGB 1x1 convs.
        from_rgb_conv_layers = [
            custom_layers.WeightScaledConv2D(
                filters=from_rgb[i][3],
                kernel_size=from_rgb[i][0:2],
                strides=from_rgb[i][4:6],
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

        from_rgb_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params["discriminator_leaky_relu_alpha"],
                name="{}_from_rgb_layers_leaky_relu_{}".format(self.name, i)
            )
            for i in range(len(from_rgb))
        ]

        return from_rgb_conv_layers, from_rgb_leaky_relu_layers

    def _create_base_conv_layer_block(self):
        """Creates discriminator base conv layer block.

        Returns:
            List of base block conv layers and list of leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = self.params["discriminator_base_conv_blocks"][0]

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
            for i in range(len(conv_block) - 1)
        ]

        # Have valid padding for layer just before flatten and logits.
        base_conv_layers.append(
            custom_layers.WeightScaledConv2D(
                filters=conv_block[-1][3],
                kernel_size=conv_block[-1][0:2],
                strides=conv_block[-1][4:6],
                padding="valid",
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
                    len(conv_block) - 1,
                    conv_block[-1][0],
                    conv_block[-1][1],
                    conv_block[-1][2],
                    conv_block[-1][3]
                )
            )
        )

        base_leaky_relu_layers = [
            tf.keras.layers.LeakyReLU(
                alpha=self.params["discriminator_leaky_relu_alpha"],
                name="{}_base_layers_leaky_relu_{}".format(self.name, i)
            )
            for i in range(len(conv_block))
        ]

        return base_conv_layers, base_leaky_relu_layers

    def _create_growth_conv_layer_block(self, block_idx):
        """Creates discriminator growth conv layer block.

        Args:
            block_idx: int, the current growth block's index.

        Returns:
            List of growth block's conv layers and list of growth block's
                leaky relu layers.
        """
        # Get conv block layer properties.
        conv_block = (
            self.params["discriminator_growth_conv_blocks"][block_idx]
        )

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
                alpha=self.params["discriminator_leaky_relu_alpha"],
                name="{}_growth_layers_leaky_relu_{}_{}".format(
                    self.name, block_idx, i
                )
            )
            for i in range(len(conv_block))
        ]

        return growth_conv_layers, growth_leaky_relu_layers

    def _create_downsample_layers(self):
        """Creates discriminator downsample layers.

        Returns:
            Lists of AveragePooling2D layers for growing and shrinking
                branches.
        """
        # Create list to hold growing branch's downsampling layers.
        growing_downsample_layers = [
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_growing_average_pooling_2d_{}".format(
                    self.name, i - 1
                )
            )
            for i in range(
                1, len(self.params["discriminator_from_rgb_layers"])
            )
        ]

        # Create list to hold shrinking branch's downsampling layers.
        shrinking_downsample_layers = [
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name="{}_shrinking_average_pooling_2d_{}".format(
                    self.name, i - 1
                )
            )
            for i in range(
                1, len(self.params["discriminator_from_rgb_layers"])
            )
        ]

        return growing_downsample_layers, shrinking_downsample_layers

    def _create_discriminator_layers(self):
        """Creates discriminator layers.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, latent_size].
        """
        # Create input layers for each image resolution.
        self.input_layers = self._create_input_layers()

        (self.from_rgb_conv_layers,
         self.from_rgb_leaky_relu_layers) = self._create_from_rgb_layers()

        (base_conv_layers,
         base_leaky_relu_layers) = self._create_base_conv_layer_block()
        self.conv_layers.append(base_conv_layers)
        self.leaky_relu_layers.append(base_leaky_relu_layers)

        for block_idx in range(
            len(self.params["discriminator_growth_conv_blocks"])
        ):
            (growth_conv_layers,
             growth_leaky_relu_layers
             ) = self._create_growth_conv_layer_block(block_idx)

            self.conv_layers.append(growth_conv_layers)
            self.leaky_relu_layers.append(growth_leaky_relu_layers)

        (self.growing_downsample_layers,
         self.shrinking_downsample_layers) = self._create_downsample_layers()

        self.minibatch_stddev_layer = custom_layers.MiniBatchStdDev(
            params={
                "use_minibatch_stddev": self.params["discriminator_use_minibatch_stddev"],
                "group_size": self.params["discriminator_minibatch_stddev_group_size"],
                "use_averaging": self.params["discriminator_minibatch_stddev_use_averaging"]
            }
        )

        self.flatten_layer = tf.keras.layers.Flatten()

        self.logits_layer = custom_layers.WeightScaledDense(
            units=1,
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
            name="{}_layers_dense_logits".format(self.name)
        )

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _use_logits_layer(self, inputs):
        """Uses flatten and logits layers to get logits tensor.

        Args:
            inputs: tensor, output of last conv layer of discriminator.

        Returns:
            Final logits tensor of discriminator.
        """
        # Set shape to remove ambiguity for dense layer.
        height, width =  self.params["generator_projection_dims"][0:2]
        valid_kernel_size = (
            self.params["discriminator_base_conv_blocks"][0][-1][0]
        )
        inputs.set_shape(
            [
                inputs.get_shape()[0],
                height - valid_kernel_size + 1,
                width - valid_kernel_size + 1,
                inputs.get_shape()[-1]]
        )

        # Flatten final block conv tensor.
        flat_inputs = self.flatten_layer(inputs=inputs)

        # Final linear layer for logits.
        logits = self.logits_layer(inputs=flat_inputs)

        return logits

    def _create_base_block_and_logits(self, inputs):
        """Creates base discriminator block and logits.

        Args:
            block_conv: tensor, output of previous `Conv2D` block's layer.

        Returns:
            Final logits tensor of discriminator.
        """
        # Only need the first conv layer block for base network.
        base_conv_layers = self.conv_layers[0]
        base_leaky_relu_layers = self.leaky_relu_layers[0]

        network = self.minibatch_stddev_layer(inputs=inputs)
        for i in range(len(base_conv_layers)):
            network = base_conv_layers[i](inputs=network)
            network = base_leaky_relu_layers[i](inputs=network)

        # Get logits now.
        logits = self._use_logits_layer(inputs=network)

        return logits

    def _create_growth_transition_weighted_sum(self, inputs, block_idx):
        """Creates growth transition img_to_vec weighted_sum.

        Args:
            inputs: tensor, input image to discriminator.
            block_idx: int, current block index of model progression.

        Returns:
            Tensor of weighted sum between shrinking and growing block paths.
        """
        # Growing side chain.
        growing_from_rgb_conv_layer = self.from_rgb_conv_layers[block_idx]
        growing_from_rgb_leaky_relu_layer = (
            self.from_rgb_leaky_relu_layers[block_idx]
        )
        growing_downsample_layer = (
            self.growing_downsample_layers[block_idx - 1]
        )

        growing_conv_layers = self.conv_layers[block_idx]
        growing_leaky_relu_layers = self.leaky_relu_layers[block_idx]

        # Pass inputs through layer chain.
        network = growing_from_rgb_conv_layer(inputs=inputs)
        network = growing_from_rgb_leaky_relu_layer(inputs=network)

        for i in range(len(growing_conv_layers)):
            network = growing_conv_layers[i](inputs=network)
            network = growing_leaky_relu_layers[i](inputs=network)

        # Down sample from 2s X 2s to s X s image.
        growing_network = growing_downsample_layer(inputs=network)

        # Shrinking side chain.
        shrinking_from_rgb_conv_layer = (
            self.from_rgb_conv_layers[block_idx - 1]
        )
        shrinking_from_rgb_leaky_relu_layer = (
            self.from_rgb_leaky_relu_layers[block_idx - 1]
        )
        shrinking_downsample_layer = (
            self.shrinking_downsample_layers[block_idx - 1]
        )

        # Pass inputs through layer chain.
        # Down sample from 2s X 2s to s X s image.
        network = shrinking_downsample_layer(inputs=inputs)

        network = shrinking_from_rgb_conv_layer(inputs=network)
        shrinking_network = shrinking_from_rgb_leaky_relu_layer(
            inputs=network
        )

        # Weighted sum.
        weighted_sum = tf.add(
            x=growing_network * self.alpha_var,
            y=shrinking_network * (1.0 - self.alpha_var),
            name="{}_growth_transition_weighted_sum_{}".format(
                self.name, block_idx
            )
        )

        return weighted_sum

    def _create_perm_growth_block_network(self, inputs, block_idx):
        """Creates discriminator permanent block network.

        Args:
            inputs: tensor, output of previous block's layer.
            block_idx: int, current block index of model progression.

        Returns:
            Tensor from final permanent block `Conv2D` layer.
        """
        # Get permanent growth blocks, so skip the base block.
        permanent_conv_layers = self.conv_layers[1:block_idx]
        permanent_leaky_relu_layers = self.leaky_relu_layers[1:block_idx]
        permanent_downsample_layers = self.growing_downsample_layers[0:block_idx - 1]

        # Reverse order of blocks.
        permanent_conv_layers = permanent_conv_layers[::-1]
        permanent_leaky_relu_layers = permanent_leaky_relu_layers[::-1]
        permanent_downsample_layers = permanent_downsample_layers[::-1]

        # Pass inputs through layer chain.
        network = inputs

        # Loop through the permanent growth blocks.
        for i in range(len(permanent_conv_layers)):
            # Get layers from ith permanent block.
            conv_layers = permanent_conv_layers[i]
            leaky_relu_layers = permanent_leaky_relu_layers[i]
            permanent_downsample_layer = permanent_downsample_layers[i]

            # Loop through layers of ith permanent block.
            for j in range(len(conv_layers)):
                network = conv_layers[j](inputs=network)
                network = leaky_relu_layers[j](inputs=network)

            # Down sample from 2s X 2s to s X s image.
            network = permanent_downsample_layer(inputs=network)

        return network

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _build_base_model(self, input_shape):
        """Builds discriminator base model.

        Args:
            input_shape: tuple, shape of image vector input of shape
                [batch_size, height, width, depth].

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to discriminator.
        # shape = (batch_size, height, width, depth)
        inputs = self.input_layers[0]

        # Only need the first fromRGB conv layer & block for base network.
        base_from_rgb_conv_layer = self.from_rgb_conv_layers[0]
        base_from_rgb_leaky_relu_layer = self.from_rgb_leaky_relu_layers[0]

        base_conv_layers = self.conv_layers[0]
        base_leaky_relu_layers = self.leaky_relu_layers[0]

        # Pass inputs through layer chain.
        network = base_from_rgb_conv_layer(inputs=inputs)
        network = base_from_rgb_leaky_relu_layer(inputs=network)

        # Get logits after continuing through base conv block.
        logits = self._create_base_block_and_logits(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=logits,
            name="{}_base".format(self.name)
        )

        return model

    def _build_growth_transition_model(
        self, input_shape, block_idx
    ):
        """Builds discriminator growth transition model.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, height, width, depth].
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to discriminator.
        # shape = (batch_size, height, width, depth)
        inputs = self.input_layers[block_idx]

        # Get weighted sum between shrinking and growing block paths.
        weighted_sum = self._create_growth_transition_weighted_sum(
            inputs=inputs, block_idx=block_idx
        )

        # Get output of final permanent growth block's last `Conv2D` layer.
        network = self._create_perm_growth_block_network(
            inputs=weighted_sum, block_idx=block_idx
        )

        # Get logits after continuing through base conv block.
        logits = self._create_base_block_and_logits(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=logits,
            name="{}_growth_transition_{}".format(self.name, block_idx)
        )

        return model

    def _build_growth_stable_model(self, input_shape, block_idx):
        """Builds generator growth stable model.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, latent_size].
            block_idx: int, current block index of model progression.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to discriminator.
        # shape = (batch_size, latent_size)
        inputs = self.input_layers[block_idx]

        # Get fromRGB layers.
        from_rgb_conv_layer = self.from_rgb_conv_layers[block_idx]
        from_rgb_leaky_relu_layer = self.from_rgb_leaky_relu_layers[block_idx]

        # Pass inputs through layer chain.
        network = from_rgb_conv_layer(inputs=inputs)
        network = from_rgb_leaky_relu_layer(inputs=network)

        # Get output of final permanent growth block's last `Conv2D` layer.
        network = self._create_perm_growth_block_network(
            inputs=network, block_idx=block_idx + 1
        )

        # Get logits after continuing through base conv block.
        logits = self._create_base_block_and_logits(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=logits,
            name="{}_growth_stable_{}".format(self.name, block_idx)
        )

        return model

    def _create_models(self, num_growths):
        """Creates list of discriminator's `Model` objects for each growth.

        Args:
            num_growths: int, number of growth phases for model.

        Returns:
            List of `Discriminator` `Model` objects.
        """
        models = []
        for growth_idx in range(num_growths):
            block_idx = (growth_idx + 1) // 2
            image_multiplier = 2 ** block_idx
            height = (
                self.params["generator_projection_dims"][0] * image_multiplier
            )
            width = (
                self.params["generator_projection_dims"][1] * image_multiplier
            )
            input_shape = (height, width, self.params["depth"])

            if growth_idx == 0:
                model = self._build_base_model(input_shape)
            elif growth_idx % 2 == 1:
                model = self._build_growth_transition_model(
                    input_shape=input_shape, block_idx=block_idx
                )
            elif growth_idx % 2 == 0:
                model = self._build_growth_stable_model(
                    input_shape=input_shape, block_idx=block_idx
                )

            models.append(model)

        return models

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def _get_gradient_penalty_loss(self, fake_images, real_images, growth_idx):
        """Gets discriminator gradient penalty loss.

        Args:
            fake_images: tensor, images generated by the generator from random
                noise of shape [batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [batch_size, image_height, image_width, 3].
            growth_idx: int, current growth index model has progressed to.

        Returns:
            Discriminator's gradient penalty loss of shape [].
        """
        batch_size = real_images.shape[0]

        # Get a random uniform number rank 4 tensor.
        random_uniform_num = tf.random.uniform(
            shape=[batch_size, 1, 1, 1],
            minval=0., maxval=1.,
            dtype=tf.float32,
            name="gp_random_uniform_num"
        )

        # Find the element-wise difference between images.
        image_difference = fake_images - real_images

        # Get random samples from this mixed image distribution.
        mixed_images = random_uniform_num * image_difference
        mixed_images += real_images

        # Get loss from interpolated mixed images and watch for gradients.
        with tf.GradientTape() as gp_tape:
            # Watch interpolated mixed images.
            gp_tape.watch(tensor=mixed_images)

            # Send to the discriminator to get logits.
            mixed_logits = self.models[growth_idx](
                inputs=mixed_images, training=True
            )

            # Get the mixed loss.
            mixed_loss = tf.reduce_sum(
                input_tensor=mixed_logits,
                name="gp_mixed_loss"
            )

        # Get gradient from returned list of length 1.
        mixed_gradients = gp_tape.gradient(
            target=mixed_loss, sources=[mixed_images]
        )[0]

        # Get gradient's L2 norm.
        mixed_norms = tf.sqrt(
            x=tf.reduce_sum(
                input_tensor=tf.square(
                    x=mixed_gradients,
                    name="gp_squared_grads"
                ),
                axis=[1, 2, 3]
            ) + 1e-8
        )

        # Get squared difference from target of 1.0.
        squared_difference = tf.square(
            x=mixed_norms - 1.0, name="gp_squared_difference"
        )

        # Get gradient penalty scalar.
        gradient_penalty = tf.reduce_mean(
            input_tensor=squared_difference, name="gp_gradient_penalty"
        )

        # Multiply with lambda to get gradient penalty loss.
        gradient_penalty_loss = tf.multiply(
            x=self.params["discriminator_gradient_penalty_coefficient"],
            y=gradient_penalty,
            name="gp_gradient_penalty_loss"
        )

        return gradient_penalty_loss

    def get_discriminator_loss(
        self,
        global_batch_size,
        fake_images,
        real_images,
        fake_logits,
        real_logits,
        global_step,
        summary_file_writer,
        growth_idx
    ):
        """Gets discriminator loss.

        Args:
            global_batch_size: int, global batch size for distribution.
            fake_images: tensor, images generated by the generator from random
                noise of shape [batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [batch_size, image_height, image_width, 3].
            fake_logits: tensor, output of discriminator using fake images
                with shape [batch_size, 1].
            real_logits: tensor, output of discriminator using real images
                with shape [batch_size, 1].
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.
            growth_idx: int, current growth index model has progressed to.

        Returns:
            Tensor of discriminator's total loss of shape [].
        """
        if self.params["distribution_strategy"]:
            # Calculate base discriminator loss.
            discriminator_fake_loss = tf.nn.compute_average_loss(
                per_example_loss=fake_logits,
                global_batch_size=global_batch_size
            )

            discriminator_real_loss = tf.nn.compute_average_loss(
                per_example_loss=real_logits,
                global_batch_size=global_batch_size
            )
        else:
            # Calculate base discriminator loss.
            discriminator_fake_loss = tf.reduce_mean(
                input_tensor=fake_logits,
                name="{}_fake_loss".format(self.name)
            )

            discriminator_real_loss = tf.reduce_mean(
                input_tensor=real_logits,
                name="{}_real_loss".format(self.name)
            )

        discriminator_loss = tf.subtract(
            x=discriminator_fake_loss,
            y=discriminator_real_loss,
            name="{}_loss".format(self.name)
        )

        # Get discriminator gradient penalty loss.
        discriminator_gradient_penalty = self._get_gradient_penalty_loss(
            fake_images, real_images, growth_idx
        )

        # Get discriminator epsilon drift penalty.
        epsilon_drift_penalty = tf.multiply(
            x=self.params["discriminator_epsilon_drift"],
            y=tf.reduce_mean(input_tensor=tf.square(x=real_logits)),
            name="epsilon_drift_penalty"
        )

        # Get discriminator Wasserstein GP loss.
        discriminator_wasserstein_gp_loss = tf.add_n(
            inputs=[
                discriminator_loss,
                discriminator_gradient_penalty,
                epsilon_drift_penalty
            ],
            name="{}_wasserstein_gp_loss".format(self.name)
        )

        if self.params["distribution_strategy"]:
            # Get regularization losses.
            discriminator_reg_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum(self.models[growth_idx].losses)
            )
        else:
            # Get regularization losses.
            discriminator_reg_loss = sum(self.models[growth_idx].losses)

        # Combine losses for total loss.
        discriminator_total_loss = tf.math.add(
            x=discriminator_wasserstein_gp_loss,
            y=discriminator_reg_loss,
            name="discriminator_total_loss"
        )

        if self.params["write_summaries"]:
            # Add summaries for TensorBoard.
            with summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=global_step,
                            y=self.params["save_summary_steps"]
                        ), y=0
                    )
                ):
                    tf.summary.scalar(
                        name="losses/discriminator_real_loss",
                        data=discriminator_real_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_fake_loss",
                        data=discriminator_fake_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_loss",
                        data=discriminator_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_gradient_penalty",
                        data=discriminator_gradient_penalty,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/epsilon_drift_penalty",
                        data=epsilon_drift_penalty,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_wasserstein_gp_loss",
                        data=discriminator_wasserstein_gp_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="losses/discriminator_reg_loss",
                        data=discriminator_reg_loss,
                        step=global_step
                    )
                    tf.summary.scalar(
                        name="optimized_losses/discriminator_total_loss",
                        data=discriminator_total_loss,
                        step=global_step
                    )
                    summary_file_writer.flush()

        return discriminator_total_loss
