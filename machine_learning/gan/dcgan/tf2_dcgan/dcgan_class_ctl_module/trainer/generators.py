import tensorflow as tf


class Generator(object):
    """Generator that takes latent vector input and outputs image.

    Fields:
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        params: dict, user passed parameters.
        model: instance of generator `Model`.
    """
    def __init__(
            self,
            input_shape,
            kernel_regularizer,
            bias_regularizer,
            name,
            params):
        """Instantiates and builds generator network.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, latent_size].
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of generator.
            params: dict, user passed parameters.
        """
        # Set name of generator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Instantiate generator `Model`.
        self.model = self._define_generator(input_shape)

    def _project_latent_vectors(self, latent_vectors):
        """Defines generator network.

        Args:
            latent_vectors: tensor, latent vector inputs of shape
                [batch_size, latent_size].

        Returns:
            Projected image of latent vector inputs.
        """
        projection_height = self.params["generator_projection_dims"][0]
        projection_width = self.params["generator_projection_dims"][1]
        projection_depth = self.params["generator_projection_dims"][2]

        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection = tf.keras.layers.Dense(
            units=projection_height * projection_width * projection_depth,
            activation=None,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="projection_dense_layer"
        )(inputs=latent_vectors)

        projection_leaky_relu = tf.keras.layers.LeakyReLU(
            alpha=self.params["generator_leaky_relu_alpha"],
            name="projection_leaky_relu"
        )(inputs=projection)

        # Add batch normalization to keep the inputs from blowing up.
        # shape = (
        #     batch_size,
        #     projection_height * projection_width * projection_depth
        # )
        projection_batch_norm = tf.keras.layers.BatchNormalization(
            name="projection_batch_norm"
        )(inputs=projection_leaky_relu)

        # Reshape projection into "image".
        # shape = (
        #     batch_size,
        #     projection_height,
        #     projection_width,
        #     projection_depth
        # )
        projected_image = tf.reshape(
            tensor=projection_batch_norm,
            shape=[
                -1, projection_height, projection_width, projection_depth
            ],
            name="projected_image"
        )

        return projected_image

    def _define_generator(
            self, input_shape):
        """Defines generator network.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, latent_size].

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to generator.
        # shape = (batch_size, latent_size)
        inputs = tf.keras.Input(
            shape=input_shape, name="{}_inputs".format(self.name)
        )

        # Dictionary containing possible final activations.
        final_activation_set = {"sigmoid", "relu", "tanh"}

        # Project latent vectors.
        network = self._project_latent_vectors(latent_vectors=inputs)

        # Iteratively build upsampling layers.
        for i in range(len(self.params["generator_num_filters"]) - 1):
            # Add conv transpose layers with given params per layer.
            # shape = (
            #     batch_size,
            #     generator_kernel_sizes[i - 1] * generator_strides[i],
            #     generator_kernel_sizes[i - 1] * generator_strides[i],
            #     generator_num_filters[i]
            # )
            network = tf.keras.layers.Conv2DTranspose(
                filters=self.params["generator_num_filters"][i],
                kernel_size=self.params["generator_kernel_sizes"][i],
                strides=self.params["generator_strides"][i],
                padding="same",
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_layers_conv2d_tranpose_{}".format(self.name, i)
            )(inputs=network)

            network = tf.keras.layers.LeakyReLU(
                alpha=self.params["generator_leaky_relu_alpha"],
                name="{}_leaky_relu_{}".format(self.name, i)
            )(inputs=network)

            # Add batch normalization to keep the inputs from blowing up.
            network = tf.keras.layers.BatchNormalization(
                name="{}_layers_batch_norm_{}".format(self.name, i)
            )(inputs=network)

        # Final conv2d transpose layer for image output.
        # shape = (batch_size, height, width, depth)
        fake_images = tf.keras.layers.Conv2DTranspose(
            filters=self.params["generator_num_filters"][-1],
            kernel_size=self.params["generator_kernel_sizes"][-1],
            strides=self.params["generator_strides"][-1],
            padding="same",
            activation=(
                self.params["generator_final_activation"].lower()
                if self.params["generator_final_activation"].lower()
                in final_activation_set
                else None
            ),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="{}_layers_conv2d_tranpose_fake_images".format(self.name)
        )(inputs=network)

        # Resize fake images to match real images in case of mismatch.
        height = self.params["height"]
        width = self.params["width"]
        fake_images = tf.keras.layers.Lambda(
            function=lambda x: tf.image.resize(
                images=x, size=[height, width], method="bilinear"
            ),
            name="{}_resize_fake_images".format(self.name)
        )(inputs=fake_images)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs, outputs=fake_images, name=self.name
        )

        return model

    def get_model(self):
        """Returns generator's `Model` object.

        Returns:
            Generator's `Model` object.
        """
        return self.model

    def get_generator_loss(
        self,
        global_batch_size,
        fake_logits,
        global_step,
        summary_file_writer
    ):
        """Gets generator loss.

        Args:
            global_batch_size: int, global batch size for distribution.
            fake_logits: tensor, shape of
                [batch_size, 1].
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.

        Returns:
            Tensor of generator's total loss of shape [].
        """
        if self.params["distribution_strategy"]:
            # Calculate base generator loss.
            generator_loss = tf.nn.compute_average_loss(
                per_example_loss=tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE
                )(
                    y_true=tf.ones_like(input=fake_logits), y_pred=fake_logits
                ),
                global_batch_size=global_batch_size
            )

            # Get regularization losses.
            generator_reg_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum(self.model.losses)
            )
        else:
            # Calculate base generator loss.
            generator_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )(
                y_true=tf.ones_like(input=fake_logits), y_pred=fake_logits
            )

            # Get regularization losses.
            generator_reg_loss = sum(self.model.losses)

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
                            x=global_step,
                            y=self.params["save_summary_steps"]
                        ), y=0
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
