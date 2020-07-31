import tensorflow as tf


class Discriminator(object):
    """Discriminator that takes image input and outputs logits.

    Fields:
        name: str, name of `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for
            kernel variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
        params: dict, user passed parameters.
        model: instance of discriminator `Model`.
    """
    def __init__(
            self,
            input_shape,
            kernel_regularizer,
            bias_regularizer,
            name,
            params):
        """Instantiates and builds discriminator network.

        Args:
            input_shape: tuple, shape of image vector input of shape
                [batch_size, height * width * depth].
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of discriminator.
            params: dict, user passed parameters.
        """
        # Set name of discriminator.
        self.name = name

        # Store regularizers.
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Store parameters.
        self.params = params

        # Instantiate discriminator `Model`.
        self.model = self._define_discriminator(input_shape)

    def _define_discriminator(self, input_shape):
        """Defines discriminator network.

        Args:
            input_shape: tuple, shape of image vector input of shape
                [batch_size, height * width * depth].

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to discriminator.
        # shape = (batch_size, height * width * depth)
        inputs = tf.keras.Input(
            shape=input_shape,
            name="{}_inputs".format(self.name)
        )
        network = inputs

        # Iteratively build downsampling layers.
        for i in range(len(self.params["discriminator_num_filters"])):
            # Add convolutional layers with given params per layer.
            # shape = (
            #     batch_size,
            #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
            #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
            #     discriminator_num_filters[i]
            # )
            network = tf.keras.layers.Conv2D(
                filters=self.params["discriminator_num_filters"][i],
                kernel_size=self.params["discriminator_kernel_sizes"][i],
                strides=self.params["discriminator_strides"][i],
                padding="same",
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_layers_conv2d_{}".format(self.name, i)
            )(inputs=network)

            network = tf.keras.layers.LeakyReLU(
                alpha=self.params["discriminator_leaky_relu_alpha"],
                name="{}_leaky_relu_{}".format(self.name, i)
            )(inputs=network)

            # Add some dropout for better regularization and stability.
            network = tf.keras.layers.Dropout(
                rate=self.params["discriminator_dropout_rates"][i],
                name="{}_layers_dropout_{}".format(self.name, i)
            )(inputs=network)

        # Flatten network output.
        # shape = (
        #     batch_size,
        #     (discriminator_kernel_sizes[-2] / discriminator_strides[-1]) ** 2 * discriminator_num_filters[-1]
        # )
        network = tf.keras.layers.Flatten()(inputs=network)

        # Final linear layer for logits.
        # shape = (batch_size, 1)
        logits = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="{}_layers_dense_logits".format(self.name)
        )(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs, outputs=logits, name=self.name
        )

        return model

    def get_model(self):
        """Returns discriminator's `Model` object.

        Returns:
            Discriminator's `Model` object.
        """
        return self.model

    def get_discriminator_loss(
        self,
        global_batch_size,
        fake_logits,
        real_logits,
        global_step,
        summary_file_writer
    ):
        """Gets discriminator loss.

        Args:
            global_batch_size: int, global batch size for distribution.
            fake_logits: tensor, shape of
                [batch_size, 1].
            real_logits: tensor, shape of
                [batch_size, 1].
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.

        Returns:
            Tensor of discriminator's total loss of shape [].
        """
        if self.params["distribution_strategy"]:
            # Calculate base discriminator loss.
            discriminator_real_loss = tf.nn.compute_average_loss(
                per_example_loss=tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    label_smoothing=self.params["label_smoothing"],
                    reduction=tf.keras.losses.Reduction.NONE
                )(
                    y_true=tf.ones_like(input=real_logits), y_pred=real_logits
                ),
                global_batch_size=global_batch_size
            )

            discriminator_fake_loss = tf.nn.compute_average_loss(
                per_example_loss=tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE
                )(
                    y_true=tf.zeros_like(input=fake_logits), y_pred=fake_logits
                ),
                global_batch_size=global_batch_size
            )
        else:
            # Calculate base discriminator loss.
            discriminator_real_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                label_smoothing=self.params["label_smoothing"]
            )(
                y_true=tf.ones_like(input=real_logits), y_pred=real_logits
            )

            discriminator_fake_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            )(
                y_true=tf.zeros_like(input=fake_logits), y_pred=fake_logits
            )

        discriminator_loss = tf.math.add(
            x=discriminator_real_loss,
            y=discriminator_fake_loss,
            name="discriminator_loss"
        )

        if self.params["distribution_strategy"]:
            # Get regularization losses.
            discriminator_reg_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum(self.model.losses)
            )
        else:
            # Get regularization losses.
            discriminator_reg_loss = sum(self.model.losses)

        # Combine losses for total losses.
        discriminator_total_loss = tf.math.add(
            x=discriminator_loss,
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
