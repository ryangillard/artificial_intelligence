import tensorflow as tf


class Discriminator(object):
    """Discriminator that takes image input and outputs logits.

    Fields:
        name: str, name of `Discriminator`.
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

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

        # Instantiate discriminator `Model`.
        self.model = self._define_discriminator(
            input_shape, kernel_regularizer, bias_regularizer, params
        )

    def _define_discriminator(
            self, input_shape, kernel_regularizer, bias_regularizer, params):
        """Defines discriminator network.

        Args:
            input_shape: tuple, shape of image vector input of shape
                [batch_size, height * width * depth].
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to our DNN.
        # shape = (batch_size, height * width * depth)
        inputs = tf.keras.Input(
            shape=input_shape,
            name="{}_inputs".format(self.name)
        )
        network = inputs

        # Add hidden layers with given number of units/neurons per layer.
        for i, units in enumerate(params["discriminator_hidden_units"]):
            # shape = (batch_size, discriminator_hidden_units[i])
            network = tf.keras.layers.Dense(
                units=units,
                activation=None,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="{}_layers_dense_{}".format(self.name, i)
            )(inputs=network)

            network = tf.keras.layers.LeakyReLU(
                alpha=params["discriminator_leaky_relu_alpha"],
                name="{}_leaky_relu_{}".format(self.name, i)
            )(inputs=network)

        # Final linear layer for logits.
        # shape = (batch_size, 1)
        logits = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
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
        params,
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
            params: dict, user passed parameters.
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.

        Returns:
            Tensor of discriminator's total loss of shape [].
        """
        # Calculate base discriminator loss.
        discriminator_real_loss = tf.nn.compute_average_loss(
            per_example_loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                label_smoothing=params["label_smoothing"],
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

        discriminator_loss = tf.add(
            x=discriminator_real_loss,
            y=discriminator_fake_loss,
            name="discriminator_loss"
        )

        # Get regularization losses.
        discriminator_reg_loss = tf.nn.scale_regularization_loss(
            regularization_loss=sum(self.model.losses)
        )

        # Combine losses for total losses.
        discriminator_total_loss = tf.math.add(
            x=discriminator_loss,
            y=discriminator_reg_loss,
            name="discriminator_total_loss"
        )

        # Add summaries for TensorBoard.
        with summary_file_writer.as_default():
            with tf.summary.record_if(
                global_step % params["save_summary_steps"] == 0
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
