import tensorflow as tf


class Generator(object):
    """Generator that takes latent vector input and outputs image.

    Fields:
        name: str, name of `Generator`.
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

        self.params = params

        # Instantiate generator `Model`.
        self.model = self._define_generator(
            input_shape, kernel_regularizer, bias_regularizer
        )

    def _define_generator(
            self, input_shape, kernel_regularizer, bias_regularizer):
        """Defines generator network.

        Args:
            input_shape: tuple, shape of latent vector input of shape
                [batch_size, latent_size].
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.

        Returns:
            Instance of `Model` object.
        """
        # Create the input layer to our DNN.
        # shape = (batch_size, latent_size)
        inputs = tf.keras.Input(
            shape=input_shape, name="{}_inputs".format(self.name)
        )
        network = inputs

        # Dictionary containing possible final activations.
        final_activation_set = {"sigmoid", "relu", "tanh"}

        # Add hidden layers with given number of units/neurons per layer.
        for i, units in enumerate(self.params["generator_hidden_units"]):
            # shape = (batch_size, generator_hidden_units[i])
            network = tf.keras.layers.Dense(
                units=units,
                activation=None,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="{}_layers_dense_{}".format(self.name, i)
            )(inputs=network)

            network = tf.keras.layers.LeakyReLU(
                alpha=self.params["generator_leaky_relu_alpha"],
                name="{}_leaky_relu_{}".format(self.name, i)
            )(inputs=network)

        # Final linear layer for outputs.
        # shape = (batch_size, height * width * depth)
        generated_outputs = tf.keras.layers.Dense(
            units=self.params["height"] * self.params["width"] * self.params["depth"],
            activation=(
                self.params["generator_final_activation"].lower()
                if self.params["generator_final_activation"].lower()
                in final_activation_set
                else None
            ),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="{}_layers_dense_generated_outputs".format(self.name)
        )(inputs=network)

        # Define model.
        model = tf.keras.Model(
            inputs=inputs, outputs=generated_outputs, name=self.name
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
