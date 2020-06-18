import tensorflow as tf

from .print_object import print_obj


class Generator(object):
    """Generator that takes latent vector input and outputs image.
    Fields:
        name: str, name of `Generator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates and builds generator network.
        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of generator.
        """
        # Set name of generator.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

    def get_fake_images(self, Z, params):
        """Creates generator network and returns generated iamges.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            params: dict, user passed parameters.

        Returns:
            Generated image tensor of shape
                [cur_batch_size, height * width * depth].
        """
        func_name = "get_fake_images"
        # Create the input layer to our DNN.
        # shape = (cur_batch_size, latent_size)
        network = Z
        print_obj("\n" + func_name, "network", network)

        # Dictionary containing possible final activations.
        final_activation_dict = {
            "sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "tanh": tf.nn.tanh
        }

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            # Add hidden layers with given number of units/neurons per layer.
            for i, units in enumerate(params["generator_hidden_units"]):
                # shape = (cur_batch_size, generator_hidden_units[i])
                network = tf.layers.dense(
                    inputs=network,
                    units=units,
                    activation=None,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_dense_{}".format(i)
                )
                print_obj(func_name, "network", network)

                network = tf.nn.leaky_relu(
                    features=network,
                    alpha=params["generator_leaky_relu_alpha"],
                    name="leaky_relu_{}".format(i)
                )
                print_obj(func_name, "network", network)

            # Final linear layer for outputs.
            # shape = (cur_batch_size, height * width * depth)
            generated_outputs = tf.layers.dense(
                inputs=network,
                units=params["height"] * params["width"] * params["depth"],
                activation=final_activation_dict.get(
                    params["generator_final_activation"].lower(), None
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="layers_dense_generated_outputs"
            )
            print_obj(func_name, "generated_outputs", generated_outputs)

        return generated_outputs

    def get_generator_loss(self, fake_logits):
        """Gets generator loss.

        Args:
            fake_logits: tensor, shape of
                [cur_batch_size, height * width * depth].

        Returns:
            Tensor of generator's total loss of shape [].
        """
        func_name = "get_generator_loss"
        # Calculate base generator loss.
        generator_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits,
                labels=tf.ones_like(tensor=fake_logits)
            ),
            name="generator_loss"
        )
        print_obj("\n" + func_name, "generator_loss", generator_loss)

        # Get regularization losses.
        generator_reg_loss = tf.losses.get_regularization_loss(
            scope="generator",
            name="generator_regularization_loss"
        )
        print_obj(func_name, "generator_reg_loss", generator_reg_loss)

        # Combine losses for total losses.
        generator_total_loss = tf.math.add(
            x=generator_loss,
            y=generator_reg_loss,
            name="generator_total_loss"
        )
        print_obj(func_name, "generator_total_loss", generator_total_loss)

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="generator_loss", tensor=generator_loss, family="losses"
        )
        tf.summary.scalar(
            name="generator_reg_loss",
            tensor=generator_reg_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="generator_total_loss",
            tensor=generator_total_loss,
            family="total_losses"
        )

        return generator_total_loss
