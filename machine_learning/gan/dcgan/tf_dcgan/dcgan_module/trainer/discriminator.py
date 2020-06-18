import tensorflow as tf

from .print_object import print_obj


class Discriminator(object):
    """Discriminator that takes image input and outputs logits.
    Fields:
        name: str, name of `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates and builds discriminator network.
        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of discriminator.
        """
        # Set name of discriminator.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

    def get_discriminator_logits(self, X, params):
        """Creates discriminator network and returns logits.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        func_name = "get_discriminator_logits"
        # Create the input layer to our CNN.
        # shape = (cur_batch_size, height * width * depth)
        network = X
        print_obj("\n" + func_name, "network", network)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            # Iteratively build downsampling layers.
            for i in range(len(params["discriminator_num_filters"])):
                # Add convolutional layers with given params per layer.
                # shape = (
                #     cur_batch_size,
                #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
                #     discriminator_kernel_sizes[i - 1] / discriminator_strides[i],
                #     discriminator_num_filters[i]
                # )
                network = tf.layers.conv2d(
                    inputs=network,
                    filters=params["discriminator_num_filters"][i],
                    kernel_size=params["discriminator_kernel_sizes"][i],
                    strides=params["discriminator_strides"][i],
                    padding="same",
                    activation=None,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_{}".format(i)
                )
                print_obj(func_name, "network", network)

                network = tf.nn.leaky_relu(
                    features=network,
                    alpha=params["discriminator_leaky_relu_alpha"],
                    name="leaky_relu_{}".format(i)
                )
                print_obj(func_name, "network", network)

                # Add some dropout for better regularization and stability.
                network = tf.layers.dropout(
                    inputs=network,
                    rate=params["discriminator_dropout_rates"][i],
                    name="layers_dropout_{}".format(i)
                )
                print_obj(func_name, "network", network)

            # Flatten network output.
            # shape = (
            #     cur_batch_size,
            #     (discriminator_kernel_sizes[-2] / discriminator_strides[-1]) ** 2 * discriminator_num_filters[-1]
            # )
            network_flat = tf.layers.Flatten()(inputs=network)
            print_obj(func_name, "network_flat", network_flat)

            # Final linear layer for logits.
            # shape = (cur_batch_size, 1)
            logits = tf.layers.dense(
                inputs=network_flat,
                units=1,
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="layers_dense_logits"
            )
            print_obj(func_name, "logits", logits)

        return logits

    def get_discriminator_loss(self, fake_logits, real_logits, params):
        """Gets discriminator loss.

        Args:
            fake_logits: tensor, shape of [cur_batch_size, 1].
            real_logits: tensor, shape of [cur_batch_size, 1].
            params: dict, user passed parameters.

        Returns:
            Tensor of discriminator's total loss of shape [].
        """
        func_name = "get_discriminator_loss"
        # Calculate base discriminator loss.
        discriminator_real_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_logits,
                labels=tf.multiply(
                    x=tf.ones_like(tensor=real_logits),
                    y=params["label_smoothing"]
                )
            ),
            name="discriminator_real_loss"
        )
        print_obj(
            "\n" + func_name,
            "discriminator_real_loss",
            discriminator_real_loss
        )

        discriminator_fake_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits,
                labels=tf.zeros_like(tensor=fake_logits)
            ),
            name="discriminator_fake_loss"
        )
        print_obj(
            func_name, "discriminator_fake_loss", discriminator_fake_loss
        )

        discriminator_loss = tf.add(
            x=discriminator_real_loss,
            y=discriminator_fake_loss,
            name="discriminator_loss"
        )
        print_obj(func_name, "discriminator_loss", discriminator_loss)

        # Get regularization losses.
        discriminator_reg_loss = tf.losses.get_regularization_loss(
            scope="discriminator",
            name="discriminator_reg_loss"
        )
        print_obj(func_name, "discriminator_reg_loss", discriminator_reg_loss)

        # Combine losses for total losses.
        discriminator_total_loss = tf.math.add(
            x=discriminator_loss,
            y=discriminator_reg_loss,
            name="discriminator_total_loss"
        )
        print_obj(
            func_name, "discriminator_total_loss", discriminator_total_loss
        )

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="discriminator_real_loss",
            tensor=discriminator_real_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="discriminator_fake_loss",
            tensor=discriminator_fake_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="discriminator_loss",
            tensor=discriminator_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="discriminator_reg_loss",
            tensor=discriminator_reg_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="discriminator_total_loss",
            tensor=discriminator_total_loss,
            family="total_losses"
        )

        return discriminator_total_loss
