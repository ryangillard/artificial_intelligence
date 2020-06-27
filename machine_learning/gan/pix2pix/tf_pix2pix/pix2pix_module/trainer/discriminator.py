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

    def get_discriminator_logits(self, source_image, target_image, params):
        """Creates discriminator network and returns logits.

        Args:
            source_image: tensor, source image tensor of shape
                [cur_batch_size, height, width, depth].
            target_image: tensor, target image tensor of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 16, 16, 1].
        """
        func_name = "get_discriminator_logits"
        print_obj("\n" + func_name, "source_image", source_image)
        print_obj(func_name, "target_image", target_image)

        # Create kernel weight initializer.
        kernel_initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.02
        )

        # Concatenate source and target images along channel dimension.
        network = tf.concat(values=[source_image, target_image], axis=-1)

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
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_{}".format(i)
                )
                print_obj(func_name, "network", network)

                if params["discriminator_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if params["discriminator_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                if params["discriminator_use_leaky_relu"]:
                    network = tf.nn.leaky_relu(
                        features=network,
                        alpha=params["discriminator_leaky_relu_alpha"],
                        name="leaky_relu_{}".format(i)
                    )
                else:
                    network = tf.nn.relu(
                        features=network,
                        name="relu_{}".format(i)
                    )
                print_obj(func_name, "network", network)

                if params["discriminator_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if not params["discriminator_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

        return network

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
                labels=tf.ones_like(tensor=real_logits)
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

        # Divide discriminator loss by 2 so that it trains slower.
        discriminator_loss *= 0.5

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
