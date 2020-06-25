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

    def get_fake_images(self, Z, mode, params):
        """Creates generator network and returns generated images.

        Args:
            Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
            mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
                PREDICT.
            params: dict, user passed parameters.

        Returns:
            Generated image tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "get_fake_images"
        # Create the input layer to our CNN.
        # shape = (cur_batch_size, latent_size)
        network = Z
        print_obj("\n" + func_name, "network", network)

        # Dictionary containing possible final activations.
        final_activation_dict = {
            "sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "tanh": tf.nn.tanh
        }

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            # Project latent vectors.
            projection_height = params["generator_projection_dims"][0]
            projection_width = params["generator_projection_dims"][1]
            projection_depth = params["generator_projection_dims"][2]

            # shape = (
            #     cur_batch_size,
            #     projection_height * projection_width * projection_depth
            # )
            projection = tf.layers.dense(
                inputs=Z,
                units=projection_height * projection_width * projection_depth,
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="projection_dense_layer"
            )
            print_obj(func_name, "projection", projection)

            if params["generator_use_leaky_relu"]:
                projection_activation = tf.nn.leaky_relu(
                    features=projection,
                    alpha=params["generator_leaky_relu_alpha"],
                    name="projection_leaky_relu"
                )
            else:
                projection_activation = tf.nn.relu(
                    features=projection,
                    name="projection_relu"
                )
            print_obj(
                func_name, "projection_activation", projection_activation
            )

            # Add batch normalization to keep the inputs from blowing up.
            # shape = (
            #     cur_batch_size,
            #     projection_height * projection_width * projection_depth
            # )
            projection_batch_norm = tf.layers.batch_normalization(
                inputs=projection_activation,
                training=(mode == tf.estimator.ModeKeys.TRAIN),
                name="projection_batch_norm"
            )
            print_obj(
                func_name, "projection_batch_norm", projection_batch_norm
            )

            # Reshape projection into "image".
            # shape = (
            #     cur_batch_size,
            #     projection_height,
            #     projection_width,
            #     projection_depth
            # )
            network = tf.reshape(
                tensor=projection_batch_norm,
                shape=[
                    -1, projection_height, projection_width, projection_depth
                ],
                name="projection_reshaped"
            )
            print_obj(func_name, "network", network)

            # Iteratively build upsampling layers.
            for i in range(len(params["generator_num_filters"]) - 1):
                # Add conv transpose layers with given params per layer.
                # shape = (
                #     cur_batch_size,
                #     generator_kernel_sizes[i - 1] * generator_strides[i],
                #     generator_kernel_sizes[i - 1] * generator_strides[i],
                #     generator_num_filters[i]
                # )
                network = tf.layers.conv2d_transpose(
                    inputs=network,
                    filters=params["generator_num_filters"][i],
                    kernel_size=params["generator_kernel_sizes"][i],
                    strides=params["generator_strides"][i],
                    padding="same",
                    activation=None,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_tranpose_{}".format(i)
                )
                print_obj(func_name, "network", network)

                if params["generator_use_leaky_relu"]:
                    network = tf.nn.leaky_relu(
                        features=network,
                        alpha=params["generator_leaky_relu_alpha"],
                        name="leaky_relu_{}".format(i)
                    )
                else:
                    network = tf.nn.relu(
                        features=network,
                        name="relu_{}".format(i)
                    )
                print_obj(func_name, "network", network)

                # Add batch normalization to keep the inputs from blowing up.
                if params["generator_use_batch_norm"][i]:
                    network = tf.layers.batch_normalization(
                        inputs=network,
                        training=(mode == tf.estimator.ModeKeys.TRAIN),
                        name="layers_batch_norm_{}".format(i)
                    )
                    print_obj(func_name, "network", network)

            # Final conv2d transpose layer for image output.
            # shape = (cur_batch_size, height, width, depth)
            fake_images = tf.layers.conv2d_transpose(
                inputs=network,
                filters=params["generator_num_filters"][-1],
                kernel_size=params["generator_kernel_sizes"][-1],
                strides=params["generator_strides"][-1],
                padding="same",
                activation=final_activation_dict.get(
                    params["generator_final_activation"].lower(), None
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="layers_conv2d_tranpose_fake_images"
            )
            print_obj(func_name, "fake_images", fake_images)

        return fake_images

    def get_generator_loss(self, fake_logits, params):
        """Gets generator loss.

        Args:
            fake_logits: tensor, shape of
                [cur_batch_size, 1].
            params: dict, user passed parameters.

        Returns:
            Tensor of generator's total loss of shape [].
        """
        func_name = "get_generator_loss"
        # Calculate base generator loss.
        generator_loss = tf.losses.mean_squared_error(
            labels=tf.ones_like(tensor=fake_logits) * params["loss_c"],
            predictions=fake_logits
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
