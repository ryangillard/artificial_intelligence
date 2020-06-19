import tensorflow as tf

from .print_object import print_obj


class Critic(object):
    """Critic that takes image input and outputs logits.
    Fields:
        name: str, name of `Critic`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates and builds critic network.
        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of critic.
        """
        # Set name of critic.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

    def get_critic_logits(self, X, params):
        """Creates critic network and returns logits.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 1].
        """
        func_name = "get_critic_logits"
        # Create the input layer to our CNN.
        # shape = (cur_batch_size, height * width * depth)
        network = X
        print_obj("\n" + func_name, "network", network)

        with tf.variable_scope("critic", reuse=tf.AUTO_REUSE):
            # Iteratively build downsampling layers.
            for i in range(len(params["critic_num_filters"])):
                # Add convolutional layers with given params per layer.
                # shape = (
                #     cur_batch_size,
                #     critic_kernel_sizes[i - 1] / critic_strides[i],
                #     critic_kernel_sizes[i - 1] / critic_strides[i],
                #     critic_num_filters[i]
                # )
                network = tf.layers.conv2d(
                    inputs=network,
                    filters=params["critic_num_filters"][i],
                    kernel_size=params["critic_kernel_sizes"][i],
                    strides=params["critic_strides"][i],
                    padding="same",
                    activation=None,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_{}".format(i)
                )
                print_obj(func_name, "network", network)

                network = tf.nn.leaky_relu(
                    features=network,
                    alpha=params["critic_leaky_relu_alpha"],
                    name="leaky_relu_{}".format(i)
                )
                print_obj(func_name, "network", network)

                # Add some dropout for better regularization and stability.
                network = tf.layers.dropout(
                    inputs=network,
                    rate=params["critic_dropout_rates"][i],
                    name="layers_dropout_{}".format(i)
                )
                print_obj(func_name, "network", network)

            # Flatten network output.
            # shape = (
            #     cur_batch_size,
            #     (critic_kernel_sizes[-2] / critic_strides[-1]) ** 2 * critic_num_filters[-1]
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

    def get_critic_loss(self, fake_logits, real_logits):
        """Gets critic loss.

        Args:
            fake_logits: tensor, shape of [cur_batch_size, 1].
            real_logits: tensor, shape of [cur_batch_size, 1].

        Returns:
            Tensor of critic's total loss of shape [].
        """
        func_name = "get_critic_loss"
        # Calculate base critic loss.
        critic_real_loss = tf.reduce_mean(
            input_tensor=real_logits, name="critic_real_loss"
        )
        print_obj("\n" + func_name, "critic_real_loss", critic_real_loss)

        critic_fake_loss = tf.reduce_mean(
            input_tensor=fake_logits, name="critic_fake_loss"
        )
        print_obj(
            func_name, "critic_fake_loss", critic_fake_loss
        )

        critic_loss = tf.subtract(
            x=critic_fake_loss, y=critic_real_loss, name="critic_loss"
        )
        print_obj(func_name, "critic_loss", critic_loss)

        # Get regularization losses.
        critic_reg_loss = tf.losses.get_regularization_loss(
            scope="critic", name="critic_reg_loss"
        )
        print_obj(func_name, "critic_reg_loss", critic_reg_loss)

        # Combine losses for total losses.
        critic_total_loss = tf.math.add(
            x=critic_loss, y=critic_reg_loss, name="critic_total_loss"
        )
        print_obj(func_name, "critic_total_loss", critic_total_loss)

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="critic_real_loss", tensor=critic_real_loss, family="losses"
        )
        tf.summary.scalar(
            name="critic_fake_loss", tensor=critic_fake_loss, family="losses"
        )
        tf.summary.scalar(
            name="critic_loss", tensor=critic_loss, family="losses"
        )
        tf.summary.scalar(
            name="critic_reg_loss", tensor=critic_reg_loss, family="losses"
        )
        tf.summary.scalar(
            name="critic_total_loss",
            tensor=critic_total_loss,
            family="total_losses"
        )

        return critic_total_loss
