import tensorflow as tf

from . import networks
from .print_object import print_obj


class Discriminator(networks.Networks):
    """Discriminator that takes image input and outputs logits.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates discriminator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of discriminator.
        """
        # Initialize base network class.
        super().__init__(kernel_regularizer, bias_regularizer, name)

    def get_discriminator_logits(self, target_image, params):
        """Creates discriminator network and returns logits.

        Args:
            target_image: tensor, target image tensor of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Logits tensor of shape [cur_batch_size, 16, 16, 1].
        """
        func_name = "get_discriminator_logits"
        print_obj("\n" + func_name, "target_image", target_image)

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Create input layer to discriminator network.
        network = target_image

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Iteratively build downsampling layers.
            for i in range(len(params["discriminator_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=self.name,
                    downscale=True,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Apply activation.
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=params["discriminator_activation"][i],
                    activation_set=activation_set,
                    params=params,
                    scope=self.name,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=self.name,
                    block_idx=0,
                    layer_idx=i
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
        func_name = "get_{}_loss".format(self.name)
        # Calculate base discriminator loss.
        if params["use_least_squares_loss"]:
            discriminator_real_loss = tf.losses.mean_squared_error(
                labels=tf.ones_like(tensor=real_logits),
                predictions=real_logits
            )
        else:
            discriminator_real_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=real_logits,
                    labels=tf.ones_like(tensor=real_logits)
                ),
                name="{}_real_loss".format(self.name)
            )
        print_obj(
            "\n" + func_name,
            "{}_real_loss".format(self.name),
            discriminator_real_loss
        )

        if params["use_least_squares_loss"]:
            discriminator_fake_loss = tf.losses.mean_squared_error(
                labels=tf.zeros_like(tensor=fake_logits),
                predictions=fake_logits
            )
        else:
            discriminator_fake_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_logits,
                    labels=tf.zeros_like(tensor=fake_logits)
                ),
                name="{}_fake_loss".format(self.name)
            )
        print_obj(
            func_name,
            "{}_fake_loss".format(self.name),
            discriminator_fake_loss
        )

        discriminator_loss = tf.add(
            x=discriminator_real_loss,
            y=discriminator_fake_loss,
            name="{}_loss".format(self.name)
        )
        print_obj(
            func_name, "{}_loss".format(self.name), discriminator_loss
        )

        # Divide discriminator loss by 2 so that it trains slower.
        discriminator_loss *= 0.5

        # Get regularization losses.
        discriminator_reg_loss = tf.losses.get_regularization_loss(
            scope=self.name,
            name="{}_reg_loss".format(self.name)
        )
        print_obj(
            func_name, "{}_reg_loss".format(self.name), discriminator_reg_loss
        )

        # Combine losses for total losses.
        discriminator_total_loss = tf.math.add(
            x=discriminator_loss,
            y=discriminator_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            func_name, "{}_total_loss".format(self.name), discriminator_total_loss
        )

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="{}_real_loss".format(self.name),
            tensor=discriminator_real_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_fake_loss".format(self.name),
            tensor=discriminator_fake_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_loss".format(self.name),
            tensor=discriminator_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_reg_loss".format(self.name),
            tensor=discriminator_reg_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_total_loss".format(self.name),
            tensor=discriminator_total_loss,
            family="losses"
        )

        return discriminator_total_loss
