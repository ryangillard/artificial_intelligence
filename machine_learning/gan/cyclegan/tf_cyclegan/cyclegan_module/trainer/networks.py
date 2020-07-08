import tensorflow as tf

from . import instance_normalization
from . import padding
from .print_object import print_obj


class Networks(object):
    """Network base class for generators and discriminators.

    Fields:
        name: str, name of network.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of network.
        """
        # Set name of network.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

    def pre_activation_network(
            self, network, params, scope, downscale, block_idx, layer_idx):
        """Creates pre-activation network graph.

        Args:
            network: tensor, rank 4 image tensor from previous block.
            params: dict, user passed parameters.
            scope: str, scope name of network.
            downscale: bool, whether using Conv2D or Conv2DTranspose layers.
            block_idx: int, current layer block index.
            layer_idx: int, current layer index within layer block.

        Returns:
            Final rank 4 image tensor of current pre-activation block.
        """
        func_name = "pre_activation_network_{}_{}".format(
            block_idx, layer_idx
        )
        print_obj("\n" + func_name, "network", network)

        scope_split = scope.split("/")

        if scope_split[0] == "generator":
            network_name = scope_split[0] + "_" + "_".join(scope_split[2:])
        else:
            network_name = scope_split[0] + "_".join(scope_split[2:])

        # Create kernel weight initializer.
        kernel_initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.02
        )

        filters = params["{}_num_filters".format(network_name)][layer_idx]
        kernel_size = params["{}_kernel_sizes".format(network_name)][layer_idx]
        strides = params["{}_strides".format(network_name)][layer_idx]

        if params["{}_downsample".format(network_name)][layer_idx]:
            # Create Conv2D layer with no padding.
            # shape = (
            #     cur_batch_size,
            #     kernel_sizes[i - 1] / strides[i],
            #     kernel_sizes[i - 1] / strides[i],
            #     num_filters[i]
            # )
            conv_outputs = tf.layers.conv2d(
                inputs=network,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                activation=None,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="layers_conv2d_{}_{}".format(block_idx, layer_idx)
            )

            # Now add padding.
            padding_type = params["{}_pad_type".format(network_name)].lower()
            if padding_type == "constant":
                network = padding.ConstantPadding2D(
                    conv_inputs=network,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    constant=params["{}_pad_constant".format(network_name)]
                )(inputs=conv_outputs)
            elif padding_type == "reflection":
                network = padding.ReflectionPadding2D(
                    conv_inputs=network,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same"
                )(inputs=conv_outputs)
            elif padding_type == "replication":
                network = padding.ReplicationPadding2D(
                    conv_inputs=network,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same"
                )(inputs=conv_outputs)
            else:
                network = padding.ConstantPadding2D(
                    conv_inputs=network,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    constant=0.0
                )(inputs=conv_outputs)
            print_obj(func_name, "network", network)
        else:
            # Create Conv2DTranspose layer with no padding.
            # shape = (
            #     cur_batch_size,
            #     kernel_sizes[i - 1] * strides[i],
            #     kernel_sizes[i - 1] * strides[i],
            #     num_filters[i]
            # )
            network = tf.layers.conv2d_transpose(
                inputs=network,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="layers_conv2d_transpose_{}_{}".format(
                    block_idx, layer_idx
                )
            )
        print_obj(func_name, "network", network)

        dropout_rates = params["{}_dropout_rates".format(network_name)]
        dropout_rate = dropout_rates[layer_idx]
        if dropout_rate:
            # Maybe add some dropout.
            if params["{}_dropout_before_act".format(network_name)]:
                if params["{}_dropout_before_norm".format(network_name)]:
                    network = tf.layers.dropout(
                        inputs=network,
                        rate=dropout_rate,
                        training=True,
                        name="layers_dropout_{}_{}".format(
                            block_idx, layer_idx
                        )
                    )
                    print_obj(func_name, "network", network)

        if params["{}_layer_norm_before_act".format(network_name)]:
            layer_norms = params["{}_layer_norm_type".format(network_name)]
            layer_norm = layer_norms[layer_idx]
            # Add layer normalization to keep inputs from blowing up.
            if layer_norm == "batch":
                network = tf.layers.batch_normalization(
                    inputs=network,
                    training=True,
                    name="layers_batch_norm_{}_{}".format(
                        block_idx, layer_idx
                    )
                )
                print_obj(func_name, "network", network)
            elif layer_norm == "instance":
                network = instance_normalization.InstanceNormalization(
                    axis=-1,
                    center=False,
                    scale=False,
                    name="layers_instance_norm_{}_{}".format(
                        block_idx, layer_idx
                    )
                )(inputs=network)
                print_obj(func_name, "network", network)

        if dropout_rate:
            # Maybe add some dropout.
            if params["{}_dropout_before_act".format(network_name)]:
                if not params["{}_dropout_before_norm".format(network_name)]:
                    network = tf.layers.dropout(
                        inputs=network,
                        rate=dropout_rate,
                        training=True,
                        name="layers_dropout_{}_{}".format(
                            block_idx, layer_idx
                        )
                    )
                    print_obj(func_name, "network", network)

        return network

    def apply_activation(
            self,
            input_tensor,
            activation_name,
            activation_set,
            params,
            scope,
            block_idx,
            layer_idx):
        """Applies activation to input tensor.

        Args:
            input_tensor: tensor, input to activation function.
            activation_name: str, name of activation function to apply.
            activation_set: set, allowable set of activation functions.
            params: dict, user passed parameters.
            scope: str, current scope of network.
            block_idx: int, current layer block index.
            layer_idx: int, current layer index within layer block.

        Returns:
            Activation tensor of same shape as input_tensor.
        """
        func_name = "apply_activation_{}_{}".format(block_idx, layer_idx)
        print_obj("\n" + func_name, "input_tensor", input_tensor)

        scope_split = scope.split("/")

        if scope_split[0] == "generator":
            network_name = scope_split[0] + "_" + "_".join(scope_split[2:])
        else:
            network_name = scope_split[0] + "_".join(scope_split[2:])

        # Lowercase activation name.
        activation_name = activation_name.lower()

        assert activation_name in activation_set
        if activation_name == "relu":
            activation = tf.nn.relu(
                features=input_tensor,
                name="relu_{}_{}".format(block_idx, layer_idx)
            )
        elif activation_name == "leaky_relu":
            activation = tf.nn.leaky_relu(
                features=input_tensor,
                alpha=params["{}_leaky_relu_alpha".format(network_name)],
                name="leaky_relu_{}_{}".format(block_idx, layer_idx)
            )
        elif activation_name == "tanh":
            activation = tf.math.tanh(
                x=input_tensor,
                name="tanh_{}_{}".format(block_idx, layer_idx)
            )
        else:
            activation = input_tensor
        print_obj(func_name, "activation", activation)

        return activation

    def post_activation_network(
            self, network, params, scope, block_idx, layer_idx):
        """Creates post-activation network graph.

        Args:
            network: tensor, rank 4 image tensor from previous block.
            params: dict, user passed parameters.
            scope: str, scope name of network.
            block_idx: int, current layer block index.
            layer_idx: int, current layer index within layer block.

        Returns:
            Final rank 4 image tensor of current post-activation block.
        """
        func_name = "post_activation_network_{}_{}".format(
            block_idx, layer_idx
        )
        print_obj("\n" + func_name, "network", network)

        scope_split = scope.split("/")

        if scope_split[0] == "generator":
            network_name = scope_split[0] + "_" + "_".join(scope_split[2:])
        else:
            network_name = scope_split[0] + "_".join(scope_split[2:])

        dropout_rates = params["{}_dropout_rates".format(network_name)]
        dropout_rate = dropout_rates[layer_idx]
        if dropout_rate:
            # Maybe add some dropout.
            if not params["{}_dropout_before_act".format(network_name)]:
                if params["{}_dropout_before_norm".format(network_name)]:
                    network = tf.layers.dropout(
                        inputs=network,
                        rate=dropout_rate,
                        training=True,
                        name="layers_dropout_{}_{}".format(
                            block_idx, layer_idx
                        )
                    )
                    print_obj(func_name, "network", network)

        if not params["{}_layer_norm_before_act".format(network_name)]:
            layer_norms = params["{}_layer_norm_type".format(network_name)]
            layer_norm = layer_norms[layer_idx]
            # Add layer normalization to keep inputs from blowing up.
            if layer_norm == "batch":
                network = tf.layers.batch_normalization(
                    inputs=network,
                    training=True,
                    name="layers_batch_norm_{}_{}".format(
                        block_idx, layer_idx
                    )
                )
                print_obj(func_name, "network", network)
            elif layer_norm == "instance":
                network = InstanceNormalization(
                    axis=-1,
                    center=False,
                    scale=False,
                    name="layers_instance_norm_{}_{}".format(
                        block_idx, layer_idx
                    )
                )(inputs=network)
                print_obj(func_name, "network", network)

        if dropout_rate:
            # Maybe add some dropout.
            if not params["{}_dropout_before_act".format(network_name)]:
                if not params["{}_dropout_before_norm".format(network_name)]:
                    network = tf.layers.dropout(
                        inputs=network,
                        rate=dropout_rate,
                        training=True,
                        name="layers_dropout_{}".format(
                            block_idx, layer_idx
                        )
                    )
                    print_obj(func_name, "network", network)

        return network
