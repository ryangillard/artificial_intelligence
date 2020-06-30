import tensorflow as tf

from .print_object import print_obj


class Generator(object):
    """Generator that takes source image input and outputs target image.
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

    def apply_activation(
            self,
            input_tensor,
            activation_name,
            activation_set,
            params,
            scope,
            layer_idx):
        """Applies activation to input tensor.

        Args:
            input_tensor: tensor, input to activation function.
            activation_name: str, name of activation function to apply.
            activation_set: set, allowable set of activation functions.
            params: dict, user passed parameters.
            scope: str, current scope of generator network.
            layer_idx: int, index of current layer of network.

        Returns:
            Activation tensor of same shape as input_tensor.
        """
        func_name = "apply_activation"
        print_obj("\n" + func_name, "input_tensor", input_tensor)

        assert activation_name in activation_set
        if activation_name == "relu":
            activation = tf.nn.relu(
                features=input_tensor,
                name="{}_relu_{}".format(scope, layer_idx)
            )
        elif activation_name == "leaky_relu":
            activation = tf.nn.leaky_relu(
                features=input_tensor,
                alpha=params["generator_{}_leaky_relu_alpha".format(scope)],
                name="{}_leaky_relu_{}".format(scope, layer_idx)
            )
        elif activation_name == "tanh":
            activation = tf.math.tanh(
                x=input_tensor,
                name="{}_tanh_{}".format(scope, layer_idx)
            )
        else:
            activation = input_tensor
        print_obj(func_name, "activation", activation)

        return activation

    def encoder(self, source_images, params):
        """Creates generator's encoder network.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Bottlenecked image tensor of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]] and
                reversed list of encoder activation tensors for use in
                optional U-net decoder.
        """
        func_name = "encoder"
        print_obj("\n" + func_name, "source_images", source_images)

        # Create kernel weight initializer.
        kernel_initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.02
        )

        # Create list for encoder activations if using optional U-net decoder.
        encoder_activations = []

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh"}

        # Create input layer to encoder.
        network = source_images

        with tf.variable_scope("generator/encoder", reuse=tf.AUTO_REUSE):
            # Iteratively build upsampling layers.
            for i in range(len(params["generator_encoder_num_filters"])):
                # Add convolutional layers with given params per layer.
                # shape = (
                #     cur_batch_size,
                #     generator_encoder_kernel_sizes[i - 1] / generator_encoder_strides[i],
                #     generator_encoder_kernel_sizes[i - 1] / generator_encoder_strides[i],
                #     generator_encoder_num_filters[i]
                # )
                network = tf.layers.conv2d(
                    inputs=network,
                    filters=params["generator_encoder_num_filters"][i],
                    kernel_size=params["generator_encoder_kernel_sizes"][i],
                    strides=params["generator_encoder_strides"][i],
                    padding="same",
                    activation=None,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_tranpose_{}".format(i)
                )
                print_obj(func_name, "network", network)

                if params["generator_encoder_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if params["generator_encoder_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                # Apply activation.
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=params["generator_encoder_activation"][i],
                    activation_set=activation_set,
                    params=params,
                    scope="encoder",
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                if params["generator_encoder_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if not params["generator_encoder_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                # Add encoder activations to list if using U-net decoder.
                if params["generator_use_unet_decoder"]:
                    encoder_activations.append(network)

            # If encoder activation list is not empty
            if encoder_activations:
                # Drop final layer since it is the bottleneck.
                encoder_activations = encoder_activations[:-1]

                # Reverse order to match decoder image sizes.
                encoder_activations = encoder_activations[::-1]

                # Add None to end of list so concatenation in decoder doesn't
                # occur for last layer.
                encoder_activations += [None]

        return network, encoder_activations

    def decoder(self, bottleneck, encoder_activations, params):
        """Creates generator's decoder network.

        Args:
            bottleneck: tensor, bottleneck of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]].
            encoder_activations: list, reversed list of encoder activation
                tensors for use in optional U-net decoder.
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "decoder"
        print_obj("\n" + func_name, "bottleneck", bottleneck)
        print_obj(func_name, "encoder_activations", encoder_activations)

        # Create kernel weight initializer.
        kernel_initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.02
        )

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh"}

        # Create input layer to decoder.
        network = bottleneck

        with tf.variable_scope("generator/decoder", reuse=tf.AUTO_REUSE):
            # Iteratively build upsampling layers.
            for i in range(len(params["generator_decoder_num_filters"])):
                # Add conv transpose layers with given params per layer.
                # shape = (
                #     cur_batch_size,
                #     generator_decoder_kernel_sizes[i - 1] * generator_decoder_strides[i],
                #     generator_decoder_kernel_sizes[i - 1] * generator_decoder_strides[i],
                #     generator_decoder_num_filters[i]
                # )
                network = tf.layers.conv2d_transpose(
                    inputs=network,
                    filters=params["generator_decoder_num_filters"][i],
                    kernel_size=params["generator_decoder_kernel_sizes"][i],
                    strides=params["generator_decoder_strides"][i],
                    padding="same",
                    activation=None,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="layers_conv2d_tranpose_{}".format(i)
                )
                print_obj(func_name, "network", network)

                if params["generator_decoder_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if params["generator_decoder_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                if params["generator_decoder_dropout_rates"][i]:
                    # Add some dropout.
                    if params["generator_decoder_dropout_before_act"]:
                        network = tf.layers.dropout(
                            inputs=network,
                            rate=params["generator_decoder_dropout_rates"][i],
                            training=True,
                            name="layers_dropout_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                if params["generator_use_unet_decoder"]:
                    if tf.is_tensor(x=encoder_activations[i]):
                        network = tf.concat(
                            values=[network, encoder_activations[i]],
                            axis=-1,
                            name="unet_concat_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                # Apply activation.
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=params["generator_decoder_activation"][i],
                    activation_set=activation_set,
                    params=params,
                    scope="decoder",
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                if params["generator_decoder_dropout_rates"][i]:
                    # Add some dropout.
                    if not params["generator_decoder_dropout_before_act"]:
                        network = tf.layers.dropout(
                            inputs=network,
                            rate=params["generator_decoder_dropout_rates"][i],
                            training=True,
                            name="layers_dropout_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

                if params["generator_decoder_use_batch_norm"][i]:
                    # Add batch normalization to keep inputs from blowing up.
                    if not params["generator_decoder_batch_norm_before_act"]:
                        network = tf.layers.batch_normalization(
                            inputs=network,
                            training=True,
                            name="layers_batch_norm_{}".format(i)
                        )
                        print_obj(func_name, "network", network)

        return network

    def get_fake_images(self, source_images, params):
        """Creates generator network and returns generated images.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "get_fake_images"
        print_obj("\n" + func_name, "source_images", source_images)

        # Encode image into bottleneck.
        bottleneck, encoder_activations = self.encoder(source_images, params)
        print_obj("\n" + func_name, "bottleneck", bottleneck)
        print_obj(func_name, "bottleneck", bottleneck)

        # Decode bottleneck back into image.
        fake_target_images = self.decoder(
            bottleneck, encoder_activations, params
        )
        print_obj("\n" + func_name, "fake_target_images", fake_target_images)

        return fake_target_images

    def get_generator_loss(
            self,
            fake_target_images,
            real_target_images,
            fake_logits,
            params):
        """Gets generator loss.

        Args:
            fake_target_images: tensor, target images generated by the
                generator from source images of shape
                [cur_batch_size, image_size, image_size, depth].
            real_target_images: tensor, real target images from input of shape
                [cur_batch_size, image_size, image_size, depth].
            fake_logits: tensor, shape of
                [cur_batch_size, 1].
            params: dict, user passed parameters.

        Returns:
            Tensor of generator's total loss of shape [].
        """
        func_name = "get_generator_loss"

        # Calculate base generator loss.
        generator_fake_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits,
                labels=tf.ones_like(tensor=fake_logits)
            ),
            name="generator_loss"
        )
        print_obj(
            "\n" + func_name, "generator_fake_loss", generator_fake_loss
        )

        # Calculate L1 loss between fake and real target images.
        generator_l1_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(
                input_tensor=tf.abs(
                    x=tf.subtract(
                        x=fake_target_images, y=real_target_images
                    )
                ),
                axis=[1, 2, 3]
            ),
            name="generator_l1_loss"
        )
        print_obj(func_name, "generator_l1_loss", generator_l1_loss)

        # Combine base and weighted L1 loss together.
        generator_loss = tf.add(
            x=generator_fake_loss,
            y=generator_l1_loss * params["generator_l1_loss_weight"],
            name="generator_loss"
        )
        print_obj(func_name, "generator_loss", generator_loss)
        

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
            name="generator_fake_loss",
            tensor=generator_fake_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="generator_l1_loss",
            tensor=generator_l1_loss,
            family="losses"
        )
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
