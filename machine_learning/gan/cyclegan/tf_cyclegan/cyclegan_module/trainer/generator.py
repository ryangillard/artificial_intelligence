import tensorflow as tf

from . import image_utils
from . import networks
from .print_object import print_obj


class Generator(networks.Networks):
    """Generator that takes source image input and outputs target image.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of generator.
        """
        # Initialize base network class.
        super().__init__(kernel_regularizer, bias_regularizer, name)

    def unet_encoder(self, source_images, params):
        """Creates generator's U-net encoder network.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Bottlenecked image tensor of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]] and
                reversed list of encoder activation tensors for use in U-net
                decoder.
        """
        func_name = "unet_encoder"
        print_obj("\n" + func_name, "source_images", source_images)

        # Create list for encoder activations if using optional U-net decoder.
        encoder_activations = []

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Define scope.
        scope = "{}/unet/encoder".format(self.name)

        # Create input layer to encoder.
        network = source_images

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Iteratively build downsampling layers.
            for i in range(len(params["generator_unet_encoder_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    downscale=True,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Apply activation.
                activation = params["generator_unet_encoder_activation"][i]
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=activation,
                    activation_set=activation_set,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Add encoder activations to list if using U-net decoder.
                encoder_activations.append(network)

            # If encoder activation list is not empty.
            if encoder_activations:
                # Drop final layer since it is the bottleneck.
                encoder_activations = encoder_activations[:-1]

                # Reverse order to match decoder image sizes.
                encoder_activations = encoder_activations[::-1]

                # Add None to end of list so concatenation in decoder doesn't
                # occur for last layer.
                encoder_activations += [None]

        return network, encoder_activations

    def unet_decoder(self, bottleneck, encoder_activations, params):
        """Creates generator's U-net decoder network.

        Args:
            bottleneck: tensor, bottleneck of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]].
            encoder_activations: list, reversed list of encoder activation
                tensors for use U-net decoder.
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "unet_decoder"
        print_obj("\n" + func_name, "bottleneck", bottleneck)
        print_obj(func_name, "encoder_activations", encoder_activations)

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Define scope.
        scope = "{}/unet/decoder".format(self.name)

        # Create input layer to decoder.
        network = bottleneck

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Iteratively build upsampling layers.
            for i in range(len(params["generator_unet_decoder_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    downscale=False,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Concatenate network layer with ith encoder activations along
                # channels.
                if tf.is_tensor(x=encoder_activations[i]):
                    network = tf.concat(
                        values=[network, encoder_activations[i]],
                        axis=-1,
                        name="unet_concat_{}".format(i)
                    )
                    print_obj(func_name, "network", network)

                # Apply activation.
                activation = params["generator_unet_decoder_activation"][i]
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=activation,
                    activation_set=activation_set,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

        return network

    def unet_generator(self, source_images, params):
        """Creates U-net generator network and returns generated images.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "unet_generator"
        print_obj("\n" + func_name, "source_images", source_images)

        # Encode image into bottleneck.
        bottleneck, encoder_activations = self.unet_encoder(
            source_images=source_images, params=params
        )
        print_obj("\n" + func_name, "bottleneck", bottleneck)
        print_obj(func_name, "encoder_activations", encoder_activations)

        # Decode bottleneck back into image.
        fake_target_images = self.unet_decoder(
            bottleneck=bottleneck,
            encoder_activations=encoder_activations,
            params=params
        )
        print_obj("\n" + func_name, "fake_target_images", fake_target_images)

        return fake_target_images

    def resnet_encoder(self, source_images, params):
        """Creates generator's resnet encoder network.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Bottlenecked image tensor of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]].
        """
        func_name = "resnet_encoder"
        print_obj("\n" + func_name, "source_images", source_images)

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Define scope.
        scope = "{}/resnet/enc".format(self.name)

        # Create input layer to encoder.
        network = source_images

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Iteratively build downsampling layers.
            for i in range(len(params["generator_resnet_enc_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    downscale=True,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Apply activation.
                activation = params["generator_resnet_enc_activation"][i]
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=activation,
                    activation_set=activation_set,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

        return network

    def resnet_res_block(self, inputs, params, block_idx):
        """Creates resnet residual block for resnet generator network.

        Args:
            inputs: tensor, input tensor to resnet residual block.
            params: dict, user passed parameters.
            block_idx: int, index of the current residual block.

        Returns:
            Output image tensor of resnet block.
        """
        func_name = "resnet_res_block_{}".format(block_idx)
        print_obj("\n" + func_name, "inputs", inputs)

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Define scope.
        scope = "{}/resnet/res".format(self.name)

        # Create input layer to residual block.
        network = inputs

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Iteratively build upsampling layers.
            for i in range(len(params["generator_resnet_res_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    downscale=True,
                    block_idx=block_idx,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Apply activation.
                activation = params["generator_resnet_res_activation"][i]
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=activation,
                    activation_set=activation_set,
                    params=params,
                    scope=scope,
                    block_idx=block_idx,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    block_idx=block_idx,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

            # Concatenate inputs with last Conv2D layer output along channels.
            network = tf.concat(values=[inputs, network], axis=-1)
            print_obj(func_name, "network", network)

        return network

    def resnet_decoder(self, bottleneck, params):
        """Creates generator's resnet decoder network.

        Args:
            bottleneck: tensor, bottleneck of shape
                [cur_batch_size, 1, 1, generator_encoder_num_filters[-1]].
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "resnet_decoder"
        print_obj("\n" + func_name, "bottleneck", bottleneck)

        # The set of allowed activations.
        activation_set = {"relu", "leaky_relu", "tanh", "none"}

        # Define scope.
        scope = "{}/resnet/dec".format(self.name)

        # Create input layer to decoder.
        network = bottleneck

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Iteratively build upsampling layers.
            for i in range(len(params["generator_resnet_dec_num_filters"])):
                # Create pre-activation network graph.
                network = self.pre_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    downscale=False,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Apply activation.
                activation = params["generator_resnet_dec_activation"][i]
                network = self.apply_activation(
                    input_tensor=network,
                    activation_name=activation,
                    activation_set=activation_set,
                    params=params,
                    scope="resnet_decoder",
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

                # Create post-activation network graph.
                network = self.post_activation_network(
                    network=network,
                    params=params,
                    scope=scope,
                    block_idx=0,
                    layer_idx=i
                )
                print_obj(func_name, "network", network)

        return network

    def resnet_generator(self, source_images, params):
        """Creates resnet generator network and returns generated images.

        Args:
            source_images: tensor, source images of shape
                [cur_batch_size, height, width, depth].
            params: dict, user passed parameters.

        Returns:
            Generated target images tensor of shape
                [cur_batch_size, height, width, depth].
        """
        func_name = "resnet_generator"
        print_obj("\n" + func_name, "source_images", source_images)

        # Encode image into bottleneck.
        bottleneck = self.resnet_encoder(
            source_images=source_images, params=params
        )
        print_obj("\n" + func_name, "bottleneck", bottleneck)

        # Pass bottleneck through residual blocks.
        for i in range(params["generator_num_resnet_blocks"]):
            bottleneck = self.resnet_res_block(
                inputs=bottleneck, params=params, block_idx=i
            )
            print_obj("\n" + func_name, "bottleneck", bottleneck)

        # Decode bottleneck back into image.
        fake_target_images = self.resnet_decoder(
            bottleneck=bottleneck, params=params
        )
        print_obj("\n" + func_name, "fake_target_images", fake_target_images)

        return fake_target_images

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

        if params["generator_use_unet"]:
            fake_target_images = self.unet_generator(
                source_images=source_images, params=params
            )
        else:
            fake_target_images = self.resnet_generator(
                source_images=source_images, params=params
            )

        # Resize fake target images to match real target image sizes.
        fake_target_images = image_utils.resize_fake_images(
            fake_images=fake_target_images, params=params
        )
        print_obj("\n" + func_name, "fake_target_images", fake_target_images)

        return fake_target_images

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
        if params["use_least_squares_loss"]:
            generator_loss = tf.losses.mean_squared_error(
                labels=tf.ones_like(tensor=fake_logits),
                predictions=fake_logits
            )
        else:
            generator_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_logits,
                    labels=tf.ones_like(tensor=fake_logits)
                ),
                name="{}_loss".format(self.name)
            )
        print_obj(
            "\n" + func_name, "{}_loss".format(self.name), generator_loss
        )

        # Get regularization losses.
        generator_reg_loss = tf.losses.get_regularization_loss(
            scope=self.name,
            name="{}_regularization_loss".format(self.name)
        )
        print_obj(
            func_name, "{}_reg_loss".format(self.name), generator_reg_loss
        )

        # Combine losses for total losses.
        generator_total_loss = tf.math.add(
            x=generator_loss,
            y=generator_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            func_name, "{}_total_loss".format(self.name), generator_total_loss
        )

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="{}_loss".format(self.name),
            tensor=generator_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_reg_loss".format(self.name),
            tensor=generator_reg_loss,
            family="losses"
        )
        tf.summary.scalar(
            name="{}_total_loss".format(self.name),
            tensor=generator_total_loss,
            family="losses"
        )

        return generator_total_loss
