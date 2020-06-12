import tensorflow as tf

from . import image_to_vector
from . import regularization
from .print_object import print_obj


class Encoder(image_to_vector.ImageToVector):
    """Encoder that takes image input and outputs logits.

    Fields:
        name: str, name of `Encoder`.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Instantiates and builds encoder network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of `Encoder`.
        """
        # Set name of encoder.
        self.name = name

        # Set kind of `ImageToVector`.
        kind = "encoder"

        # Initialize base class.
        super().__init__(kernel_regularizer, bias_regularizer, params, kind)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def instantiate_img_to_vec_logits_layer(self, params):
        """Instantiates encoder flatten and logits layers.

        Args:
            params: dict, user passed parameters.
        Returns:
            Flatten and logits layers of encoder.
        """
        func_name = "instantiate_img_to_vec_logits_layer"
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten layer to ready final block conv tensor for dense layer.
            flatten_layer = tf.layers.Flatten(
                name="{}_flatten_layer".format(self.name)
            )
            print_obj(func_name, "flatten_layer", flatten_layer)

            # Final linear layer for logits with same shape as latent vector.
            logits_layer = tf.layers.Dense(
                units=params["latent_size"],
                activation=None,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="{}_layers_dense_logits".format(self.name)
            )
            print_obj(func_name, "logits_layer", logits_layer)

        return flatten_layer, logits_layer

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def build_img_to_vec_base_conv_layer_block(self, params):
        """Creates encoder base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from base `Conv2D` layers.
        """
        func_name = "build_img_to_vec_base_conv_layer_block"
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["encoder_base_conv_blocks"][0]

            # The base conv block is always the 0th one.
            base_conv_layer_block = self.conv_layer_blocks[0]

            # Build base conv block layers, store in list.
            base_conv_tensors = [
                base_conv_layer_block[i](
                    inputs=tf.zeros(
                        shape=[1] + conv_block[i][0:3], dtype=tf.float32
                    )
                )
                for i in range(len(conv_block))
            ]
            print_obj(
                "\n" + func_name, "base_conv_tensors", base_conv_tensors
            )

        return base_conv_tensors

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def create_base_img_to_vec_network(self, X, params):
        """Creates base encoder network.

        Args:
            X: tensor, input image to encoder.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        func_name = "create_base_img_to_vec_network"
        print_obj("\n" + func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the first fromRGB conv layer for base network.
            from_rgb_conv_layer = self.from_rgb_conv_layers[0]

            # Pass inputs through layer chain.
            from_rgb_conv = from_rgb_conv_layer(inputs=X)
            print_obj(func_name, "from_rgb_conv", from_rgb_conv)

            from_rgb_conv = tf.nn.leaky_relu(
                features=from_rgb_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_from_rgb_conv_2d_leaky_relu".format(self.kind)
            )
            print_obj(func_name, "from_rgb_conv_leaky", from_rgb_conv)

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=from_rgb_conv, params=params
            )

        return logits

    def create_growth_transition_img_to_vec_network(
            self, X, alpha_var, params, trans_idx):
        """Creates growth transition encoder network.

        Args:
            X: tensor, input image to encoder.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Final logits tensor of encoder.
        """
        func_name = "create_growth_transition_{}_network".format(self.kind)

        print_obj("\nEntered {}".format(func_name), "trans_idx", trans_idx)
        print_obj(func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get weighted sum between shrinking and growing block paths.
            weighted_sum = self.create_growth_transition_img_to_vec_weighted_sum(
                X=X, alpha_var=alpha_var, trans_idx=trans_idx)
            print_obj(func_name, "weighted_sum", weighted_sum)

            # Get output of final permanent growth block's last `Conv2D` layer.
            block_conv = self.create_growth_transition_img_to_vec_perm_block_network(
                block_conv=weighted_sum, params=params, trans_idx=trans_idx)
            print_obj(func_name, "block_conv", block_conv)

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=block_conv, params=params
            )
            print_obj(func_name, "logits", logits)

        return logits

    def create_final_img_to_vec_network(self, X, params):
        """Creates final encoder network.

        Args:
            X: tensor, input image to encoder.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of encoder.
        """
        func_name = "create_final_img_to_vec_network"
        print_obj("\n" + func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the last fromRGB conv layer.
            from_rgb_conv_layer = self.from_rgb_conv_layers[-1]

            # Pass inputs through layer chain.
            block_conv = from_rgb_conv_layer(inputs=X)
            print_obj(func_name, "block_conv", block_conv)

            block_conv = tf.nn.leaky_relu(
                features=from_rgb_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_from_rgb_conv_2d_leaky_relu".format(self.kind)
            )
            print_obj(func_name, "block_conv_leaky", block_conv)

            # Get output of final permanent growth block's last `Conv2D` layer.
            block_conv = self.create_growth_transition_img_to_vec_perm_block_network(
                block_conv=block_conv,
                params=params,
                trans_idx=len(params["conv_num_filters"]) - 1
            )
            print_obj(func_name, "block_conv", block_conv)

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=block_conv, params=params
            )
            print_obj(func_name, "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_predict_encoder_logits(self, X, params, block_idx):
        """Uses encoder network and returns encoded logits for predict.

        Args:
            X: tensor, image tensors of shape
                [cur_batch_size, image_size, image_size, depth].
            params: dict, user passed parameters.
            block_idx: int, current conv layer block's index.

        Returns:
            Logits tensor of shape [cur_batch_size, latent_size].
        """
        func_name = "get_predict_encoder_logits"
        print_obj("\n" + func_name, "X", X)

        # Get encoder's logits tensor.
        if block_idx == 0:
            # 4x4
            logits = self.create_base_img_to_vec_network(X=X, params=params)
        elif block_idx < len(params["conv_num_filters"]) - 1:
            # 8x8 through 512x512
            logits = self.create_growth_transition_img_to_vec_network(
                X=X,
                alpha_var=tf.ones(shape=[], dtype=tf.float32),
                params=params,
                trans_idx=block_idx - 1
            )
        else:
            # 1024x1024
            logits = self.create_final_img_to_vec_network(X=X, params=params)

        print_obj("\n" + func_name, "logits", logits)

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_encoder_loss(self, fake_images, encoded_images, params):
        """Gets encoder loss.

        Args:
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            encoded_images: tensor, images generated by the generator from
                encoder's vector output of shape
                [cur_batch_size, image_size, image_size, 3].
            params: dict, user passed parameters.

        Returns:
            Encoder's total loss tensor of shape [].
        """
        func_name = "get_encoder_loss"
        # Get difference between fake images and encoder images.
        generator_encoder_image_diff = tf.subtract(
            x=fake_images,
            y=encoded_images,
            name="generator_encoder_image_diff"
        )
        print_obj(
            func_name,
            "generator_encoder_image_diff",
            generator_encoder_image_diff
        )

        # Get L1 norm of image difference.
        image_diff_l1_norm = tf.reduce_sum(
            input_tensor=tf.abs(x=generator_encoder_image_diff),
            axis=[1, 2, 3]
        )
        print_obj(func_name, "image_diff_l1_norm", image_diff_l1_norm)

        # Calculate base encoder loss.
        encoder_loss = tf.reduce_mean(
            input_tensor=image_diff_l1_norm,
            name="{}_loss".format(self.name)
        )
        print_obj(func_name, "encoder_loss", encoder_loss)

        # Get encoder regularization losses.
        encoder_reg_loss = regularization.get_regularization_loss(
            lambda1=params["encoder_l1_regularization_scale"],
            lambda2=params["encoder_l2_regularization_scale"],
            scope=self.name
        )
        print_obj(func_name, "encoder_reg_loss", encoder_reg_loss)

        # Combine losses for total losses.
        encoder_total_loss = tf.add(
            x=encoder_loss,
            y=encoder_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(func_name, "encoder_total_loss", encoder_total_loss)

        if not params["use_tpu"]:
            # Add summaries for TensorBoard.
            tf.summary.scalar(
                name="encoder_loss",
                tensor=encoder_loss,
                family="losses"
            )
            tf.summary.scalar(
                name="encoder_reg_loss",
                tensor=encoder_reg_loss,
                family="losses"
            )
            tf.summary.scalar(
                name="encoder_total_loss",
                tensor=encoder_reg_loss,
                family="total_losses"
            )

        return encoder_total_loss
