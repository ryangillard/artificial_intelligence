import tensorflow as tf

from . import equalized_learning_rate_layers
from . import image_to_vector
from . import regularization
from .print_object import print_obj


class Discriminator(image_to_vector.ImageToVector):
    """Discriminator that takes image input and outputs logits.

    Fields:
        name: str, name of `Discriminator`.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Instantiates and builds discriminator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of `Discriminator`.
        """
        # Set name of discriminator.
        self.name = name

        # Set kind of `ImageToVector`.
        kind = "discriminator"

        # Initialize base class.
        super().__init__(kernel_regularizer, bias_regularizer, params, kind)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def instantiate_img_to_vec_logits_layer(self, params):
        """Instantiates discriminator flatten and logits layers.

        Args:
            params: dict, user passed parameters.
        Returns:
            Flatten and logits layers of discriminator.
        """
        func_name = "instantiate_img_to_vec_logits_layer"
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Flatten layer to ready final block conv tensor for dense layer.
            flatten_layer = tf.layers.Flatten(
                name="{}_flatten_layer".format(self.name)
            )
            print_obj(func_name, "flatten_layer", flatten_layer)

            # Final linear layer for logits.
            logits_layer = equalized_learning_rate_layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=(
                    tf.random_normal_initializer(mean=0., stddev=1.0)
                    if params["use_equalized_learning_rate"]
                    else "he_normal"
                ),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                equalized_learning_rate=params["use_equalized_learning_rate"],
                name="{}_layers_dense_logits".format(self.name)
            )
            print_obj(func_name, "logits_layer", logits_layer)

        return flatten_layer, logits_layer

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def build_img_to_vec_base_conv_layer_block(self, params):
        """Creates discriminator base conv layer block.

        Args:
            params: dict, user passed parameters.

        Returns:
            List of tensors from base `Conv2D` layers.
        """
        func_name = "build_{}_base_conv_layer_block".format(self.kind)

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get conv block layer properties.
            conv_block = params["{}_base_conv_blocks".format(self.kind)][0]

            # The base conv block is always the 0th one.
            base_conv_layer_block = self.conv_layer_blocks[0]

            # batch_batch stddev comes before first base conv layer,
            # creating 1 extra feature map.
            if params["use_minibatch_stddev"]:
                # Therefore, the number of input channels will be 1 higher
                # for first base conv block.
                num_in_channels = conv_block[0][3] + 1
            else:
                num_in_channels = conv_block[0][3]

            # Get first base conv layer from list.
            first_base_conv_layer = base_conv_layer_block[0]

            # Build first layer with bigger tensor.
            base_conv_tensors = [
                first_base_conv_layer(
                    inputs=tf.zeros(
                        shape=[1] + conv_block[0][0:2] + [num_in_channels],
                        dtype=tf.float32
                    )
                )
            ]

            # Now build the rest of the base conv block layers, store in list.
            base_conv_tensors.extend(
                [
                    base_conv_layer_block[i](
                        inputs=tf.zeros(
                            shape=[1] + conv_block[i][0:3], dtype=tf.float32
                        )
                    )
                    for i in range(1, len(conv_block))
                ]
            )
            print_obj(
                "\n" + func_name, "base_conv_tensors", base_conv_tensors
            )

        return base_conv_tensors

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def minibatch_stddev_common(
            self,
            variance,
            tile_multiples,
            params,
            caller):
        """Adds minibatch stddev feature map to image using grouping.

        This is the code that is common between the grouped and ungroup
        minibatch stddev functions.

        Args:
            variance: tensor, variance of minibatch or minibatch groups.
            tile_multiples: list, length 4, used to tile input to final shape
                input_dims[i] * mutliples[i].
            params: dict, user passed parameters.
            caller: str, name of the calling function.

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [cur_batch_size, image_size, image_size, 1].
        """
        func_name = "minibatch_stddev_common".format(self.kind)

        with tf.variable_scope(
                "{}/{}_minibatch_stddev".format(self.name, caller)):
            # Calculate standard deviation over the group plus small epsilon.
            # shape = (
            #     {"grouped": cur_batch_size / group_size, "ungrouped": 1},
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            stddev = tf.sqrt(
                x=variance + 1e-8, name="{}_stddev".format(caller)
            )
            print_obj(func_name, "{}_stddev".format(caller), stddev)

            # Take average over feature maps and pixels.
            if params["minibatch_stddev_averaging"]:
                # grouped shape = (cur_batch_size / group_size, 1, 1, 1)
                # ungrouped shape = (1, 1, 1, 1)
                stddev = tf.reduce_mean(
                    input_tensor=stddev,
                    axis=[1, 2, 3],
                    keepdims=True,
                    name="{}_stddev_average".format(caller)
                )
                print_obj(
                    func_name, "{}_stddev_average".format(caller), stddev
                )

            # Replicate over group and pixels.
            # shape = (
            #     cur_batch_size,
            #     image_size,
            #     image_size,
            #     1
            # )
            stddev_feature_map = tf.tile(
                input=stddev,
                multiples=tile_multiples,
                name="{}_stddev_feature_map".format(caller)
            )
            print_obj(
                func_name,
                "{}_stddev_feature_map".format(caller),
                stddev_feature_map
            )

        return stddev_feature_map

    def grouped_minibatch_stddev(
            self,
            X,
            cur_batch_size,
            static_image_shape,
            params,
            group_size):
        """Adds minibatch stddev feature map to image using grouping.

        Args:
            X: tf.float32 tensor, image of shape
                [cur_batch_size, image_size, image_size, num_channels].
            cur_batch_size: tf.int64 tensor, the dynamic batch size (in case
                of partial batch).
            static_image_shape: list, the static shape of each image.
            params: dict, user passed parameters.
            group_size: int, size of image groups.

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [cur_batch_size, image_size, image_size, 1].
        """
        func_name = "grouped_minibatch_stddev".format(self.kind)

        with tf.variable_scope(
                "{}/grouped_minibatch_stddev".format(self.name)):
            # The group size should be less than or equal to the batch size.
            if params["use_tpu"]:
                group_size = min(group_size, cur_batch_size)
            else:
                # shape = ()
                group_size = tf.minimum(
                    x=group_size, y=cur_batch_size, name="group_size"
                )
            print_obj("\n" + func_name, "group_size", group_size)

            # Split minibatch into M groups of size group_size, rank 5 tensor.
            # shape = (
            #     group_size,
            #     cur_batch_size / group_size,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            grouped_image = tf.reshape(
                tensor=X,
                shape=[group_size, -1] + static_image_shape,
                name="grouped_image"
            )
            print_obj(func_name, "grouped_image", grouped_image)

            # Find the mean of each group.
            # shape = (
            #     1,
            #     cur_batch_size / group_size,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            grouped_mean = tf.reduce_mean(
                input_tensor=grouped_image,
                axis=0,
                keepdims=True,
                name="grouped_mean"
            )
            print_obj(func_name, "grouped_mean", grouped_mean)

            # Center each group using the mean.
            # shape = (
            #     group_size,
            #     cur_batch_size / group_size,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            centered_grouped_image = tf.subtract(
                x=grouped_image, y=grouped_mean, name="centered_grouped_image"
            )
            print_obj(
                func_name, "centered_grouped_image", centered_grouped_image
            )

            # Calculate variance over group.
            # shape = (
            #     cur_batch_size / group_size,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            grouped_variance = tf.reduce_mean(
                input_tensor=tf.square(x=centered_grouped_image),
                axis=0,
                name="grouped_variance"
            )
            print_obj(func_name, "grouped_variance", grouped_variance)

            # Get stddev image using ops common to both grouped & ungrouped.
            stddev_feature_map = self.minibatch_stddev_common(
                variance=grouped_variance,
                tile_multiples=[group_size] + static_image_shape[0:2] + [1],
                params=params,
                caller="grouped"
            )
            print_obj(func_name, "stddev_feature_map", stddev_feature_map)

        return stddev_feature_map

    def ungrouped_minibatch_stddev(
            self,
            X,
            cur_batch_size,
            static_image_shape,
            params):
        """Adds minibatch stddev feature map added to image channels.

        Args:
            X: tensor, image of shape
                [cur_batch_size, image_size, image_size, num_channels].
            cur_batch_size: tf.int64 tensor, the dynamic batch size (in case
                of partial batch).
            static_image_shape: list, the static shape of each image.
            params: dict, user passed parameters.

        Returns:
            Minibatch standard deviation feature map image added to
                channels of shape
                [cur_batch_size, image_size, image_size, 1].
        """
        func_name = "ungrouped_minibatch_stddev".format(self.kind)

        with tf.variable_scope(
                "{}/ungrouped_minibatch_stddev".format(self.name)):
            # Find the mean of each group.
            # shape = (
            #     1,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            mean = tf.reduce_mean(
                input_tensor=X, axis=0, keepdims=True, name="mean"
            )
            print_obj("\n" + func_name, "mean", mean)

            # Center each group using the mean.
            # shape = (
            #     cur_batch_size,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            centered_image = tf.subtract(
                x=X, y=mean, name="centered_image"
            )
            print_obj(func_name, "centered_image", centered_image)

            # Calculate variance over group.
            # shape = (
            #     1,
            #     image_size,
            #     image_size,
            #     num_channels
            # )
            variance = tf.reduce_mean(
                input_tensor=tf.square(x=centered_image),
                axis=0,
                keepdims=True,
                name="variance"
            )
            print_obj(func_name, "variance", variance)

            # Get stddev image using ops common to both grouped & ungrouped.
            stddev_feature_map = self.minibatch_stddev_common(
                variance=variance,
                tile_multiples=[cur_batch_size] + static_image_shape[0:2] + [1],
                params=params,
                caller="ungrouped"
            )
            print_obj(func_name, "stddev_feature_map", stddev_feature_map)

        return stddev_feature_map

    def minibatch_stddev(self, X, params, group_size=4):
        """Adds minibatch stddev feature map added to image.

        Args:
            X: tensor, image of shape
                [cur_batch_size, image_size, image_size, num_channels].
            params: dict, user passed parameters.
            group_size: int, size of image groups.

        Returns:
            Image with minibatch standard deviation feature map added to
                channels of shape
                [cur_batch_size, image_size, image_size, num_channels + 1].
        """
        func_name = "minibatch_stddev".format(self.kind)

        with tf.variable_scope("{}/minibatch_stddev".format(self.name)):
            # Get static shape of image.
            # shape = (3,)
            static_image_shape = params["generator_projection_dims"]
            print_obj(
                "\n" + func_name, "static_image_shape", static_image_shape
            )

            if params["use_tpu"]:
                if (params["batch_size"] % group_size == 0 or
                   params["batch_size"] < group_size):
                    stddev_feature_map = self.grouped_minibatch_stddev(
                        X=X,
                        cur_batch_size=params["batch_size"],
                        static_image_shape=static_image_shape,
                        params=params,
                        group_size=group_size
                    )
                else:
                    stddev_feature_map = self.ungrouped_minibatch_stddev(
                        X=X,
                        cur_batch_size=params["batch_size"],
                        static_image_shape=static_image_shape,
                        params=params
                    )
            else:
                # Get dynamic shape of image.
                # shape = (4,)
                dynamic_image_shape = tf.shape(
                    input=X, name="dynamic_image_shape"
                )
                print_obj(
                    func_name, "dynamic_image_shape", dynamic_image_shape
                )

                # Extract current batch size (in case this is a partial batch).
                cur_batch_size = dynamic_image_shape[0]

                # batch_size must be divisible by or smaller than group_size.
                divisbility_condition = tf.equal(
                    x=tf.mod(x=cur_batch_size, y=group_size),
                    y=0,
                    name="divisbility_condition"
                )

                less_than_condition = tf.less(
                    x=cur_batch_size, y=group_size, name="less_than_condition"
                )

                or_condition = tf.logical_or(
                    x=divisbility_condition,
                    y=less_than_condition,
                    name="or_condition"
                )

                # Get minibatch stddev feature map image from grouped or
                # ungrouped branch.
                stddev_feature_map = tf.cond(
                    pred=or_condition,
                    true_fn=lambda: self.grouped_minibatch_stddev(
                        X=X,
                        cur_batch_size=cur_batch_size,
                        static_image_shape=static_image_shape,
                        params=params,
                        group_size=group_size
                    ),
                    false_fn=lambda: self.ungrouped_minibatch_stddev(
                        X=X,
                        cur_batch_size=cur_batch_size,
                        static_image_shape=static_image_shape,
                        params=params
                    ),
                    name="stddev_feature_map_cond"
                )
            print_obj(func_name, "stddev_feature_map", stddev_feature_map)

            # Append to image as new feature map.
            # shape = (
            #     cur_batch_size,
            #     image_size,
            #     image_size,
            #     num_channels + 1
            # )
            appended_image = tf.concat(
                values=[X, stddev_feature_map],
                axis=-1,
                name="appended_image"
            )
            print_obj(func_name, "appended_image", appended_image)

        return appended_image

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def create_base_img_to_vec_network(self, X, params):
        """Creates base discriminator network.

        Args:
            X: tensor, input image to discriminator.
            params: dict, user passed parameters.

        Returns:
            Final logits tensor of discriminator.
        """
        func_name = "create_base_discriminator_network"

        print_obj("\n" + func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Only need the first fromRGB conv layer & block for base network.
            from_rgb_conv_layer = self.from_rgb_conv_layers[0]
            block_layers = self.conv_layer_blocks[0]

            # Pass inputs through layer chain.
            from_rgb_conv = from_rgb_conv_layer(inputs=X)
            print_obj(func_name, "from_rgb_conv", from_rgb_conv)

            from_rgb_conv = tf.nn.leaky_relu(
                features=from_rgb_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_from_rgb_conv_2d_leaky_relu".format(self.kind)
            )
            print_obj(func_name, "from_rgb_conv_leaky", from_rgb_conv)

            if params["use_minibatch_stddev"]:
                block_conv = self.minibatch_stddev(
                    X=from_rgb_conv,
                    params=params,
                    group_size=params["minibatch_stddev_group_size"]
                )
            else:
                block_conv = from_rgb_conv

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=block_conv, params=params
            )

        return logits

    def create_growth_transition_img_to_vec_network(
            self, X, alpha_var, params, trans_idx):
        """Creates growth transition discriminator network.

        Args:
            X: tensor, input image to img_to_vec.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Final logits tensor of discriminator.
        """
        func_name = "create_growth_transition_discriminator_network"

        print_obj("\nEntered {}".format(func_name), "trans_idx", trans_idx)
        print_obj(func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get weighted sum between shrinking and growing block paths.
            weighted_sum = self.create_growth_transition_img_to_vec_weighted_sum(
                X=X, alpha_var=alpha_var, params=params, trans_idx=trans_idx)
            print_obj(func_name, "weighted_sum", weighted_sum)

            # Get output of final permanent growth block's last `Conv2D` layer.
            block_conv = self.create_img_to_vec_perm_growth_block_network(
                block_conv=weighted_sum, params=params, trans_idx=trans_idx
            )
            print_obj(func_name, "block_conv", block_conv)

            # Conditionally add minibatch stddev as an additional feature map.
            if params["use_minibatch_stddev"]:
                block_conv = self.minibatch_stddev(
                    X=block_conv,
                    params=params,
                    group_size=params["minibatch_stddev_group_size"]
                )
                print_obj(func_name, "minibatch_stddev_block_conv", block_conv)

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=block_conv, params=params
            )
            print_obj(func_name, "logits", logits)

        return logits

    def create_growth_stable_img_to_vec_network(self, X, params, trans_idx):
        """Creates stable growth discriminator network.

        Args:
            X: tensor, input image to discriminator.
            params: dict, user passed parameters.
            trans_idx: int, index of current growth transition.

        Returns:
            Final logits tensor of discriminator.
        """
        func_name = "create_growth_stable_discriminator_network"

        print_obj("\n" + func_name, "X", X)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            # Get transition index fromRGB conv layer.
            from_rgb_conv_layer = self.from_rgb_conv_layers[trans_idx + 1]

            # Pass inputs through layer chain.
            from_rgb_conv = from_rgb_conv_layer(inputs=X)
            print_obj(func_name, "from_rgb_conv", from_rgb_conv)

            block_conv = tf.nn.leaky_relu(
                features=from_rgb_conv,
                alpha=params["{}_leaky_relu_alpha".format(self.kind)],
                name="{}_final_from_rgb_conv_2d_leaky_relu".format(self.kind)
            )
            print_obj(func_name, "from_rgb_conv_leaky", block_conv)

            # Get output of final permanent growth block's last `Conv2D` layer.
            block_conv = self.create_img_to_vec_perm_growth_block_network(
                block_conv=block_conv, params=params, trans_idx=trans_idx + 1
            )
            print_obj(func_name, "block_conv", block_conv)

            if params["use_minibatch_stddev"]:
                block_conv = self.minibatch_stddev(
                    X=block_conv,
                    params=params,
                    group_size=params["minibatch_stddev_group_size"]
                )
                print_obj(
                    func_name, "minibatch_stddev_block_conv", block_conv
                )

            # Get logits after continuing through base conv block.
            logits = self.create_base_img_to_vec_block_and_logits(
                block_conv=block_conv, params=params
            )

        return logits

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_gradient_penalty_loss(
            self,
            cur_batch_size,
            fake_images,
            real_images,
            alpha_var,
            params):
        """Gets discriminator gradient penalty loss.

        Args:
            cur_batch_size: tensor, in case of a partial batch instead of
                using the user passed int.
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [cur_batch_size, image_size, image_size, 3].
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Discriminator's gradient penalty loss of shape [].
        """
        func_name = "get_gradient_penalty_loss"

        with tf.name_scope(name="{}/gradient_penalty".format(self.name)):
            # Get a random uniform number rank 4 tensor.
            random_uniform_num = tf.random.uniform(
                shape=[cur_batch_size, 1, 1, 1],
                minval=0., maxval=1.,
                dtype=tf.float32,
                name="random_uniform_num"
            )
            print_obj(
                "\n" + func_name, "random_uniform_num", random_uniform_num
            )

            # Find the element-wise difference between images.
            image_difference = fake_images - real_images
            print_obj(func_name, "image_difference", image_difference)

            # Get random samples from this mixed image distribution.
            mixed_images = random_uniform_num * image_difference
            mixed_images += real_images
            print_obj(func_name, "mixed_images", mixed_images)

            # Send to the discriminator to get logits.
            mixed_logits = self.get_train_eval_img_to_vec_logits(
                X=mixed_images, alpha_var=alpha_var, params=params
            )
            print_obj(func_name, "mixed_logits", mixed_logits)

            # Get the mixed loss.
            mixed_loss = tf.reduce_sum(
                input_tensor=mixed_logits,
                name="mixed_loss"
            )
            print_obj(func_name, "mixed_loss", mixed_loss)

            # Get gradient from returned list of length 1.
            mixed_gradients = tf.gradients(
                ys=mixed_loss,
                xs=[mixed_images],
                name="gradients"
            )[0]
            print_obj(func_name, "mixed_gradients", mixed_gradients)

            # Get gradient's L2 norm.
            mixed_norms = tf.sqrt(
                x=tf.reduce_sum(
                    input_tensor=tf.square(
                        x=mixed_gradients,
                        name="squared_grads"
                    ),
                    axis=[1, 2, 3]
                ) + 1e-8
            )
            print_obj(func_name, "mixed_norms", mixed_norms)

            # Get squared difference from target of 1.0.
            squared_difference = tf.square(
                x=mixed_norms - 1.0,
                name="squared_difference"
            )
            print_obj(func_name, "squared_difference", squared_difference)

            # Get gradient penalty scalar.
            gradient_penalty = tf.reduce_mean(
                input_tensor=squared_difference, name="gradient_penalty"
            )
            print_obj(func_name, "gradient_penalty", gradient_penalty)

            # Multiply with lambda to get gradient penalty loss.
            gradient_penalty_loss = tf.multiply(
                x=params["discriminator_gradient_penalty_coefficient"],
                y=gradient_penalty,
                name="gradient_penalty_loss"
            )

        return gradient_penalty_loss

    def get_discriminator_loss(
            self,
            cur_batch_size,
            fake_images,
            real_images,
            fake_logits,
            real_logits,
            alpha_var,
            params):
        """Gets discriminator loss.

        Args:
            cur_batch_size: tensor, in case of a partial batch instead of
                using the user passed int.
            fake_images: tensor, images generated by the generator from random
                noise of shape [cur_batch_size, image_size, image_size, 3].
            real_images: tensor, real images from input of shape
                [cur_batch_size, image_size, image_size, 3].
            fake_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed generator's output image.
            real_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed real image.
            alpha_var: variable, alpha for weighted sum of fade-in of layers.
            params: dict, user passed parameters.

        Returns:
            Discriminator's total loss tensor of shape [].
        """
        func_name = "get_discriminator_loss"

        # Calculate base discriminator loss.
        discriminator_real_loss = tf.reduce_mean(
            input_tensor=real_logits,
            name="{}_real_loss".format(self.name)
        )
        print_obj(
            "\n" + func_name,
            "discriminator_real_loss",
            discriminator_real_loss
        )

        discriminator_generated_loss = tf.reduce_mean(
            input_tensor=fake_logits,
            name="{}_generated_loss".format(self.name)
        )
        print_obj(
            func_name,
            "discriminator_generated_loss",
            discriminator_generated_loss
        )

        discriminator_loss = tf.subtract(
            x=discriminator_generated_loss, y=discriminator_real_loss,
            name="{}_loss".format(self.name)
        )
        print_obj(
            func_name, "discriminator_loss", discriminator_loss
        )

        # Get discriminator gradient penalty loss.
        discriminator_gradient_penalty = self.get_gradient_penalty_loss(
            cur_batch_size=cur_batch_size,
            fake_images=fake_images,
            real_images=real_images,
            alpha_var=alpha_var,
            params=params
        )
        print_obj(
            func_name,
            "discriminator_gradient_penalty",
            discriminator_gradient_penalty
        )

        # Get discriminator epsilon drift penalty.
        epsilon_drift_penalty = tf.multiply(
            x=params["epsilon_drift"],
            y=tf.reduce_mean(input_tensor=tf.square(x=real_logits)),
            name="epsilon_drift_penalty"
        )
        print_obj(
            func_name, "epsilon_drift_penalty", epsilon_drift_penalty
        )

        # Get discriminator Wasserstein GP loss.
        discriminator_wasserstein_gp_loss = tf.add_n(
            inputs=[
                discriminator_loss,
                discriminator_gradient_penalty,
                epsilon_drift_penalty
            ],
            name="{}_wasserstein_gp_loss".format(self.name)
        )
        print_obj(
            func_name,
            "discriminator_wasserstein_gp_loss",
            discriminator_wasserstein_gp_loss
        )

        # Get discriminator regularization losses.
        discriminator_reg_loss = regularization.get_regularization_loss(
            lambda1=params["discriminator_l1_regularization_scale"],
            lambda2=params["discriminator_l2_regularization_scale"],
            scope=self.name
        )
        print_obj(
            func_name, "discriminator_reg_loss", discriminator_reg_loss
        )

        # Combine losses for total losses.
        discriminator_total_loss = tf.add(
            x=discriminator_wasserstein_gp_loss,
            y=discriminator_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(
            func_name, "discriminator_total_loss", discriminator_total_loss
        )

        if not params["use_tpu"]:
            # Add summaries for TensorBoard.
            tf.summary.scalar(
                name="discriminator_real_loss",
                tensor=discriminator_real_loss,
                family="losses"
            )
            tf.summary.scalar(
                name="discriminator_generated_loss",
                tensor=discriminator_generated_loss,
                family="losses"
            )
            tf.summary.scalar(
                name="discriminator_loss",
                tensor=discriminator_loss,
                family="losses"
            )
            tf.summary.scalar(
                name="discriminator_gradient_penalty",
                tensor=discriminator_gradient_penalty,
                family="losses"
            )
            tf.summary.scalar(
                name="epsilon_drift_penalty",
                tensor=epsilon_drift_penalty,
                family="losses"
            )
            tf.summary.scalar(
                name="discriminator_wasserstein_gp_loss",
                tensor=discriminator_wasserstein_gp_loss,
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
