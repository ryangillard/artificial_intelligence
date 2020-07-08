import tensorflow as tf

from .print_object import print_obj


class ImagePool():
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.

    Fields:
        pool_domain: str, name of the domain of images the pool contains.
        pool_capacity: int, the max size of the image pool.
        num_images: int, the current number of images in the pool.
        images: list, previously saved generated images.
    """
    def __init__(self, pool_domain, pool_capacity, params):
        """Initialize an `ImagePool`.

        Args:
            pool_domain: str, name of the domain of images the pool contains.
            pool_capacity: int, size of image buffer, if pool_capacity = 0, no
                buffer will be created.
            params: dict, user passed parameters.
        """
        self.pool_domain = pool_domain
        self.pool_capacity = pool_capacity
        # Create an empty pool
        if self.pool_capacity > 0:
            self._create_image_pool_variables(params=params)

    def _create_image_pool_variables(self, params):
        """Creates image pool variables.

        Args:
            params: dict, user passed parameters.
        """
        func_name = "_create_image_pool_variables_{}".format(self.pool_domain)

        with tf.variable_scope(
            name_or_scope="image_pool_{}".format(self.pool_domain),
            reuse=tf.AUTO_REUSE
        ):
            # Keeps tally of how many images are currently in the image pool.
            self.num_images_var = tf.get_variable(
                name="num_images",
                initializer=tf.zeros(shape=[], dtype=tf.int32),
                trainable=False
            )
            print_obj(func_name, "num_images_var", self.num_images_var)

            # Contains previous generated images from generator.
            self.images_var = tf.get_variable(
                name="images",
                initializer=tf.zeros(
                    shape=[
                        self.pool_capacity,
                        params["height"],
                        params["width"],
                        params["depth"]
                    ],
                    dtype=tf.float32
                ),
                trainable=False
            )
            print_obj(func_name, "images_var", self.images_var)

    def return_pool_variable_values(self):
        """Returns image pool variable values.

        Returns:
            Returns number of images currently in pool of shape [] and current
                images in pool of shape [pool_capacity, height, width, depth].
        """
        return self.num_images_var.value(), self.images_var.value()

    def _add_images_to_not_full_pool(self, images):
        """Adds images to pool iff pool isn't already full.

        Args:
            images: tensor, generated images of shape
                [batch_size, height, width, depth].
        Returns:
            A tensor of the new images added to the pool of shape
                [num_new_images_to_add, height, width, depth], a tensor of the
                number of new images added to the pool of shape [], and a
                tensor of the number of new images still remaining to add of
                shape [].
        """
        func_name = "_add_images_to_not_full_pool"
        print_obj("\n" + func_name, "images", images)

        # Get batch size.
        batch_size = images.shape[0].value

        # Calculate remaining capacity.
        remaining_capacity = self.pool_capacity - self.num_images_var

        # Find how many images we can add to not full pool.
        num_new_images_to_add = tf.minimum(
            x=batch_size, y=remaining_capacity
        )

        # Check if there will be any remaining new images to add later.
        num_new_images_remaining = batch_size - num_new_images_to_add

        # Slice full new images tensor for just the ones to add to not
        # overflow capacity.
        new_images_add = images[0:num_new_images_to_add, :, :, :]
        print_obj(func_name, "new_images_add", new_images_add)

        # Assign new images to pool.
        images_var_assign_op = self.images_var.assign(
            value=tf.concat(
                values=[
                    self.images_var[0:self.num_images_var, :, :, :],
                    new_images_add,
                    self.images_var[self.num_images_var + num_new_images_to_add:, :, :, :],
                ],
                axis=0
            ),
            use_locking=True,
            read_value=False
        )
        print_obj(func_name, "images_var_assign_op", images_var_assign_op)

        with tf.control_dependencies(control_inputs=[images_var_assign_op]):
            # Assign new image count.
            num_images_var_assign_op = self.num_images_var.assign_add(
                delta=num_new_images_to_add,
                use_locking=True,
                read_value=False
            )
            print_obj(
                func_name, "num_images_var_assign_op", num_images_var_assign_op
            )

            with tf.control_dependencies(
                control_inputs=[
                    num_images_var_assign_op
                ]
            ):
                num_new_images_remaining = tf.identity(
                    input=num_new_images_remaining
                )
                print_obj(
                    func_name,
                    "num_new_images_remaining",
                    num_new_images_remaining
                )

        return (new_images_add,
                num_new_images_to_add,
                num_new_images_remaining
                )

    def _select_previous_images(self, num_new_images_remaining):
        """Selects previous images from pool.

        Args:
            num_new_images_remaining: tensor, number of new images still
                remaining to add of shape [].
        Returns:
            Tensor of previous images from the pool of shape
                [num_new_images_remaining, height, width, depth].
        """
        func_name = "_select_previous_images"
        print_obj(
            "\n" + func_name,
            "num_new_images_remaining",
            num_new_images_remaining
        )

        # Shuffle indices to get a random, unique ordering.
        shuffled_indices = tf.random.shuffle(
            value=tf.range(
                start=0, limit=self.num_images_var.value(), dtype=tf.int32
            ),
            name="{}_random_shuffle".format(self.pool_domain)
        )
        print_obj(func_name, "shuffled_indices", shuffled_indices)

        # Since it is shuffled, take just as many stored images as needed.
        sliced_indices = shuffled_indices[0:num_new_images_remaining]
        print_obj(func_name, "sliced_indices", sliced_indices)

        # Gather images using indices.
        prev_images = tf.gather(
            params=self.images_var,
            indices=sliced_indices,
            axis=0,
            batch_dims=1,
            name="{}_gathered_prev_images".format(self.pool_domain)
        )
        print_obj(func_name, "prev_images", prev_images)

        return prev_images

    def _get_mix_of_new_and_prev_images(
            self, images, num_new_images_added, num_new_images_remaining):
        """Gets mix of new and previous images from the pool.

        Args:
            images: tensor, generated images of shape
                [batch_size, height, width, depth].
            num_new_images_added: tensor, number of new images already added
                to the pool of shape [].
            num_new_images_remaining: tensor, number of new images still
                remaining to add of shape [].
        Returns:
            Tensor of new and previous images from the pool mixed of shape
                [num_new_images_remaining, height, width, depth].
        """
        func_name = "_get_mix_of_new_and_prev_images"
        print_obj("\n" + func_name, "images", images)

        # Randomly choose previous images.
        prev_images = self._select_previous_images(
            num_new_images_remaining
        )
        print_obj(func_name, "prev_images", prev_images)

        # Determine what branch to do with 50%/50% probability.
        random_probs = tf.random.uniform(
            shape=[num_new_images_remaining],
            minval=0.,
            maxval=1.,
            dtype=tf.float32,
            name="{}_random_probabilities".format(self.pool_domain)
        )
        print_obj(func_name, "random_probs", random_probs)

        # With 50% probability, choose whether to use previous or new image
        # for each remaining image slot.
        mixed_images = tf.where(
            condition=tf.greater(x=random_probs, y=0.5),
            x=prev_images,
            y=images[num_new_images_added:, :, :, :],
        )
        print_obj(func_name, "mixed_images", mixed_images)

        # Gather previous images to keep and not overwrite.
        prev_images_to_keep = tf.gather(
            params=self.images_var,
            indices=tf.range(
                start=num_new_images_remaining,
                limit=self.pool_capacity,
                dtype=tf.int32
            ),
            axis=0,
            batch_dims=1
        )
        print_obj(func_name, "prev_images_to_keep", prev_images_to_keep)

        # Combine previous images to keep with mixed images for mixed pool.
        mixed_pool = tf.concat(
            values=[prev_images_to_keep, mixed_images], axis=0
        )
        print_obj(func_name, "mixed_pool", mixed_pool)

        # Assign to image pool variable.
        images_var_assign_op = self.images_var.assign(
            value=mixed_pool,
            use_locking=True,
            read_value=False
        )
        print_obj(func_name, "images_var_assign_op", images_var_assign_op)

        # Make sure pool image variable assignment is complete.
        with tf.control_dependencies(control_inputs=[images_var_assign_op]):
            # With 50% probability, choose whether to return previous or new
            # images for each image.
            return_images = tf.where(
                condition=tf.greater(x=random_probs, y=0.5),
                x=prev_images,
                y=images[num_new_images_added:, :, :, :]
            )
            print_obj(func_name, "return_images", return_images)

        return return_images

    def query(self, images, params):
        """Returns an image from the pool.

        50% chance buffer will return input images or 50% chance buffer will
        return previous images previously and put current images into buffer.

        Args:
            images: tensor, the latest generated images from the generator of
                shape [batch_size, height, width, depth].
        Returns:
            Mix of new images and images from the buffer.
        """
        func_name = "query"
        print_obj("\n" + func_name, "num_new_images_remaining", images)

        if self.pool_capacity == 0:
            # If the buffer capacity is zero, then just return input images.
            return images

        # If still room in buffer, put current images into it.
        # Add images to pool if not full.
        (new_return_images,
         num_new_images_added,
         num_new_images_remaining
         ) = self._add_images_to_not_full_pool(images)
        print_obj(func_name, "new_return_images", new_return_images)

        # If there are still some new images remaining.
        mix_return_images = self._get_mix_of_new_and_prev_images(
            images, num_new_images_added, num_new_images_remaining
        )
        print_obj(func_name, "mix_return_images", mix_return_images)

        # Concatenate both sets of return images.
        return_images = tf.concat(
            values=[new_return_images, mix_return_images],
            axis=0,
            name="{}_updated_return_list".format(self.pool_domain)
        )
        print_obj(func_name, "return_images", return_images)

        return return_images
