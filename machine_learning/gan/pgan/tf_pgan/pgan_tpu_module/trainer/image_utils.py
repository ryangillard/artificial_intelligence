import tensorflow as tf

from .print_object import print_obj


def preprocess_image(image, params):
    """Preprocess image tensor.

    Args:
        image: tensor, input image with shape
            [cur_batch_size, height, width, depth].
        params: dict, user passed parameters.

    Returns:
        Preprocessed image tensor with shape
            [cur_batch_size, height, width, depth].
    """
    func_name = "preprocess_image"
    # Convert from [0, 255] -> [-1.0, 1.0] floats.
    image = tf.cast(x=image, dtype=tf.float32) * (2. / 255) - 1.0
    print_obj(func_name, "image", image)

    return image


def resize_real_image(image, params, block_idx):
    """Resizes real images to match the GAN's current size.

    Args:
        image: tensor, original image.
        params: dict, user passed parameters.
        block_idx: int, index of current block.

    Returns:
        Resized image tensor.
    """
    func_name = "resize_real_image"
    print_obj("\n" + func_name, "block_idx", block_idx)
    print_obj(func_name, "image", image)

    # Resize image to match GAN size at current block index.
    resized_image = tf.image.resize(
        images=image,
        size=[
            params["generator_projection_dims"][0] * (2 ** block_idx),
            params["generator_projection_dims"][1] * (2 ** block_idx)
        ],
        method="nearest",
        name="{}_resized_image_{}".format(func_name, block_idx)
    )
    print_obj(func_name, "resized_image", resized_image)

    return resized_image


def resize_real_images(image, params):
    """Resizes real images to match the GAN's current size.

    Args:
        image: tensor, original image.
        params: dict, user passed parameters.

    Returns:
        Resized image tensor.
    """
    func_name = "resize_real_images"
    print_obj("\n" + func_name, "image", image)
    # Resize real image for each block.
    if len(params["conv_num_filters"]) == 1:
        print(
            "\n: NEVER GOING TO GROW, SKIP SWITCH CASE!".format(func_name)
        )
        # If we never are going to grow, no sense using the switch case.
        # 4x4
        resized_image = resize_real_image(
            image=image, params=params, block_idx=0
        )
    else:
        if params["growth_idx"] is not None:
            block_idx = min(
                (params["growth_idx"] - 1) // 2 + 1,
                len(params["conv_num_filters"]) - 1
            )
            resized_image = resize_real_image(
                image=image, params=params, block_idx=block_idx
            )
        else:
            # Find growth index based on global step and growth frequency.
            growth_index = tf.add(
                x=tf.floordiv(
                    x=tf.minimum(
                        x=tf.cast(
                            x=tf.floordiv(
                                x=tf.train.get_or_create_global_step() - 1,
                                y=params["num_steps_until_growth"],
                                name="{}_global_step_floordiv".format(
                                    func_name
                                )
                            ),
                            dtype=tf.int32
                        ),
                        y=(len(params["conv_num_filters"]) - 1) * 2
                    ) - 1,
                    y=2
                ),
                y=1,
                name="{}_growth_index".format(func_name)
            )

            # Switch to case based on number of steps for resized image.
            resized_image = tf.switch_case(
                branch_index=growth_index,
                branch_fns=[
                    # 4x4
                    lambda: resize_real_image(
                        image=image, params=params, block_idx=0
                    ),
                    # 8x8
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(1, len(params["conv_num_filters"]) - 1)
                    ),
                    # 16x16
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(2, len(params["conv_num_filters"]) - 1)
                    ),
                    # 32x32
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(3, len(params["conv_num_filters"]) - 1)
                    ),
                    # 64x64
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(4, len(params["conv_num_filters"]) - 1)
                    ),
                    # 128x128
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(5, len(params["conv_num_filters"]) - 1)
                    ),
                    # 256x256
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(6, len(params["conv_num_filters"]) - 1)
                    ),
                    # 512x512
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(7, len(params["conv_num_filters"]) - 1)
                    ),
                    # 1024x1024
                    lambda: resize_real_image(
                        image=image,
                        params=params,
                        block_idx=min(8, len(params["conv_num_filters"]) - 1)
                    )
                ],
                name="{}_switch_case_resized_image".format(func_name)
            )
    print_obj(func_name, "resized_image", resized_image)

    return resized_image
