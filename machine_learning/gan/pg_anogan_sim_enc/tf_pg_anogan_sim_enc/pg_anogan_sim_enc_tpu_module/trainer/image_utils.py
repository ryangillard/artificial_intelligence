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
    # Convert from [0, 255] -> [-1.0, 1.0] floats.
    image = tf.cast(x=image, dtype=tf.float32) * (2. / 255) - 1.0
    print_obj("preprocess_image", "image", image)

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
    print_obj("\nresize_real_image", "block_idx", block_idx)
    print_obj("resize_real_image", "image", image)

    # Resize image to match GAN size at current block index.
    resized_image = tf.image.resize(
        images=image,
        size=[
            params["generator_projection_dims"][0] * (2 ** block_idx),
            params["generator_projection_dims"][1] * (2 ** block_idx)
        ],
        method="nearest",
        name="resize_real_images_resized_image_{}".format(block_idx)
    )
    print_obj("resize_real_images", "resized_image", resized_image)

    return resized_image


def resize_real_images(image, params):
    """Resizes real images to match the GAN's current size.

    Args:
        image: tensor, original image.
        params: dict, user passed parameters.

    Returns:
        Resized image tensor.
    """
    print_obj("\nresize_real_images", "image", image)
    # Resize real image for each block.
    train_steps = params["train_steps"] + params["prev_train_steps"]
    num_steps_until_growth = params["num_steps_until_growth"]
    num_stages = train_steps // num_steps_until_growth
    if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
        print(
            "\nresize_real_images: NEVER GOING TO GROW, SKIP SWITCH CASE!"
        )
        # If we never are going to grow, no sense using the switch case.
        # 4x4
        resized_image = resize_real_image(
            image=image, params=params, block_idx=0
        )
    else:
        if params["use_tpu"]:
            block_idx = min(
                params["growth_idx"], len(params["conv_num_filters"]) - 1
            )
            resized_image = resize_real_image(
                image=image, params=params, block_idx=block_idx
            )
        else:
            # Find growth index based on global step and growth frequency.
            growth_index = tf.cast(
                x=tf.floordiv(
                    x=tf.train.get_or_create_global_step(),
                    y=params["num_steps_until_growth"],
                    name="resize_real_images_global_step_floordiv"
                ),
                dtype=tf.int32,
                name="resize_real_images_growth_index"
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
                name="resize_real_images_switch_case_resized_image"
            )
    print_obj("resize_real_images", "resized_image", resized_image)

    return resized_image
