import tensorflow as tf

from .print_object import print_obj


def decode_image(image_bytes, params):
    """Decodes image bytes tensor.

    Args:
        image_bytes: tensor, image bytes with shape [?,].
        params: dict, user passed parameters.

    Returns:
        Decoded image tensor of shape [height, width, depth].
    """
    func_name = "decode_image"
    print_obj("\n" + func_name, "image_bytes", image_bytes)

    # Convert from a scalar string tensor (whose single string has
    # length height * width * depth) to a uint8 tensor with shape
    # [height * width * depth].
    image = tf.decode_raw(
        input_bytes=image_bytes,
        out_type=tf.uint8,
        name="image_decoded"
    )
    print_obj(func_name, "image", image)

    # Reshape flattened image back into normal dimensions.
    image = tf.reshape(
        tensor=image,
        shape=[params["height"], params["width"], params["depth"]],
        name="image_reshaped"
    )
    print_obj(func_name, "image", image)

    return image


def preprocess_image(image, mode, params):
    """Preprocess image tensor.

    Args:
        image: tensor, input image with shape
            [cur_batch_size, height, width, depth].
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Preprocessed image tensor with shape
            [cur_batch_size, height, width, depth].
    """
    func_name = "preprocess_image"
    print_obj("\n" + func_name, "image", image)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Add some random jitter.
        if params["preprocess_image_resize_jitter_size"]:
            image = tf.image.resize(
                images=image,
                size=params["preprocess_image_resize_jitter_size"],
                method="bilinear",
                name="{}_jitter_resize".format(func_name)
            )
            print_obj(func_name, "image", image)

            image = tf.image.random_crop(
                value=image,
                size=[params["height"], params["width"], params["depth"]],
                name="{}_jitter_crop".format(func_name)
            )
            print_obj(func_name, "image", image)

    # Convert from [0, 255] -> [-1.0, 1.0] floats.
    image = tf.subtract(
        x=tf.cast(x=image, dtype=tf.float32) * (2. / 255),
        y=1.0,
        name="{}_scaled".format(func_name)
    )
    print_obj(func_name, "image", image)

    return image


def handle_input_image(image_bytes, mode, params):
    """Handles image tensor transformations.

    Args:
        image_bytes: tensor, image bytes with shape [?,].
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Preprocessed image tensor with shape
            [cur_batch_size, height, width, depth].
    """
    func_name = "handle_input_image"
    print_obj("\n" + func_name, "image_bytes", image_bytes)

    # Decode image.
    image = decode_image(image_bytes=image_bytes, params=params)
    print_obj(func_name, "image", image)

    # Preprocess image.
    image = preprocess_image(image=image, mode=mode, params=params)
    print_obj(func_name, "image", image)

    return image


def resize_fake_images(fake_images, params):
    """Resizes fake images to match real image sizes.

    Args:
        fake_images: tensor, fake images from generator.
        params: dict, user passed parameters.

    Returns:
        Resized image tensor.
    """
    func_name = "resize_fake_images"
    print_obj("\n" + func_name, "fake_images", fake_images)

    # Resize fake images to match real image sizes.
    resized_fake_images = tf.image.resize(
        images=fake_images,
        size=[params["height"], params["width"]],
        method="nearest",
        name="resized_fake_images"
    )
    print_obj(func_name, "resized_fake_images", resized_fake_images)

    return resized_fake_images
