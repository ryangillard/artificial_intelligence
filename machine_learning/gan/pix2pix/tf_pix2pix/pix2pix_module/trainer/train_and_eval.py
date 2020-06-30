import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def get_logits_and_losses(features, generator, discriminator, mode, params):
    """Gets logits and losses for both train and eval modes.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses"

    # Extract images from features dictionary.
    source_images = features["source_image"]
    real_target_images = features["target_image"]
    print_obj("\n" + func_name, "source_images", source_images)
    print_obj(func_name, "real_target_images", real_target_images)

    # Get generated target image from generator network from source image.
    print("\nCall generator with source_images = {}.".format(source_images))
    fake_target_images = generator.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(func_name, "fake_target_images", fake_target_images)

    # Resize fake target images to match real target image sizes.
    fake_target_images = image_utils.resize_fake_images(
        fake_images=fake_target_images, params=params
    )
    print_obj(func_name, "fake_target_images", fake_target_images)

    # Add summaries for TensorBoard.
    tf.summary.image(
        name="fake_target_images",
        tensor=tf.reshape(
            tensor=fake_target_images,
            shape=[-1, params["height"], params["width"], params["depth"]]
        ),
        max_outputs=5,
    )

    # Get fake logits from discriminator with generator's output target image.
    print(
        "\nCall discriminator with fake_target_images = {}.".format(
            fake_target_images
        )
    )
    fake_logits = discriminator.get_discriminator_logits(
        source_image=source_images,
        target_image=fake_target_images,
        params=params
    )
    print_obj(func_name, "fake_logits", fake_logits)

    # Get real logits from discriminator using real target image.
    print(
        "\nCall discriminator with real_target_images = {}.".format(
            real_target_images
        )
    )
    real_logits = discriminator.get_discriminator_logits(
        source_image=source_images,
        target_image=real_target_images,
        params=params
    )
    print_obj(func_name, "fake_logits", fake_logits)

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        fake_target_images=fake_target_images,
        real_target_images=real_target_images,
        fake_logits=fake_logits,
        params=params
    )

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        fake_logits=fake_logits, real_logits=real_logits, params=params
    )

    return (real_logits,
            fake_logits,
            generator_total_loss,
            discriminator_total_loss)
