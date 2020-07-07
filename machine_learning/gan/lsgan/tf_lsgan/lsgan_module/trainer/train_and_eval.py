import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def get_logits_and_losses(features, generator, discriminator, mode, params):
    """Gets logits and losses for both train and eval modes.

    Args:
        features: dict, feature tensors from input function.
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses"
    # Extract real images from features dictionary.
    real_images = features["image"]
    print_obj("\n" + func_name, "real_images", real_images)

    # Get dynamic batch size in case of partial batch.
    cur_batch_size = tf.shape(
        input=real_images,
        out_type=tf.int32,
        name="{}_cur_batch_size".format(func_name)
    )[0]

    # Create random noise latent vector for each batch example.
    Z = tf.random.normal(
        shape=[cur_batch_size, params["latent_size"]],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32
    )
    print_obj(func_name, "Z", Z)

    # Get generated image from generator network from gaussian noise.
    print("\nCall generator with Z = {}.".format(Z))
    fake_images = generator.get_fake_images(Z=Z, mode=mode, params=params)

    # Resize fake images to match real image sizes.
    fake_images = image_utils.resize_fake_images(fake_images, params)
    print_obj(func_name, "fake_images", fake_images)

    # Add summaries for TensorBoard.
    tf.summary.image(
        name="fake_images",
        tensor=tf.reshape(
            tensor=fake_images,
            shape=[-1, params["height"], params["width"], params["depth"]]
        ),
        max_outputs=5,
    )

    # Get fake logits from discriminator using generator's output image.
    print("\nCall discriminator with fake_images = {}.".format(fake_images))
    fake_logits = discriminator.get_discriminator_logits(
        X=fake_images, mode=mode, params=params
    )

    # Get real logits from discriminator using real image.
    print(
        "\nCall discriminator with real_images = {}.".format(real_images)
    )
    real_logits = discriminator.get_discriminator_logits(
        X=real_images, mode=mode, params=params
    )

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        fake_logits=fake_logits, params=params
    )

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        fake_logits=fake_logits, real_logits=real_logits, params=params
    )

    return (real_logits,
            fake_logits,
            generator_total_loss,
            discriminator_total_loss)
