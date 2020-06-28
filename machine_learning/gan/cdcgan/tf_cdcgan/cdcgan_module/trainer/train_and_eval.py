import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def get_logits_and_losses(
        features, labels, generator, discriminator, mode, params):
    """Gets logits and losses for both train and eval modes.

    Args:
        features: dict, feature tensors from serving input function.
        labels: tensor, labels to condition on of shape
            [cur_batch_size, 1].
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses"

    # For training discriminator.
    print("\nTraining discriminator.")

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
        dtype=tf.float32,
        name="discriminator_Z"
    )
    print_obj(func_name, "Z", Z)

    # Get generated image from generator network from gaussian noise.
    print("\nCall generator with Z = {}.".format(Z))
    fake_images = generator.get_fake_images(
        Z=Z, labels=labels, mode=mode, params=params
    )
    print_obj(func_name, "fake_images", fake_images)

    # Resize fake images to match real image sizes.
    fake_images = image_utils.resize_fake_images(fake_images, params)
    print_obj(func_name, "fake_images", fake_images)

    # Get fake logits from discriminator using generator's output image.
    print("\nCall discriminator with fake_images = {}.".format(fake_images))
    fake_logits = discriminator.get_discriminator_logits(
        X=fake_images, labels=labels, params=params
    )
    print_obj(func_name, "fake_logits", fake_logits)

    # Get real logits from discriminator using real image.
    print(
        "\nCall discriminator with real_images = {}.".format(real_images)
    )
    real_logits = discriminator.get_discriminator_logits(
        X=real_images, labels=labels, params=params
    )
    print_obj(func_name, "real_logits", real_logits)

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        fake_logits=fake_logits, real_logits=real_logits, params=params
    )
    print_obj(func_name, "discriminator_total_loss", discriminator_total_loss)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    # For training generator.
    print("\nTraining generator.")

    # Create random noise latent vector for each batch example.
    fake_Z = tf.random.normal(
        shape=[cur_batch_size, params["latent_size"]],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        name="generator_Z"
    )
    print_obj(func_name, "fake_Z", fake_Z)

    # Create random (fake) labels.
    fake_labels = tf.random.uniform(
        shape=[cur_batch_size, 1],
        minval=0,
        maxval=params["num_classes"],
        dtype=tf.int32,
        name="fake_labels"
    )
    print_obj(func_name, "fake_labels", fake_labels)

    # Get generated image from generator network from gaussian noise.
    print("\nCall generator with fake_Z = {}.".format(fake_Z))
    fake_fake_images = generator.get_fake_images(
        Z=fake_Z, labels=fake_labels, mode=mode, params=params
    )
    print_obj(func_name, "fake_fake_images", fake_fake_images)

    # Get fake logits from discriminator using generator's output image.
    print(
        "\nCall discriminator with fake_fake_images = {}.".format(
            fake_fake_images
        )
    )
    fake_fake_logits = discriminator.get_discriminator_logits(
        X=fake_fake_images, labels=fake_labels, params=params
    )
    print_obj(func_name, "fake_fake_logits", fake_fake_logits)

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        fake_logits=fake_fake_logits
    )
    print_obj(func_name, "generator_total_loss", generator_total_loss)

    # Add summaries for TensorBoard.
    tf.summary.image(
        name="fake_images",
        tensor=image_utils.resize_fake_images(
            fake_images=generator.get_fake_images(
                Z=tf.random.normal(
                    shape=[params["num_classes"], params["latent_size"]],
                    mean=0.0,
                    stddev=1.0,
                    dtype=tf.float32,
                    name="image_summary_Z"
                ),
                labels=tf.expand_dims(
                    input=tf.range(
                        start=0, limit=params["num_classes"], dtype=tf.int32
                    ),
                    axis=-1,
                    name="image_summary_fake_labels"
                ),
                mode=tf.estimator.ModeKeys.PREDICT,
                params=params
            ),
            params=params
        ),
        max_outputs=params["num_classes"],
    )

    return (real_logits,
            fake_logits,
            generator_total_loss,
            discriminator_total_loss)
