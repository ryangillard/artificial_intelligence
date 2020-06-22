import tensorflow as tf

from .print_object import print_obj


def get_logits_and_losses(features, labels, generator, discriminator, params):
    """Gets logits and losses for both train and eval modes.

    Args:
        features: dict, feature tensors from serving input function.
        labels: tensor, labels to condition on of shape
            [cur_batch_size, 1].
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses"

    # For training discriminator.
    print("\nTraining discriminator.")

    # Extract real images from features dictionary.
    real_images = tf.reshape(
        tensor=features["image"],
        shape=[-1, params["height"] * params["width"] * params["depth"]],
        name="real_images"
    )
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
    fake_images = generator.get_fake_images(Z=Z, labels=labels, params=params)

    # Get fake logits from discriminator using generator's output image.
    print("\nCall discriminator with fake_images = {}.".format(fake_images))
    fake_logits = discriminator.get_discriminator_logits(
        X=fake_images, labels=labels, params=params
    )

    # Get real logits from discriminator using real image.
    print(
        "\nCall discriminator with real_images = {}.".format(real_images)
    )
    real_logits = discriminator.get_discriminator_logits(
        X=real_images, labels=labels, params=params
    )

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        fake_logits=fake_logits, real_logits=real_logits, params=params
    )

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
        Z=fake_Z, labels=fake_labels, params=params
    )

    # Get fake logits from discriminator using generator's output image.
    print(
        "\nCall discriminator with fake_fake_images = {}.".format(
            fake_fake_images
        )
    )
    fake_fake_logits = discriminator.get_discriminator_logits(
        X=fake_fake_images, labels=fake_labels, params=params
    )

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        fake_logits=fake_fake_logits
    )

    # Add summaries for TensorBoard.
    tf.summary.image(
        name="fake_images",
        tensor=tf.reshape(
            tensor=generator.get_fake_images(
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
                params=params
            ),
            shape=[-1, params["height"], params["width"], params["depth"]],
            name="image_summary_fake_fake_images"
        ),
        max_outputs=params["num_classes"],
    )

    return (real_logits,
            fake_logits,
            generator_total_loss,
            discriminator_total_loss)
