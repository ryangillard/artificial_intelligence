import tensorflow as tf


def generator_loss_phase(
    global_batch_size,
    generator,
    discriminator,
    params,
    global_step,
    summary_file_writer,
    mode,
    training
):
    """Gets fake logits and loss for generator.

    Args:
        global_batch_size: int, global batch size for distribution.
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.
        training: bool, if in training mode.

    Returns:
        Fake logits of shape [batch_size, 1] and generator loss.
    """
    batch_size = (
        params["train_batch_size"]
        if mode == "TRAIN"
        else params["eval_batch_size"]
    )

    # Create random noise latent vector for each batch example.
    Z = tf.random.normal(
        shape=[batch_size, params["latent_size"]],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32
    )

    # Get generated image from generator network from gaussian noise.
    fake_images = generator.get_model()(inputs=Z, training=training)

    if mode == "TRAIN":
        # Add summaries for TensorBoard.
        with summary_file_writer.as_default():
            with tf.summary.record_if(
                global_step % params["save_summary_steps"] == 0
            ):
                tf.summary.image(
                    name="fake_images",
                    data=tf.reshape(
                        tensor=fake_images,
                        shape=[
                            -1,
                            params["height"],
                            params["width"],
                            params["depth"]
                        ]
                    ),
                    step=global_step,
                    max_outputs=5,
                )
                summary_file_writer.flush()

    # Get fake logits from discriminator using generator's output image.
    fake_logits = discriminator.get_model()(
        inputs=fake_images, training=False
    )

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        global_batch_size=global_batch_size,
        fake_logits=fake_logits,
        params=params,
        global_step=global_step,
        summary_file_writer=summary_file_writer
    )

    return fake_logits, generator_total_loss


def discriminator_loss_phase(
    global_batch_size,
    discriminator,
    real_images,
    fake_logits,
    params,
    global_step,
    summary_file_writer,
    mode,
    training
):
    """Gets real logits and loss for discriminator.

    Args:
        global_batch_size: int, global batch size for distribution.
        discriminator: instance of `Discriminator`.
        real_images: tensor, real images of shape
            [batch_size, height * width * depth].
        fake_logits: tensor, discriminator logits of fake images of shape
            [batch_size, 1].
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.
        training: bool, if in training mode.

    Returns:
        Real logits and discriminator loss.
    """
    # Get real logits from discriminator using real image.
    real_logits = discriminator.get_model()(
        inputs=real_images, training=training
    )

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        global_batch_size=global_batch_size,
        fake_logits=fake_logits,
        real_logits=real_logits,
        params=params,
        global_step=global_step,
        summary_file_writer=summary_file_writer
    )

    return real_logits, discriminator_total_loss
