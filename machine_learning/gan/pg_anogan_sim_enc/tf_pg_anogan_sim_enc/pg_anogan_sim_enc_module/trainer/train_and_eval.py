import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def get_logits_and_losses(
        features, generator, discriminator, encoder, alpha_var, params):
    """Gets logits and losses for both train and eval modes.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        encoder: instance of encoder.`Encoder`.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator, discriminator, and encoder losses.
    """
    # Extract image from features dictionary.
    X = features["image"]
    print_obj("\nget_logits_and_losses", "X", X)

    # Get dynamic batch size in case of partial batch.
    cur_batch_size = tf.shape(
        input=X,
        out_type=tf.int32,
        name="get_logits_and_losses_cur_batch_size"
    )[0]

    # Create random noise latent vector for each batch example.
    Z = tf.random.normal(
        shape=[cur_batch_size, params["latent_size"]],
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32
    )
    print_obj("get_logits_and_losses", "Z", Z)

    # Get generated image from generator network from gaussian noise.
    print("\nCall generator with Z = {}.".format(Z))
    generator_outputs = generator.get_train_eval_generator_outputs(
        Z=Z, alpha_var=alpha_var, params=params
    )

    # Get fake logits from discriminator using generator's output image.
    print(
        "\nCall discriminator with generator_outputs = {}.".format(
            generator_outputs
        )
    )
    fake_logits = discriminator.get_discriminator_logits(
        X=generator_outputs, alpha_var=alpha_var, params=params
    )

    # Resize real images based on the current size of the GAN.
    real_images = image_utils.resize_real_images(image=X, params=params)

    # Get real logits from discriminator using real image.
    print(
        "\nCall discriminator with real_images = {}.".format(real_images)
    )
    real_logits = discriminator.get_discriminator_logits(
        X=real_images, alpha_var=alpha_var, params=params
    )

    # Get encoder logits using generated images from generator.
    print(
        "\nCall encoder with generator_outputs = {}.".format(
            generator_outputs
        )
    )
    encoder_logits = encoder.get_train_eval_encoder_logits(
        X=generator_outputs, alpha_var=alpha_var, params=params
    )

    # Get encoded images from generator using encoder's logits.
    print(
        "\nCall generator with encoder_logits = {}.".format(
            encoder_logits
        )
    )
    encoded_images = generator.get_train_eval_generator_outputs(
        Z=encoder_logits, alpha_var=alpha_var, params=params
    )

    # Get generator total loss.
    generator_total_loss = generator.get_generator_loss(
        fake_logits=fake_logits, params=params
    )

    # Get discriminator total loss.
    discriminator_total_loss = discriminator.get_discriminator_loss(
        cur_batch_size=cur_batch_size,
        fake_images=generator_outputs,
        real_images=real_images,
        fake_logits=fake_logits,
        real_logits=real_logits,
        alpha_var=alpha_var,
        params=params
    )

    # Get encoder total loss.
    encoder_total_loss = encoder.get_encoder_loss(
        fake_images=generator_outputs,
        encoded_images=encoded_images,
        params=params
    )

    return (real_logits,
            fake_logits,
            generator_total_loss,
            discriminator_total_loss,
            encoder_total_loss)
