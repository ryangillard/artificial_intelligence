import tensorflow as tf

from . import image_pool
from .print_object import print_obj


def get_logits_and_losses(
        source_images,
        real_target_images,
        generator,
        discriminator,
        image_pool,
        mode,
        params,
        domain_index):
    """Gets logits and losses for both train and eval modes.

    Args:
        source_images: tensor, images of source domain.
        real_target_images: tensor, real images of target domain.
        generator: instance of generator.`Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        image_pool: instance of `ImagePool` buffer to store generated images
            for disriminator.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.
        domain_index: int, index of current domain.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses"
    print_obj(func_name, "source_images", source_images)
    print_obj(func_name, "real_target_images", real_target_images)

    # Get generated target image from generator network from source image.
    print("\nCall generator with source_images = {}.".format(source_images))
    fake_target_images = generator.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(func_name, "fake_target_images", fake_target_images)

    # Add summaries for TensorBoard.
    tf.summary.image(
        name="fake_target_images_domain{}".format(domain_index + 1),
        tensor=tf.reshape(
            tensor=fake_target_images,
            shape=[-1, params["height"], params["width"], params["depth"]]
        ),
        max_outputs=5,
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Add generated images to `ImagePool` and retrieve previous ones.
        fake_target_images = image_pool.query(
            images=fake_target_images, params=params
        )
        print_obj(func_name, "fake_target_images", fake_target_images)

    # Get fake logits from discriminator with generator's output target image.
    print(
        "\nCall discriminator with fake_target_images = {}.".format(
            fake_target_images
        )
    )
    fake_logits = discriminator.get_discriminator_logits(
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
        target_image=real_target_images,
        params=params
    )
    print_obj(func_name, "fake_logits", fake_logits)

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
            discriminator_total_loss
            )


def get_cycle_consistency_loss(
        source_images, generator1, generator2, params, domain_index):
    """Gets cycle consistency L1 loss between source and fake source images.

    Args:
        source_images: tensor, source images of domain 2.
        generator1: instance of `Generator` of domain 1.
        generator2: instance of `Generator` of domain 2.
        params: dict, user passed parameters.
        domain_index: int, index of current domain.

    Returns:
        Cycle L1 loss between source and fake source images.
    """
    func_name = "get_cycle_consistency_loss"
    print_obj("\n" + func_name, "source_images", source_images)

    # Get generated target image from generator network from source image.
    print(
        "\nCall generator{} with source_images = {}.".format(
            domain_index, source_images
        )
    )
    fake_target_images = generator1.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(func_name, "fake_target_images", fake_target_images)

    # Get generated target image from generator network from source image.
    print(
        "\nCall generator{} with fake_target_images = {}.".format(
            1 - domain_index, fake_target_images
        )
    )
    fake_source_images = generator2.get_fake_images(
        source_images=fake_target_images, params=params
    )
    print_obj(func_name, "fake_source_images", fake_source_images)

    # Get L1 loss from difference between source and fake source images.
    cycle_l1_loss = tf.reduce_mean(
        input_tensor=tf.abs(
            x=tf.subtract(
                x=source_images, y=fake_source_images
            )
        ),
        name="cycle_l1_loss_{}".format(domain_index)
    ) + 1e-8
    print_obj(func_name, "cycle_l1_loss", cycle_l1_loss)

    return cycle_l1_loss


def get_identity_loss(real_target_images, generator, params, domain_index):
    """Gets identity loss between target and fake target images.

    Args:
        real_target_images: tensor, real images of target domain.
        generator: instance of `Generator`.
        params: dict, user passed parameters.
        domain_index: int, index of current domain.

    Returns:
        Identity loss between target and fake target images.
    """
    func_name = "get_identity_loss"
    print_obj("\n" + func_name, "real_target_images", real_target_images)

    # Get generated target image from generator network from target image.
    print(
        "\nCall generator{} with real_target_images = {}.".format(
            domain_index, real_target_images
        )
    )
    fake_target_images = generator.get_fake_images(
        source_images=real_target_images, params=params
    )
    print_obj(func_name, "fake_target_images", fake_target_images)

    # Get L1 loss from difference between real and fake target images.
    identity_l1_loss = tf.reduce_mean(
        input_tensor=tf.abs(
            x=tf.subtract(
                x=real_target_images, y=fake_target_images
            )
        ),
        name="identity_l1_loss_{}".format(domain_index)
    ) + 1e-8
    print_obj(func_name, "identity_l1_loss", identity_l1_loss)

    return identity_l1_loss


def get_logits_and_losses_combined_domains(
        features,
        generator_domain_a2b,
        discriminator_domain_b,
        generator_domain_b2a,
        discriminator_domain_a,
        mode,
        params):
    """Gets logits and losses for both train and eval modes for both domains.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of `Generator`.
        discriminator: instance of discriminator.`Discriminator`.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Real and fake logits and generator and discriminator losses.
    """
    func_name = "get_logits_and_losses_combined_domains"

    # Extract images from features dictionary.
    domain_a_images = features["domain_a_image"]
    domain_b_images = features["domain_b_image"]
    print_obj("\n" + func_name, "domain_a_images", domain_a_images)
    print_obj(func_name, "domain_b_images", domain_b_images)

    # Instantiate `ImagePool` buffers to store generated images.
    image_pool_domain_a = image_pool.ImagePool(
        pool_domain="domain_a",
        pool_capacity=params["image_pool_capacity"],
        params=params
    )

    image_pool_domain_b = image_pool.ImagePool(
        pool_domain="domain_b",
        pool_capacity=params["image_pool_capacity"],
        params=params
    )

    # Get logits and losses using domain a as source and domain b as target.
    (real_logits_domain_b,
     fake_logits_domain_b,
     generator_domain_a2b_total_loss,
     discriminator_domain_b_total_loss) = get_logits_and_losses(
        source_images=domain_a_images,
        real_target_images=domain_b_images,
        generator=generator_domain_a2b,
        discriminator=discriminator_domain_b,
        image_pool=image_pool_domain_b,
        mode=mode,
        params=params,
        domain_index=0
    )

    # Get logits and losses using domain b as source and domain a as target.
    (real_logits_domain_a,
     fake_logits_domain_a,
     generator_domain_b2a_total_loss,
     discriminator_domain_a_total_loss) = get_logits_and_losses(
        source_images=domain_b_images,
        real_target_images=domain_a_images,
        generator=generator_domain_b2a,
        discriminator=discriminator_domain_a,
        image_pool=image_pool_domain_a,
        mode=mode,
        params=params,
        domain_index=1
    )

    if params["forward_cycle_loss_lambda"] > 0.0:
        # Get forward cycle consistency loss.
        forward_cycle_loss = tf.multiply(
            x=get_cycle_consistency_loss(
                source_images=domain_a_images,
                generator1=generator_domain_a2b,
                generator2=generator_domain_b2a,
                params=params,
                domain_index=0
            ),
            y=params["forward_cycle_loss_lambda"],
            name="forward_cycle_loss"
        )
        print_obj(func_name, "forward_cycle_loss", forward_cycle_loss)
    else:
        forward_cycle_loss = 0.0

    if params["backward_cycle_loss_lambda"] > 0.0:
        # Get backward cycle consistency loss.
        backward_cycle_loss = tf.multiply(
            x=get_cycle_consistency_loss(
                source_images=domain_b_images,
                generator1=generator_domain_b2a,
                generator2=generator_domain_a2b,
                params=params,
                domain_index=1
            ),
            y=params["backward_cycle_loss_lambda"],
            name="backward_cycle_loss"
        )
        print_obj(func_name, "backward_cycle_loss", backward_cycle_loss)
    else:
        backward_cycle_loss = 0.0

    # Combine cycle consistency losses together for forward and backward.
    cycle_loss = tf.add(
        x=forward_cycle_loss, y=backward_cycle_loss, name="cycle_loss"
    )
    print_obj(func_name, "cycle_loss", cycle_loss)

    # Add summaries for TensorBoard.
    tf.summary.scalar(
        name="forward_cycle_loss",
        tensor=forward_cycle_loss,
        family="losses"
    )
    tf.summary.scalar(
        name="backward_cycle_loss",
        tensor=backward_cycle_loss,
        family="losses"
    )
    tf.summary.scalar(
        name="cycle_loss",
        tensor=cycle_loss,
        family="losses"
    )

    # Add cycle consistency losses to generator losses.
    generator_domain_a2b_total_loss = tf.add(
        x=generator_domain_a2b_total_loss,
        y=cycle_loss,
        name="generator_domain_a2b_total_loss"
    )
    print_obj(
        func_name,
        "generator_domain_a2b_total_loss",
        generator_domain_a2b_total_loss
    )

    generator_domain_b2a_total_loss = tf.add(
        x=generator_domain_b2a_total_loss,
        y=cycle_loss,
        name="generator_domain_b2a_total_loss"
    )
    print_obj(
        func_name,
        "generator_domain_b2a_total_loss",
        generator_domain_b2a_total_loss
    )

    if params["identity_loss_lambda"] > 0.0:
        # Get identity loss for domain a2b generator.
        identity_loss_domain_a2b = tf.multiply(
            x=get_identity_loss(
                real_target_images=domain_b_images,
                generator=generator_domain_a2b,
                params=params,
                domain_index=1
            ),
            y=params["backward_cycle_loss_lambda"] * params["identity_loss_lambda"],
            name="identity_loss_domain_a2b"
        )
        print_obj(
            func_name, "identity_loss_domain_a2b", identity_loss_domain_a2b
        )

        # Add identity loss to generator domain a2b.
        generator_domain_a2b_total_loss = tf.add(
            x=generator_domain_a2b_total_loss,
            y=identity_loss_domain_a2b,
            name="generator_domain_a2b_total_loss_w_identity"
        )

        # Get identity loss for domain b2a generator.
        identity_loss_domain_b2a = tf.multiply(
            x=get_identity_loss(
                real_target_images=domain_a_images,
                generator=generator_domain_b2a,
                params=params,
                domain_index=0
            ),
            y=params["forward_cycle_loss_lambda"] * params["identity_loss_lambda"],
            name="identity_loss_domain_b2a"
        )
        print_obj(
            func_name, "identity_loss_domain_b2a", identity_loss_domain_b2a
        )

        # Add identity loss to generator domain b2a.
        generator_domain_b2a_total_loss = tf.add(
            x=generator_domain_b2a_total_loss,
            y=identity_loss_domain_b2a,
            name="generator_domain_b2a_total_loss_w_identity"
        )

        # Add summaries for TensorBoard.
        tf.summary.scalar(
            name="identity_loss_domain_a2b",
            tensor=identity_loss_domain_a2b,
            family="losses"
        )
        tf.summary.scalar(
            name="identity_loss_domain_b2a",
            tensor=identity_loss_domain_b2a,
            family="losses"
        )

    # Add summaries for TensorBoard.
    tf.summary.scalar(
        name="generator_domain_a2b_total_loss",
        tensor=generator_domain_a2b_total_loss,
        family="losses_for_minimization"
    )
    tf.summary.scalar(
        name="generator_domain_b2a_total_loss",
        tensor=generator_domain_b2a_total_loss,
        family="losses_for_minimization"
    )
    tf.summary.scalar(
        name="discriminator_domain_a_total_loss",
        tensor=discriminator_domain_a_total_loss,
        family="losses_for_minimization"
    )
    tf.summary.scalar(
        name="discriminator_domain_b_total_loss",
        tensor=discriminator_domain_b_total_loss,
        family="losses_for_minimization"
    )

    return (real_logits_domain_b,
            fake_logits_domain_b,
            generator_domain_a2b_total_loss,
            discriminator_domain_b_total_loss,
            real_logits_domain_a,
            fake_logits_domain_a,
            generator_domain_b2a_total_loss,
            discriminator_domain_a_total_loss
            )
