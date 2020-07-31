import tensorflow as tf

from . import train_and_eval


def get_variables_and_gradients(
    loss,
    network,
    gradient_tape,
    params,
    scope
):
    """Gets variables and gradients from model wrt. loss.

    Args:
        loss: tensor, shape of [].
        network: instance of network; either `Generator` or `Discriminator`.
        gradient_tape: instance of `GradientTape`.
        params: dict, user passed parameters.
        scope: str, the name of the network of interest.

    Returns:
        Lists of network's variables and gradients.
    """
    # Get trainable variables.
    variables = network.get_model().trainable_variables

    # Get gradients from gradient tape.
    gradients = gradient_tape.gradient(
        target=loss, sources=variables
    )

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )

    # Add variable names back in for identification.
    gradients = [
        tf.identity(
            input=g,
            name="{}_{}_gradients".format(scope, v.name[:-2])
        )
        if tf.is_tensor(x=g) else g
        for g, v in zip(gradients, variables)
    ]

    return variables, gradients


def get_generator_loss_variables_and_gradients(
    global_batch_size,
    generator,
    discriminator,
    global_step,
    summary_file_writer,
    params
):
    """Gets generator's loss, variables, and gradients.

    Args:
        global_batch_size: int, global batch size for distribution.
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Generator's loss, variables, and gradients.
    """
    with tf.GradientTape() as generator_tape:
        # Get generator loss.
        _, generator_loss = train_and_eval.generator_loss_phase(
            global_batch_size,
            generator,
            discriminator,
            params,
            global_step,
            summary_file_writer,
            mode="TRAIN",
            training=True
        )

    # Get variables and gradients from generator wrt. loss.
    variables, gradients = get_variables_and_gradients(
        loss=generator_loss,
        network=generator,
        gradient_tape=generator_tape,
        params=params,
        scope="generator"
    )

    return generator_loss, variables, gradients


def get_discriminator_loss_variables_and_gradients(
    global_batch_size,
    real_images,
    generator,
    discriminator,
    global_step,
    summary_file_writer,
    params
):
    """Gets discriminator's loss, variables, and gradients.

    Args:
        global_batch_size: int, global batch size for distribution.
        real_images: tensor, real images of shape
            [batch_size, height * width * depth].
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Discriminator's loss, variables, and gradients.
    """
    with tf.GradientTape() as discriminator_tape:
        # Get fake logits from generator.
        fake_logits, _ = train_and_eval.generator_loss_phase(
            global_batch_size,
            generator,
            discriminator,
            params,
            global_step,
            summary_file_writer,
            mode="TRAIN",
            training=True
        )

        # Get discriminator loss.
        _, discriminator_loss = train_and_eval.discriminator_loss_phase(
            global_batch_size,
            discriminator,
            real_images,
            fake_logits,
            params,
            global_step,
            summary_file_writer,
            mode="TRAIN",
            training=True
        )

    # Get variables and gradients from discriminator wrt. loss.
    variables, gradients = get_variables_and_gradients(
        loss=discriminator_loss,
        network=discriminator,
        gradient_tape=discriminator_tape,
        params=params,
        scope="discriminator"
    )

    return discriminator_loss, variables, gradients


def create_variable_and_gradient_histogram_summaries(
    variables,
    gradients,
    params,
    global_step,
    summary_file_writer,
    scope
):
    """Creates variable and gradient histogram summaries.

    Args:
        variables: list, network's trainable variables.
        gradients: list, gradients of networks trainable variables wrt. loss.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.
        scope: str, the name of the network of interest.
    """
    # Add summaries for TensorBoard.
    with summary_file_writer.as_default():
        with tf.summary.record_if(
            global_step % params["save_summary_steps"] == 0
        ):
            for v, g in zip(variables, gradients):
                tf.summary.histogram(
                    name="{}_variables/{}".format(scope, v.name[:-2]),
                    data=v,
                    step=global_step
                )
                if tf.is_tensor(x=g):
                    tf.summary.histogram(
                        name="{}_gradients/{}".format(scope, v.name[:-2]),
                        data=g,
                        step=global_step
                    )
            summary_file_writer.flush()


def get_select_loss_variables_and_gradients(
    global_batch_size,
    real_images,
    generator,
    discriminator,
    global_step,
    summary_file_writer,
    params,
    scope
):
    """Gets selected network's loss, variables, and gradients.

    Args:
        global_batch_size: int, global batch size for distribution.
        real_images: tensor, real images of shape
            [batch_size, height * width * depth].
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.
        scope: str, the name of the network of interest.

    Returns:
        Selected network's loss, variables, and gradients.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        # Get fake logits from generator.
        fake_logits, generator_loss = train_and_eval.generator_loss_phase(
            global_batch_size,
            generator,
            discriminator,
            params,
            global_step,
            summary_file_writer,
            mode="TRAIN",
            training=True
        )

        # Get discriminator loss.
        _, discriminator_loss = train_and_eval.discriminator_loss_phase(
            global_batch_size,
            discriminator,
            real_images,
            fake_logits,
            params,
            global_step,
            summary_file_writer,
            mode="TRAIN",
            training=True
        )

    # Create empty dicts to hold loss, variables, gradients.
    loss_dict = {}
    vars_dict = {}
    grads_dict = {}

    # Loop over generator and discriminator.
    for (loss, network, gradient_tape, scope) in zip(
        [generator_loss, discriminator_loss],
        [generator, discriminator],
        [gen_tape, dis_tape],
        ["generator", "discriminator"]
    ):
        # Get variables and gradients from generator wrt. loss.
        variables, gradients = get_variables_and_gradients(
            loss, network, gradient_tape, params, scope
        )

        # Add loss, variables, and gradients to dictionaries.
        loss_dict[scope] = loss
        vars_dict[scope] = variables
        grads_dict[scope] = gradients

        # Create variable and gradient histogram summaries.
        create_variable_and_gradient_histogram_summaries(
            variables,
            gradients,
            params,
            global_step,
            summary_file_writer,
            scope
        )

    return loss_dict[scope], vars_dict[scope], grads_dict[scope]


def train_network(variables, gradients, optimizer):
    """Trains network variables using gradients with optimizer.

    Args:
        variables: list, network's trainable variables.
        gradients: list, gradients of networks trainable variables wrt. loss.
        optimizer: instance of `Optimizer`.
    """
    # Zip together gradients and variables.
    grads_and_vars = zip(gradients, variables)

    # Applying gradients to variables using optimizer.
    optimizer.apply_gradients(grads_and_vars=grads_and_vars)


def train_discriminator(
    global_batch_size,
    features,
    generator,
    discriminator,
    discriminator_optimizer,
    params,
    global_step,
    summary_file_writer
):
    """Trains discriminator network.

    Args:
        global_batch_size: int, global batch size for distribution.
        features: dict, feature tensors from input function.
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        discriminator_optimizer: instance of `Optimizer`, discriminator's
            optimizer.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Discriminator loss tensor.
    """
    # Extract real images from features dictionary.
    real_images = tf.reshape(
        tensor=features["image"],
        shape=[-1, params["height"] * params["width"] * params["depth"]]
    )

    # Get gradients for training by running inputs through networks.
    if global_step % params["save_summary_steps"] == 0:
        # More computation, but needed for ALL histogram summaries.
        loss, variables, gradients = (
            get_select_loss_variables_and_gradients(
                global_batch_size,
                real_images,
                generator,
                discriminator,
                global_step,
                summary_file_writer,
                params,
                scope="discriminator"
            )
        )
    else:
        # More efficient computation.
        loss, variables, gradients = (
            get_discriminator_loss_variables_and_gradients(
                global_batch_size,
                real_images,
                generator,
                discriminator,
                global_step,
                summary_file_writer,
                params
            )
        )

    # Train discriminator network.
    train_network(variables, gradients, optimizer=discriminator_optimizer)

    return loss


def train_generator(
    global_batch_size,
    features,
    generator,
    discriminator,
    generator_optimizer,
    params,
    global_step,
    summary_file_writer
):
    """Trains generator network.

    Args:
        global_batch_size: int, global batch size for distribution.
        features: dict, feature tensors from input function.
        generator: instance of `Generator`.
        discriminator: instance of `Discriminator`.
        generator_optimizer: instance of `Optimizer`, generator's
            optimizer.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Generator loss tensor.
    """
    # Get gradients for training by running inputs through networks.
    if global_step % params["save_summary_steps"] == 0:
        # Extract real images from features dictionary.
        real_images = tf.reshape(
            tensor=features["image"],
            shape=[-1, params["height"] * params["width"] * params["depth"]]
        )

        # More computation, but needed for ALL histogram summaries.
        loss, variables, gradients = (
            get_select_loss_variables_and_gradients(
                global_batch_size,
                real_images,
                generator,
                discriminator,
                global_step,
                summary_file_writer,
                params,
                scope="generator"
            )
        )
    else:
        # More efficient computation.
        loss, variables, gradients = (
            get_generator_loss_variables_and_gradients(
                global_batch_size,
                generator,
                discriminator,
                global_step,
                summary_file_writer,
                params
            )
        )

    # Train generator network.
    train_network(variables, gradients, optimizer=generator_optimizer)

    return loss


def train_step(
    global_batch_size,
    features,
    network_dict,
    optimizer_dict,
    params,
    global_step,
    summary_file_writer
):
    """Perform one train step.

    Args:
        global_batch_size: int, global batch size for distribution.
        features: dict, feature tensors from input function.
        network_dict: dict, dictionary of network objects.
        optimizer_dict: dict, dictionary of optimizer objects.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Loss tensor for chosen network.
    """
    # Determine if it is time to train generator or discriminator.
    cycle_step = global_step % (
        params["discriminator_train_steps"] + params["generator_train_steps"]
    )

    # Conditionally choose to train generator or discriminator subgraph.
    if cycle_step < params["discriminator_train_steps"]:
        loss = train_discriminator(
            global_batch_size=global_batch_size,
            features=features,
            generator=network_dict["generator"],
            discriminator=network_dict["discriminator"],
            discriminator_optimizer=optimizer_dict["discriminator"],
            params=params,
            global_step=global_step,
            summary_file_writer=summary_file_writer
        )
    else:
        loss = train_generator(
            global_batch_size=global_batch_size,
            features=features,
            generator=network_dict["generator"],
            discriminator=network_dict["discriminator"],
            generator_optimizer=optimizer_dict["generator"],
            params=params,
            global_step=global_step,
            summary_file_writer=summary_file_writer
        )

    return loss
