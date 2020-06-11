import tensorflow as tf

from .print_object import print_obj


def get_gradients(loss, global_step, params, scope):
    """Returns the gradients and variables of the current training step.

    Args:
        loss: tensor, shape of [].
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        scope: str, the network's name to find its variables to train.

    Returns:
        Gradient tensors.
    """
    print_obj("\nget_gradients", "loss", loss)
    print_obj("get_gradients", "global_step", global_step)
    print_obj("get_gradients", "scope", scope)

    # Get trainable variables.
    variables = tf.trainable_variables(scope=scope)
    print_obj("\nget_gradients", "variables", variables)

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=variables,
        unconnected_gradients="zero",
        name="{}_gradients".format(scope)
    )
    print_obj("\nget_gradients", "gradients", gradients)

    # Add variable names back in for identification.
    gradients = [
        tf.identity(
            input=g,
            name="{}_get_gradients_gradients".format(v.name[:-2])
        )
        for g, v in zip(gradients, variables)
    ]
    print_obj("\nget_gradients", "gradients", gradients)

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )
        print_obj("\nget_gradients", "gradients", gradients)

        # Add variable names back in for identification.
        gradients = [
            tf.identity(
                input=g,
                name="{}_get_gradients_clip_gradients".format(v.name[:-2])
            )
            for g, v in zip(gradients, variables)
        ]
        print_obj("\nget_gradients", "gradients", gradients)

    return gradients


def jointly_train_generator_encoder(
        generator_loss,
        encoder_loss,
        global_step,
        params,
        generator_scope,
        encoder_scope,
        discriminator_scope):
    """Returns generator's/encoder's combined objects needed for training.

    Args:
        generator_loss: tensor, generator's loss with shape [].
        encoder_loss: tensor, encoder's loss with shape [].
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        generator_scope: str, the generator's name to find its variables.
        encoder_scope: str, the encoder's name to find its variables.
        discriminator_scope: str, the discriminator's name to find its
            variables.

    Returns:
        Loss tensor anddict of gradient tensors.
    """
    # Add generator and encoder losses together.
    loss = tf.add(
        x=generator_loss,
        y=encoder_loss,
        name="jointly_train_generator_encoder_add_loss"
    )
    print_obj("\njointly_train_generator_encoder", "loss", loss)

    # Get generator gradients.
    generator_gradients = get_gradients(
        loss=generator_loss,
        global_step=global_step,
        params=params,
        scope=generator_scope
    )
    print_obj(
        "\njointly_train_generator_encoder",
        "generator_gradients",
        generator_gradients
    )

    # Get encoder gradients.
    encoder_gradients = get_gradients(
        loss=encoder_loss,
        global_step=global_step,
        params=params,
        scope=encoder_scope
    )
    print_obj(
        "\njointly_train_generator_encoder",
        "encoder_gradients",
        encoder_gradients
    )

    # Get discriminator variables and set gradients to zero.
    discriminator_variables = tf.trainable_variables(scope="discriminator")
    discriminator_gradients = [
        tf.zeros_like(
            tensor=v,
            dtype=tf.float32,
            name="{}_jointly_train_generator_encoder_zeros_like".format(
                v.name[:-2]
            )
        )
        for v in discriminator_variables
    ]

    # Combine gradients into a dictionary.
    gradients = {
        generator_scope: generator_gradients,
        encoder_scope: encoder_gradients,
        discriminator_scope: discriminator_gradients
    }
    print_obj("\njointly_train_generator_encoder", "gradients", gradients)
    print(
        "\njointly_train_generator_encoder: gradients = {}".format(gradients)
    )

    return loss, gradients


def train_discriminator(
        discriminator_loss,
        global_step,
        params,
        generator_scope,
        encoder_scope,
        discriminator_scope):
    """Returns discriminator's objects needed for training.

    Args:
        discriminator_loss: tensor, discriminator's loss with shape [].
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        generator_scope: str, the generator's name to find its variables.
        encoder_scope: str, the encoder's name to find its variables.
        discriminator_scope: str, the discriminator's name to find its
            variables.

    Returns:
        Loss tensor and dict of gradient tensors.
    """
    # The loss is just the discriminator loss.
    loss = discriminator_loss

    # Get generator variables and set gradients to zero.
    generator_variables = tf.trainable_variables(scope=generator_scope)
    generator_gradients = [
        tf.zeros_like(
            tensor=v,
            dtype=tf.float32,
            name="{}_train_discriminator_zeros_like".format(v.name[:-2])
        )
        for v in generator_variables
    ]

    # Get encoder variables and set gradients to zero.
    encoder_variables = tf.trainable_variables(scope=encoder_scope)
    encoder_gradients = [
        tf.zeros_like(
            tensor=v,
            dtype=tf.float32,
            name="{}_train_discriminator_zeros_like".format(v.name[:-2])
        )
        for v in encoder_variables
    ]

    # Get discriminator gradients.
    discriminator_gradients = get_gradients(
        loss=discriminator_loss,
        global_step=global_step,
        params=params,
        scope=discriminator_scope
    )
    print_obj(
        "\ntrain_discriminator",
        "discriminator_gradients",
        discriminator_gradients
    )

    # Combine gradients into a dictionary.
    gradients = {
        generator_scope: generator_gradients,
        encoder_scope: encoder_gradients,
        discriminator_scope: discriminator_gradients
    }
    print_obj("\ntrain_discriminator", "gradients", gradients)
    print("\ntrain_discriminator: gradients = {}".format(gradients))
    tf.logging.info("\ntrain_discriminator: gradients = {}".format(gradients))

    return loss, gradients


def get_optimizer(params, scope):
    """Returns instance of chosen `Optimizer` class.

    Args:
        params: dict, user passed parameters.
        scope: str, the current network's scope.

    Returns:
        Instance of chosen `Optimizer` class.
    """
    # Create optimizer map.
    optimizers = {
        "Adam": tf.train.AdamOptimizer,
        "Adadelta": tf.train.AdadeltaOptimizer,
        "AdagradDA": tf.train.AdagradDAOptimizer,
        "Adagrad": tf.train.AdagradOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "GradientDescent": tf.train.GradientDescentOptimizer,
        "Momentum": tf.train.MomentumOptimizer,
        "ProximalAdagrad": tf.train.ProximalAdagradOptimizer,
        "ProximalGradientDescent": tf.train.ProximalGradientDescentOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer
    }

    # Get optimizer and instantiate it.
    optimizer_name = params["{}_optimizer".format(scope)]
    learning_rate = params["{}_learning_rate".format(scope)]

    optimizer = optimizers[optimizer_name](learning_rate=learning_rate)
    print_obj("\nget_optimizer", "optimizer", optimizer)

    # If using TPU, wrap optimizer to use an allreduce to aggregate gradients
    # and broadcast the result to each shard.
    if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(opt=optimizer)
        print_obj("get_optimizer", "optimizer", optimizer)

    return optimizer


def get_train_op(gradients, global_step, params, scope):
    """Returns train op of applying gradients with optimizer to variavles.

    Args:
        gradients: list, gradient tensors for in scope trainable variables.
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        scope: str, the network's name to find its variables to train.

    Returns:
        Training op.
    """
    print_obj("\nget_train_op", "gradients", gradients)
    print_obj("\nget_train_op", "global_step", global_step)
    print_obj("get_train_op", "scope", scope)

    # Get trainables variables from scope.
    variables = tf.trainable_variables(scope=scope)

    # Zip together gradients and variables.
    grads_and_vars = zip(gradients, variables)
    print_obj("\nget_train_op", "grads_and_vars", grads_and_vars)

    # Get optimizers.
    optimizer = get_optimizer(params=params, scope=scope)
    print_obj("get_train_op", "optimizer", optimizer)

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )
    print_obj("get_train_op", "train_op", train_op)

    return train_op


def get_train_ops(gradients, global_step, params):
    """Returns train op of applying gradients with optimizer to variavles.

    Args:
        gradients: list, gradient tensors for in scope trainable variables.
        optimizer: instance of `Optimizer` class.
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.

    Returns:
        Training op.
    """
    print_obj("\nget_train_ops", "gradients", gradients)
    print_obj("\nget_train_ops", "global_step", global_step)

    # Create dict for global step so that apply gradients only increments it
    # once since apply gradients will be called three times every step.
    global_step_dict = {
        "generator": None, "encoder": None, "discriminator": global_step
    }

    # Create list of train ops.
    train_ops = [
        get_train_op(
            gradients=g,
            global_step=global_step_dict[s],
            params=params,
            scope=s
        )
        for s, g in gradients.items()
    ]
    print_obj("\nget_train_op", "train_ops", train_ops)

    # Group together train ops.
    train_op = tf.group(
        train_ops,
        name="jointly_train_generator_encoder_group_train_op"
    )
    print_obj("\nget_train_ops", "train_op", train_op)

    return train_op


def update_alpha(global_step, alpha_var, params):
    """Returns update op for alpha variable.

    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Alpha variable update operation.
    """
    # If never grow, then no need to update alpha since it is not used.
    if len(params["conv_num_filters"]) > 1:
        # Update alpha var to linearly scale from 0 to 1 based on steps.
        alpha_var_update_op = tf.assign(
            ref=alpha_var,
            value=tf.divide(
                x=tf.cast(
                    x=tf.mod(
                        x=global_step, y=params["num_steps_until_growth"]
                    ),
                    dtype=tf.float32
                ),
                y=params["num_steps_until_growth"]
            ),
            name="alpha_var_update_op_assign"
        )
    else:
        alpha_var_update_op = tf.no_op(name="alpha_var_update_op_no_op")
    print_obj(
        "update_alpha", "alpha_var_update_op", alpha_var_update_op
    )

    return alpha_var_update_op


def get_loss_and_train_op(
        generator_total_loss,
        encoder_total_loss,
        discriminator_total_loss,
        alpha_var,
        params):
    """Returns loss and train op for train mode.

    Args:
        generator_total_loss: tensor, scalar total loss of generator.
        encoder_total_loss: tensor, scalar total loss of encoder.
        discriminator_total_loss: tensor, scalar total loss of discriminator.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Loss scalar tensor and train_op to be used by the EstimatorSpec.
    """
    # Get global step.
    global_step = tf.train.get_or_create_global_step()

    # Determine if it is time to train generator or discriminator.
    cycle_step = tf.mod(
        x=global_step,
        y=tf.cast(
            x=tf.add(
                x=params["generator_train_steps"],
                y=params["discriminator_train_steps"]
            ),
            dtype=tf.int64
        ),
        name="get_loss_and_train_op_cycle_step"
    )

    # Create choose generator condition.
    condition = tf.less(
        x=cycle_step,
        y=params["generator_train_steps"],
        name="get_loss_and_train_op_condition"
    )

    # Needed for batch normalization, but has no effect otherwise.
    update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(control_inputs=update_ops):
        # Conditionally choose to train generator/encoder or discriminator.
        loss, gradients = tf.cond(
            pred=condition,
            false_fn=lambda: jointly_train_generator_encoder(
                generator_loss=generator_total_loss,
                encoder_loss=encoder_total_loss,
                global_step=global_step,
                params=params,
                generator_scope="generator",
                encoder_scope="encoder",
                discriminator_scope="discriminator"
            ),
            true_fn=lambda: train_discriminator(
                discriminator_loss=discriminator_total_loss,
                global_step=global_step,
                params=params,
                generator_scope="generator",
                encoder_scope="encoder",
                discriminator_scope="discriminator"
            ),
            name="get_loss_and_train_op_cond"
        )
        print_obj("\nget_loss_and_train_op", "gradients", gradients)

        # Crete train_op with whatever was returned from conditional branch.
        train_op = get_train_ops(gradients, global_step, params)

        # Get update op for the alpha variable.
        alpha_var_update_op = update_alpha(global_step, alpha_var, params)

        # Ensure alpha variable gets updated.
        with tf.control_dependencies(control_inputs=[alpha_var_update_op]):
            loss = tf.identity(
                input=loss,
                name="get_loss_and_train_op_loss_identity"
            )

    return loss, train_op
