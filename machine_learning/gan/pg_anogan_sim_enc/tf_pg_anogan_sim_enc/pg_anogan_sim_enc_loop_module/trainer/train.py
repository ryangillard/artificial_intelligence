import tensorflow as tf

from .print_object import print_obj


def train_network(loss, global_step, alpha_var, params, scope):
    """Trains network and returns loss and train op.

    Args:
        loss: tensor, shape of [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        scope: str, the network's name to find its variables to train.

    Returns:
        Loss tensor and training op.
    """
    print_obj("\ntrain_network", "loss", loss)
    print_obj("train_network", "global_step", global_step)
    print_obj("train_network", "alpha_var", alpha_var)
    print_obj("train_network", "scope", scope)

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
    optimizer = optimizers[params["{}_optimizer".format(scope)]](
        learning_rate=params["{}_learning_rate".format(scope)]
    )
    print_obj("train_network", "optimizer", optimizer)

    # If using TPU, wrap optimizer to use an allreduce to aggregate gradients
    # and broadcast the result to each shard.
    if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(opt=optimizer)

    # Get trainable variables.
    variables = tf.trainable_variables(scope=scope)
    print_obj("\ntrain_network", "variables", variables)

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=variables,
        name="{}_gradients".format(scope)
    )
    print_obj("\ntrain_network", "gradients", gradients)

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )
        print_obj("\ntrain_network", "gradients", gradients)

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, variables)
    print_obj("train_network", "grads_and_vars", grads_and_vars)

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )
    print_obj("train_network", "train_op", train_op)

    # If never grow, then no need to update alpha since it is not used.
    if len(params["conv_num_filters"]) > 1:
        # Don't want to double count when generator also increases alpha.
        if scope != "encoder":
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
                name="{}_alpha_var_update_op".format(scope)
            )
            print_obj(
                "train_network", "alpha_var_update_op", alpha_var_update_op
            )

            # Ensure alpha variable gets updated.
            with tf.control_dependencies(control_inputs=[alpha_var_update_op]):
                loss = tf.identity(
                    input=loss,
                    name="{}_train_network_loss_identity".format(scope)
                )

    return loss, train_op


def jointly_train_generator_encoder(
        generator_loss,
        encoder_loss,
        global_step,
        alpha_var,
        params,
        generator_scope,
        encoder_scope):
    """Trains network and returns loss and train op.

    Args:
        generator_loss: tensor, generator's loss with shape [].
        encoder_loss: tensor, encoder's loss with shape [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        generator_scope: str, the generator's name to find its variables.
        encoder_scope: str, the generator's name to find its variables.

    Returns:
        Loss tensor and training op.
    """
    # Get loss and train_op for generator.
    generator_loss, generator_train_op = train_network(
        loss=generator_loss,
        global_step=global_step,
        alpha_var=alpha_var,
        params=params,
        scope=generator_scope
    )

    # Get loss and train_op for encoder.
    encoder_loss, encoder_train_op = train_network(
        loss=encoder_loss,
        global_step=global_step,
        alpha_var=None,
        params=params,
        scope=encoder_scope
    )

    # Add generator and encoder losses together.
    loss = tf.add(
        x=generator_loss,
        y=encoder_loss,
        name="jointly_train_generator_encoder_add_loss"
    )
    print_obj("\njointly_train_generator_encoder", "loss", loss)

    # Group train_ops together.
    train_op = tf.group(
        generator_train_op,
        encoder_train_op,
        name="jointly_train_generator_encoder_group_train_op"
    )
    print_obj("jointly_train_generator_encoder", "train_op", train_op)

    return loss, train_op


def get_loss_and_train_op(
        generator_total_loss,
        discriminator_total_loss,
        encoder_total_loss,
        alpha_var,
        params):
    """Gets loss and train op for train mode.

    Args:
        generator_total_loss: tensor, scalar total loss of generator.
        discriminator_total_loss: tensor, scalar total loss of discriminator.
        encoder_total_loss: tensor, scalar total loss of encoder.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.

    Returns:
        Loss scalar tensor and train_op to be used by the EstimatorSpec.
    """
    # Get global step.
    global_step = tf.train.get_or_create_global_step()

    # Needed for batch normalization, but has no effect otherwise.
    update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(control_inputs=update_ops):
        # Conditionally choose to train generator or discriminator.
        if params["training_phase"] == "generator":
            loss, train_op = jointly_train_generator_encoder(
                generator_loss=generator_total_loss,
                encoder_loss=encoder_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                generator_scope="generator",
                encoder_scope="encoder"
            )
        else:
            loss, train_op = train_network(
                loss=discriminator_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                scope="discriminator"
            )

    return loss, train_op
