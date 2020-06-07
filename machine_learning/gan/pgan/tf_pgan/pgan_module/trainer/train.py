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

    # Update alpha variable to linearly scale from 0 to 1 based on steps.
    alpha_var_update_op = tf.assign(
        ref=alpha_var,
        value=tf.divide(
            x=tf.cast(
                x=tf.mod(x=global_step, y=params["num_steps_until_growth"]),
                dtype=tf.float32
            ),
            y=params["num_steps_until_growth"]
        ),
        name="{}_alpha_var_update_op".format(scope)
    )
    print_obj("train_network", "alpha_var_update_op", alpha_var_update_op)

    # Ensure alpha variable gets updated.
    with tf.control_dependencies(control_inputs=[alpha_var_update_op]):
        loss = tf.identity(
            input=loss, name="{}_train_network_loss_identity".format(scope)
        )

    return loss, train_op


def get_loss_and_train_op(
        generator_total_loss, discriminator_total_loss, alpha_var, params):
    """Gets loss and train op for train mode.

    Args:
        generator_total_loss: tensor, scalar total loss of generator.
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
        x=cycle_step, y=params["generator_train_steps"]
    )

    # Needed for batch normalization, but has no effect otherwise.
    update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(control_inputs=update_ops):
        # Conditionally choose to train generator or discriminator.
        loss, train_op = tf.cond(
            pred=condition,
            true_fn=lambda: train_network(
                loss=generator_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                scope="generator"
            ),
            false_fn=lambda: train_network(
                loss=discriminator_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                scope="discriminator"
            ),
            name="train_network_cond"
        )

    return loss, train_op
