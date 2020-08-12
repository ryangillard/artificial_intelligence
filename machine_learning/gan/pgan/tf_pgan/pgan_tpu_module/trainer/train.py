import tensorflow as tf

from .print_object import print_obj


def get_variables_and_gradients(loss, scope):
    """Gets variables and their gradients wrt. loss.
    Args:
        loss: tensor, shape of [].
        scope: str, the network's name to find its variables to train.
    Returns:
        Lists of variables and their gradients.
    """
    func_name = "get_variables_and_gradients"
    # Get trainable variables.
    variables = tf.trainable_variables(scope=scope)
    print_obj("\n{}_{}".format(func_name, scope), "variables", variables)

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=variables,
        name="{}_gradients".format(scope)
    )
    print_obj("\n{}_{}".format(func_name, scope), "gradients", gradients)

    # Add variable names back in for identification.
    gradients = [
        tf.identity(
            input=g,
            name="{}_{}_gradients".format(func_name, v.name[:-2])
        )
        if tf.is_tensor(x=g) else g
        for g, v in zip(gradients, variables)
    ]
    print_obj("\n{}_{}".format(func_name, scope), "gradients", gradients)

    return variables, gradients


def create_variable_and_gradient_histogram_summaries(loss_dict, params):
    """Creates variable and gradient histogram summaries.
    Args:
        loss_dict: dict, keys are scopes and values are scalar loss tensors
            for each network kind.
        params: dict, user passed parameters.
    """
    if not params["use_tpu"]:
        for scope, loss in loss_dict.items():
            # Get variables and their gradients wrt. loss.
            variables, gradients = get_variables_and_gradients(loss, scope)

            # Add summaries for TensorBoard.
            for g, v in zip(gradients, variables):
                tf.summary.histogram(
                    name="{}".format(v.name[:-2]),
                    values=v,
                    family="{}_variables".format(scope)
                )
                if tf.is_tensor(x=g):
                    tf.summary.histogram(
                        name="{}".format(v.name[:-2]),
                        values=g,
                        family="{}_gradients".format(scope)
                    )


def instantiate_optimizer_slots(optimizer, variables, params, scope):
    """Instantiates optimizer slots for all parameters ahead of time.
    Args:
        optimizer: instance of `Optimizer`.
        variables: list, list of scoped trainable variables.
        params: dict, user passed parameters.
        scope: str, the network's name to find its variables to train.
    Returns:
        Apply gradients op to instantiate all optimizer slots and add to
            collection op for optimizer slot metric variables.
    """
    func_name = "instantiate_optimizer_slots"
    # Create zero gradients for every scoped trainable variable.
    zero_gradients = [
        tf.zeros_like(
            tensor=v,
            dtype=tf.float32,
            name="{}_{}_{}_zeros_like".format(func_name, scope, v.name[:-2])
        )
        for v in variables
    ]
    print_obj(
        "{}_{}".format(func_name, scope), "zero_gradients", zero_gradients
    )

    # Zip together gradients and variables.
    grads_and_vars = zip(zero_gradients, variables)
    print_obj(
        "{}_{}".format(func_name, scope), "grads_and_vars", grads_and_vars
    )

    # Apply zero gradients to create all optimizer slots ahead of time. Since
    # this is when global_step is zero, it won't change the parameters or the
    # moment accumulators.
    instantiate_optimizer_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=None,
        name="{}_{}_apply_gradients".format(func_name, scope)
    )
    print_obj(
        "{}_{}".format(func_name, scope),
        "instantiate_optimizer_op",
        instantiate_optimizer_op
    )

    if params["save_optimizer_metrics_to_checkpoint"]:
        optimizer_name = "{}_{}_optimizer".format(
            scope, params["{}_optimizer".format(scope)]
        )
        # Add optimizer slot metric variables to global collection so that they
        # will be written to checkpoints.
        add_to_collection_ops = [
            tf.add_to_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, value=v)
            for v in tf.get_collection(
                key=tf.GraphKeys.METRIC_VARIABLES, scope=optimizer_name
            )
        ]
    else:
        add_to_collection_ops = []
    print_obj(
        "{}_{}".format(func_name, scope),
        "add_to_collection_ops",
        add_to_collection_ops
    )

    return instantiate_optimizer_op, add_to_collection_ops


def dont_instantiate_optimizer_slots(scope):
    """Wrapper for not instantiating optimizer slots for tf.cond.
    Args:
        scope: str, the network's name to find its variables to train.
    Returns:
        Apply gradients no op to instantiate all optimizer slots and add to
            collection no op for optimizer slot metric variables.
    """
    instantiate_optimizer_no_op = tf.no_op(
        name="{}_instantiate_optimizer_no_op".format(scope)
    )

    return instantiate_optimizer_no_op, []


def train_network(
        loss, global_step, alpha_var, params, scope, increment_global_step):
    """Trains network and returns loss and train op.
    Args:
        loss: tensor, shape of [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        scope: str, the network's name to find its variables to train.
        increment_global_step: int, whether to increment global step or not.
    Returns:
        Loss tensor and training op.
    """
    func_name = "train_network"
    print_obj("\n" + func_name, "loss", loss)
    print_obj(func_name, "global_step", global_step)
    print_obj(func_name, "alpha_var", alpha_var)
    print_obj(func_name, "scope", scope)

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
    if params["{}_optimizer".format(scope)] == "Adam":
        optimizer = optimizers[params["{}_optimizer".format(scope)]](
            learning_rate=params["{}_learning_rate".format(scope)],
            beta1=params["{}_adam_beta1".format(scope)],
            beta2=params["{}_adam_beta2".format(scope)],
            epsilon=params["{}_adam_epsilon".format(scope)],
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(scope)].lower()
            )
        )
    else:
        optimizer = optimizers[params["{}_optimizer".format(scope)]](
            learning_rate=params["{}_learning_rate".format(scope)],
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(scope)].lower()
            )
        )
    print_obj("{}_{}".format(func_name, scope), "optimizer", optimizer)

    # If using TPU, wrap optimizer to use an allreduce to aggregate gradients
    # and broadcast the result to each shard.
    if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(opt=optimizer)
        print_obj("{}_{}".format(func_name, scope), "optimizer", optimizer)

    # Get variables and their gradients wrt. loss.
    variables, gradients = get_variables_and_gradients(loss, scope)

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )
        print_obj("\n{}_{}".format(func_name, scope), "gradients", gradients)

        # Add variable names back in for identification.
        gradients = [
            tf.identity(
                input=g,
                name="{}_{}_clip_gradients".format(func_name, v.name[:-2])
            )
            if tf.is_tensor(x=g) else g
            for g, v in zip(gradients, variables)
        ]
        print_obj("\n{}_{}".format(func_name, scope), "gradients", gradients)

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, variables)
    print_obj(
        "{}_{}".format(func_name, scope), "grads_and_vars", grads_and_vars
    )

    if params["{}_optimizer".format(scope)] != "GradientDescent":
        # Instantiate ALL optimizer slots, not just for ones without None grad.
        instantiate_optimizer_op, add_to_collection_ops = tf.cond(
            pred=tf.equal(
                x=global_step, y=0, name="instantiate_optimizer_op_pred"
            ),
            true_fn=lambda: instantiate_optimizer_slots(
                optimizer=optimizer,
                variables=variables,
                params=params,
                scope=scope
            ),
            false_fn=lambda: dont_instantiate_optimizer_slots(scope),
            name="instantiate_optimizer_op_cond"
        )

        with tf.control_dependencies(
                control_inputs=[instantiate_optimizer_op]):
            with tf.control_dependencies(
                    control_inputs=add_to_collection_ops):
                loss = tf.identity(
                    input=loss,
                    name="{}_{}_loss_identity".format(func_name, scope)
                )

    # Create train op by applying gradients to variables and possibly
    # incrementing global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step if increment_global_step else None,
        name="{}_apply_gradients".format(scope)
    )
    print_obj("{}_{}".format(func_name, scope), "train_op", train_op)

    return loss, train_op


def train_discriminator(
        discriminator_loss,
        global_step,
        alpha_var,
        params,
        discriminator_scope):
    """Wrapper that trains discriminator network & returns loss and train op.
    Args:
        discriminator_loss: tensor, discriminator's loss with shape [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        discriminator_scope: str, the discriminator's name to find its
            variables.
    Returns:
        Loss tensor and training op.
    """
    # Get loss and train_op for discriminator.
    loss, train_op = train_network(
        loss=discriminator_loss,
        global_step=global_step,
        alpha_var=alpha_var,
        params=params,
        scope=discriminator_scope,
        increment_global_step=True
    )

    return loss, train_op


def train_generator(
        generator_loss,
        global_step,
        alpha_var,
        params,
        generator_scope):
    """Wrapper that trains generator network & returns loss and train op.
    Args:
        generator_loss: tensor, generator's loss with shape [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        generator_scope: str, the generator's name to find its variables.
    Returns:
        Loss tensor and training op.
    """
    # Get loss and train_op for generator.
    loss, train_op = train_network(
        loss=generator_loss,
        global_step=global_step,
        alpha_var=alpha_var,
        params=params,
        scope=generator_scope,
        increment_global_step=True
    )

    return loss, train_op


def known_update_alpha(global_step, alpha_var, params):
    """Returns ref for updated alpha variable.
    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Ref for updated alpha variable.
    """
    func_name = "known_update_alpha"
    # If never grow, then no need to update alpha since it is not used.
    if len(params["conv_num_filters"]) > 1 and params["growth_idx"] > 0:
        if params["growth_idx"] % 2 == 1:
            # Update alpha var to linearly scale from 0 to 1 based on steps.
            alpha_var = tf.assign(
                ref=alpha_var,
                value=tf.divide(
                    x=tf.cast(
                        # Add 1 since it trains on global step 0, so off by 1.
                        x=tf.add(
                            x=tf.mod(
                                x=tf.subtract(
                                    x=global_step,
                                    y=params["previous_train_steps"]
                                ),
                                y=params["num_steps_until_growth"]
                            ),
                            y=1
                        ),
                        dtype=tf.float32
                    ),
                    y=params["num_steps_until_growth"]
                ),
                name="update_alpha_assign_linear"
            )
        else:
            alpha_var = tf.assign(
                ref=alpha_var,
                value=tf.ones(shape=[], dtype=tf.float32),
                name="update_alpha_assign_ones"
            )
    print_obj(func_name, "alpha_var", alpha_var)

    return alpha_var


def unknown_update_alpha_transition(global_step, alpha_var, params):
    """Returns ref for updated alpha variable.
    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Ref for updated alpha variable.
    """
    alpha_var = tf.assign(
        ref=alpha_var,
        value=tf.divide(
            x=tf.cast(
                # Add 1 since it trains on global step 0, so off by 1.
                x=tf.add(
                    x=tf.mod(
                        x=tf.subtract(
                            x=global_step,
                            y=params["previous_train_steps"]
                        ),
                        y=params["num_steps_until_growth"]
                    ),
                    y=1
                ),
                dtype=tf.float32
            ),
            y=params["num_steps_until_growth"]
        ),
        name="update_alpha_assign_linear"
    )

    return alpha_var


def unknown_update_alpha_stable(global_step, alpha_var, params):
    """Returns ref for updated alpha variable.
    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Ref for updated alpha variable.
    """
    alpha_var = tf.assign(
        ref=alpha_var,
        value=tf.ones(shape=[], dtype=tf.float32),
        name="update_alpha_assign_ones"
    )

    return alpha_var


def unknown_update_alpha(global_step, alpha_var, params):
    """Returns ref for updated alpha variable.
    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Ref for updated alpha variable.
    """
    func_name = "unknown_update_alpha"

    # If never grow, then no need to update alpha since it is not used.
    if len(params["conv_num_filters"]) > 1:
        # Find growth index based on global step and growth frequency.
        growth_index = tf.minimum(
            x=tf.cast(
                x=tf.add(
                    x=tf.floordiv(
                        x=tf.subtract(
                            x=global_step,
                            y=params["previous_train_steps"]
                        ),
                        y=params["num_steps_until_growth"]
                    ),
                    y=0 if not params["growth_idx"] else params["growth_idx"]
                ),
                dtype=tf.int32),
            y=(len(params["conv_num_filters"]) - 1) * 2,
            name="update_alpha_growth_index"
        )

        # True if this is a transition stage, False if this is a stable stage.
        condition = tf.equal(
            x=tf.mod(x=growth_index, y=2),
            y=1,
            name="{}_condition".format(func_name)
        )

        # Conditionally update alpha.
        alpha_var = tf.cond(
            pred=condition,
            true_fn=lambda: unknown_update_alpha_transition(
                global_step, alpha_var, params
            ),
            false_fn=lambda: unknown_update_alpha_stable(
                global_step, alpha_var, params
            ),
            name="{}_cond".format(func_name)
        )
    print_obj(func_name, "alpha_var", alpha_var)

    return alpha_var


def update_alpha(global_step, alpha_var, params):
    """Returns ref for updated alpha variable.
    Args:
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Ref for updated alpha variable.
    """
    func_name = "update_alpha"
    # If never grow, then no need to update alpha since it is not used.
    if params["growth_idx"] is not None:
        alpha_var = known_update_alpha(global_step, alpha_var, params)
    else:
        alpha_var = unknown_update_alpha(global_step, alpha_var, params)
    print_obj(func_name, "alpha_var", alpha_var)

    return alpha_var


def get_loss_and_train_op(
        generator_total_loss,
        discriminator_total_loss,
        alpha_var,
        params):
    """Gets loss and train op for train mode.
    Args:
        generator_total_loss: tensor, scalar total loss of generator.
        discriminator_total_loss: tensor, scalar total loss of discriminator.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
    Returns:
        Loss scalar tensor and train_op to be used by the EstimatorSpec.
    """
    func_name = "get_loss_and_train_op"
    # Get global step.
    global_step = tf.train.get_or_create_global_step()

    # Determine if it is time to train generator or discriminator.
    cycle_step = tf.mod(
        x=global_step,
        y=tf.cast(
            x=tf.add(
                x=params["discriminator_train_steps"],
                y=params["generator_train_steps"]
            ),
            dtype=tf.int64
        ),
        name="{}_cycle_step".format(func_name)
    )

    # Create choose discriminator condition.
    condition = tf.less(
        x=cycle_step, y=params["discriminator_train_steps"]
    )

    # Needed for batch normalization, but has no effect otherwise.
    update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

    # Ensure update ops get updated.
    with tf.control_dependencies(
            control_inputs=update_ops):
        # Conditionally choose to train generator or discriminator.
        loss, train_op = tf.cond(
            pred=condition,
            true_fn=lambda: train_discriminator(
                discriminator_loss=discriminator_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                discriminator_scope="discriminator"
            ),
            false_fn=lambda: train_generator(
                generator_loss=generator_total_loss,
                global_step=global_step,
                alpha_var=alpha_var,
                params=params,
                generator_scope="generator"
            ),
            name="{}_cond".format(func_name)
        )

    return loss, train_op
