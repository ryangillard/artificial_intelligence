import tensorflow as tf

from .print_object import print_obj


def get_variables_and_gradients(loss, scope, params):
    """Gets variables and their gradients wrt. loss.
    Args:
        loss: tensor, shape of [].
        scope: str, the network's name to find its variables to train.
        params: dict, user passed parameters.
    Returns:
        Lists of variables and their gradients.
    """
    func_name = "get_variables_and_gradients"
    print_obj("\n" + func_name, "loss", loss)
    print_obj(func_name, "scope", scope)

    # Determine kind of network.
    network_kind = scope.split("/")[0]

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

    # Clip gradients by value.
    clip_by_value = params["{}_clip_gradients_by_value".format(network_kind)]
    if clip_by_value:
        gradients = [
            tf.clip_by_value(
                t=g,
                clip_value_min=-clip_by_value,
                clip_value_max=clip_by_value,
                name="{}_clip_by_by_value_gradients".format(scope)
            )
            if tf.is_tensor(x=g) else g
            for g, v in zip(gradients, variables)
        ]
        print_obj("\n{}_{}".format(func_name, scope), "gradients", gradients)

    # Clip gradients by global norm.
    clip_by_norm = params["{}_clip_gradients_global_norm".format(network_kind)]
    if clip_by_norm:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=clip_by_norm,
            name="{}_clip_by_global_norm_gradients".format(scope)
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
        loss_dict: dict, keys are kinds and values are dictionaries with
            keys are domains and values are scalar loss tensors for each
            network name.
        params: dict, user passed parameters.
    """
    func_name = "create_variable_and_gradient_histogram_summaries"
    print_obj("\n" + func_name, "loss_dict", loss_dict)
    # Loop through network kinds.
    for kind, kind_dict in loss_dict.items():
        # Loops through domains.
        for domain, loss in kind_dict.items():
            scope = "{}/domain_{}".format(kind, domain)

            # Get variables and their gradients wrt. loss.
            variables, gradients = get_variables_and_gradients(
                loss, scope, params
            )
            print_obj(
                "\n{}_{}_{}".format(func_name, kind, domain),
                "variables",
                variables
            )
            print_obj(
                "\n{}_{}_{}".format(func_name, kind, domain),
                "gradients",
                gradients
            )

            # Add summaries for TensorBoard.
            for g, v in zip(gradients, variables):
                tf.summary.histogram(
                    name="{}".format(v.name[:-2]),
                    values=v,
                    family="{}_domain_{}_variables".format(kind, domain)
                )
                if tf.is_tensor(x=g):
                    tf.summary.histogram(
                        name="{}".format(v.name[:-2]),
                        values=g,
                        family="{}_domain_{}_gradients".format(kind, domain)
                    )


def train_network(loss, global_step, params, scope):
    """Trains network and returns loss and train op.

    Args:
        loss: tensor, shape of [].
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        scope: str, the variables that to train.

    Returns:
        Loss tensor and training op.
    """
    func_name = "train_network"
    print_obj("\n" + func_name, "scope", scope)

    # Determine kind of network.
    network_kind = scope.split("/")[0]

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

    # Get effective learning rate.
    lr_global_step = tf.train.get_or_create_global_step()
    lr_decay_type = params["{}_learning_rate_decay_type".format(network_kind)]

    if lr_decay_type == "constant":
        learning_rate = params["{}_learning_rate".format(network_kind)]
    elif lr_decay_type == "polynomial":
        learning_rate = tf.train.polynomial_decay(
            learning_rate=params["{}_learning_rate".format(network_kind)],
            global_step=lr_global_step,
            decay_steps=params["{}_learning_rate_decay_steps".format(network_kind)],
            end_learning_rate=params["{}_learning_rate_end_learning_rate".format(network_kind)],
            power=params["{}_learning_rate_power".format(network_kind)],
            cycle=params["{}_learning_rate_cycle".format(network_kind)],
            name="{}_learning_rate_polynomial_decay".format(network_kind)
        )
    elif lr_decay_type == "exponential":
        learning_rate = tf.train.exponential_decay(
            learning_rate=params["{}_learning_rate".format(network_kind)],
            global_step=lr_global_step,
            decay_steps=params["{}_learning_rate_decay_steps".format(network_kind)],
            decay_rate=params["{}_learning_rate_decay_rate".format(network_kind)],
            staircase=params["{}_learning_rate_staircase".format(network_kind)],
            name="{}_learning_rate_exponential_decay".format(network_kind)
        )
    elif lr_decay_type == "cosine":
        learning_rate = tf.train.cosine_decay(
            learning_rate=params["{}_learning_rate".format(network_kind)],
            global_step=lr_global_step,
            decay_steps=params["{}_learning_rate_decay_steps".format(network_kind)],
            alpha=params["{}_learning_rate_alpha".format(network_kind)],
            name="{}_learning_rate_cosine_decay".format(network_kind)
        )
    elif lr_decay_type == "piecewise_polynomial":
        learning_rate = tf.where(
            condition=tf.less(
                x=lr_global_step,
                y=params["{}_learning_rate_constant_steps".format(network_kind)]
            ),
            x=params["{}_learning_rate".format(network_kind)],
            y=tf.train.polynomial_decay(
                learning_rate=params["{}_learning_rate".format(network_kind)],
                global_step=tf.subtract(
                    x=lr_global_step,
                    y=params["{}_learning_rate_constant_steps".format(network_kind)]
                ),
                decay_steps=params["{}_learning_rate_decay_steps".format(network_kind)],
                end_learning_rate=params["{}_learning_rate_end_learning_rate".format(network_kind)],
                power=params["{}_learning_rate_power".format(network_kind)],
                cycle=params["{}_learning_rate_cycle".format(network_kind)],
                name="{}_learning_rate_polynomial_decay".format(network_kind)
            )
        )
    else:
        learning_rate = params["{}_learning_rate".format(network_kind)]

    # Get optimizer and instantiate it.
    if params["{}_optimizer".format(network_kind)] == "Adam":
        optimizer = optimizers[params["{}_optimizer".format(network_kind)]](
            learning_rate=learning_rate,
            beta1=params["{}_adam_beta1".format(network_kind)],
            beta2=params["{}_adam_beta2".format(network_kind)],
            epsilon=params["{}_adam_epsilon".format(network_kind)],
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(network_kind)].lower()
            )
        )
    else:
        optimizer = optimizers[params["{}_optimizer".format(network_kind)]](
            learning_rate=learning_rate,
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(network_kind)].lower()
            )
        )
    print_obj("{}_{}".format(func_name, scope), "optimizer", optimizer)

    # Get variables and gradients.
    variables, gradients = get_variables_and_gradients(loss, scope, params)

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, variables)
    print_obj(
        "{}_{}".format(func_name, scope), "grads_and_vars", grads_and_vars
    )

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )

    return loss, train_op


def train_network_kind(loss_dict, global_step, params, kind):
    """Gets loss and train op for train mode for network kind.

    Args:
        loss_dict: dict, keys are domains and values are scalar loss tensors
            for each network name.
        global_step: tensor, the current training step or batch in the
            training loop.
        params: dict, user passed parameters.
        kind: str, the kind of network, generator or discriminator.
    Returns:
        Loss scalar tensor and train_op to be used by the EstimatorSpec.
    """
    losses = []
    train_ops = []

    # Pass global step once so that apply gradients only increments it once.
    global_steps = [None for _ in list(loss_dict.keys())[:-1]] + [global_step]

    # Train network for each domain.
    for (domain, loss), g_step in zip(list(loss_dict.items()), global_steps):
        loss_domain, train_op_domain = train_network(
            loss=loss,
            global_step=g_step,
            params=params,
            scope="{}/domain_{}".format(kind, domain)
        )

        # Append to lists.
        losses.append(loss_domain)
        train_ops.append(train_op_domain)

    # Combine losses.
    loss = losses[0] + losses[1]

    # Group train ops.
    train_op = tf.group(train_ops, name="{}_train_op".format(kind))

    return loss, train_op


def get_loss_and_train_op(loss_dict, params):
    """Gets loss and train op for train mode.

    Args:
        loss_dict: dict, keys are scopes and values are scalar loss tensors
            for each network name.
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
        name="{}_learning_rate_cycle_step".format(func_name)
    )

    # Create choose discriminator condition.
    condition = tf.less(
        x=cycle_step, y=params["discriminator_train_steps"]
    )

    # Needed for batch normalization, but has no effect otherwise.
    update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

    # Ensure update ops get updated.
    with tf.control_dependencies(control_inputs=update_ops):
        # Conditionally choose to train generator or discriminator subgraph.
        loss, train_op = tf.cond(
            pred=condition,
            true_fn=lambda: train_network_kind(
                loss_dict=loss_dict["discriminator"],
                global_step=global_step,
                params=params,
                kind="discriminator"
            ),
            false_fn=lambda: train_network_kind(
                loss_dict=loss_dict["generator"],
                global_step=global_step,
                params=params,
                kind="generator"
            )
        )

    return loss, train_op
