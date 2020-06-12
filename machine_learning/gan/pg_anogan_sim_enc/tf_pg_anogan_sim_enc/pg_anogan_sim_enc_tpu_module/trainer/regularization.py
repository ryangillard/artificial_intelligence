import tensorflow as tf

from .print_object import print_obj


def get_regularization_loss(lambda1=0., lambda2=0., scope=None):
    """Gets regularization losses from variables attached to a regularizer.

    Args:
        lambda1: float, L1 regularization scale parameter.
        lambda2: float, L2 regularization scale parameter.
        scope: str, the name of the variable scope.

    Returns:
        Scalar regularization loss tensor.
    """
    def sum_nd_tensor_list_to_scalar_tensor(t_list):
        """Sums different shape tensors into a scalar tensor.

        Args:
            t_list: list, tensors of varying shapes.

        Returns:
            Scalar tensor.
        """
        func_name = "sum_nd_tensor_list_to_scalar_tensor"
        # Sum list of tensors into a list of scalars.
        t_reduce_sum_list = [
            tf.reduce_sum(
                # Remove the :0 from the end of the name.
                input_tensor=t, name="{}_reduce_sum".format(t.name[:-2])
            )
            for t in t_list
        ]
        print_obj("\n" + func_name, "t_reduce_sum_list", t_reduce_sum_list)

        # Add all scalars together into one scalar.
        t_scalar_sum_tensor = tf.add_n(
            inputs=t_reduce_sum_list,
            name="{}_t_scalar_sum_tensor".format(scope)
        )
        print_obj(func_name, "t_scalar_sum_tensor", t_scalar_sum_tensor)

        return t_scalar_sum_tensor

    func_name = "get_regularization_loss"
    print_obj("\n" + func_name, "scope", scope)
    if lambda1 <= 0. and lambda2 <= 0.:
        # No regularization so return zero.
        return tf.zeros(shape=[], dtype=tf.float32)

    # Get list of trainable variables with a regularizer attached in scope.
    trainable_reg_vars_list = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
    print_obj(
        func_name, "trainable_reg_vars_list", trainable_reg_vars_list
    )

    for var in trainable_reg_vars_list:
        print_obj(
            "{}_{}".format(func_name, scope), "{}".format(var.name), var.graph
        )

    l1_loss = 0.
    if lambda1 > 0.:
        # For L1 regularization, take the absolute value element-wise of each.
        trainable_reg_vars_abs_list = [
            tf.abs(
                x=var,
                # Clean up regularizer scopes in variable names.
                name="{}_abs".format(("/").join(var.name.split("/")[0:3]))
            )
            for var in trainable_reg_vars_list
        ]

        # Get L1 loss
        l1_loss = tf.multiply(
            x=lambda1,
            y=sum_nd_tensor_list_to_scalar_tensor(
                t_list=trainable_reg_vars_abs_list
            ),
            name="{}_l1_loss".format(scope)
        )

    l2_loss = 0.
    if lambda2 > 0.:
        # For L2 regularization, square all variables element-wise.
        trainable_reg_vars_squared_list = [
            tf.square(
                x=var,
                # Clean up regularizer scopes in variable names.
                name="{}_squared".format(("/").join(var.name.split("/")[0:3]))
            )
            for var in trainable_reg_vars_list
        ]
        print_obj(
            func_name,
            "trainable_reg_vars_squared_list",
            trainable_reg_vars_squared_list
        )

        # Get L2 loss
        l2_loss = tf.multiply(
            x=lambda2,
            y=sum_nd_tensor_list_to_scalar_tensor(
                t_list=trainable_reg_vars_squared_list
            ),
            name="{}_l2_loss".format(scope)
        )

    l1_l2_loss = tf.add(
        x=l1_loss, y=l2_loss, name="{}_l1_l2_loss".format(scope)
    )

    return l1_l2_loss
