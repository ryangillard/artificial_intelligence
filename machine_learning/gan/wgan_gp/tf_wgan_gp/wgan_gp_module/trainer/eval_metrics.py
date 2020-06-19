import tensorflow as tf

from .print_object import print_obj


def get_eval_metric_ops(fake_logits, real_logits, params):
    """Gets eval metric ops.

    Args:
        fake_logits: tensor, shape of [cur_batch_size, 1] that came from
            critic having processed generator's output image.
        real_logits: tensor, shape of [cur_batch_size, 1] that came from
            critic having processed real image.
        params: dict, user passed parameters.

    Returns:
        Dictionary of eval metric ops.
    """
    func_name = "get_eval_metric_ops"
    # Concatenate critic logits and labels.
    critic_logits = tf.concat(
        values=[real_logits, fake_logits],
        axis=0,
        name="critic_concat_logits"
    )
    print_obj("\n" + func_name, "critic_logits", critic_logits)

    critic_labels = tf.concat(
        values=[
            tf.ones_like(tensor=real_logits),
            tf.zeros_like(tensor=fake_logits)
        ],
        axis=0,
        name="critic_concat_labels"
    )
    print_obj(func_name, "critic_labels", critic_labels)

    # Calculate critic probabilities.
    critic_probabilities = tf.nn.sigmoid(
        x=critic_logits, name="critic_probabilities"
    )
    print_obj(
        func_name, "critic_probabilities", critic_probabilities
    )

    # Create eval metric ops dictionary.
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=critic_labels,
            predictions=critic_probabilities,
            name="critic_accuracy"
        ),
        "precision": tf.metrics.precision(
            labels=critic_labels,
            predictions=critic_probabilities,
            name="critic_precision"
        ),
        "recall": tf.metrics.recall(
            labels=critic_labels,
            predictions=critic_probabilities,
            name="critic_recall"
        ),
        "auc_roc": tf.metrics.auc(
            labels=critic_labels,
            predictions=critic_probabilities,
            num_thresholds=200,
            curve="ROC",
            name="critic_auc_roc"
        ),
        "auc_pr": tf.metrics.auc(
            labels=critic_labels,
            predictions=critic_probabilities,
            num_thresholds=200,
            curve="PR",
            name="critic_auc_pr"
        )
    }
    print_obj(func_name, "eval_metric_ops", eval_metric_ops)

    return eval_metric_ops
