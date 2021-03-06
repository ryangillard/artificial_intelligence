import tensorflow as tf

from .print_object import print_obj


def get_loss_and_eval_metric_ops(
        discriminator_total_loss, fake_logits, real_logits):
    """Gets logits and losses for both train and eval modes.

    Args:
        discriminator_total_loss: tensor, scalar total loss of discriminator.
        fake_logits: tensor, shape of [cur_batch_size, 1] that came from
            discriminator having processed generator's output image.
        real_logits: tensor, shape of [cur_batch_size, 1] that came from
            discriminator having processed real image.

    Returns:
        Dictionary of eval metric ops.
    """
    loss = discriminator_total_loss

    # Concatenate discriminator logits and labels.
    discriminator_logits = tf.concat(
        values=[real_logits, fake_logits],
        axis=0,
        name="discriminator_concat_logits"
    )
    print_obj(
        "\get_eval_metric_ops", "discriminator_logits", discriminator_logits
    )

    discriminator_labels = tf.concat(
        values=[
            tf.ones_like(tensor=real_logits),
            tf.zeros_like(tensor=fake_logits)
        ],
        axis=0,
        name="discriminator_concat_labels"
    )
    print_obj(
        "get_eval_metric_ops", "discriminator_labels", discriminator_labels
    )

    # Calculate discriminator probabilities.
    discriminator_probabilities = tf.nn.sigmoid(
        x=discriminator_logits, name="discriminator_probabilities"
    )
    print_obj(
        "get_eval_metric_ops",
        "discriminator_probabilities",
        discriminator_probabilities
    )

    # Create eval metric ops dictionary.
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=discriminator_labels,
            predictions=discriminator_probabilities,
            name="discriminator_accuracy"
        ),
        "precision": tf.metrics.precision(
            labels=discriminator_labels,
            predictions=discriminator_probabilities,
            name="discriminator_precision"
        ),
        "recall": tf.metrics.recall(
            labels=discriminator_labels,
            predictions=discriminator_probabilities,
            name="discriminator_recall"
        ),
        "auc_roc": tf.metrics.auc(
            labels=discriminator_labels,
            predictions=discriminator_probabilities,
            num_thresholds=200,
            curve="ROC",
            name="discriminator_auc_roc"
        ),
        "auc_pr": tf.metrics.auc(
            labels=discriminator_labels,
            predictions=discriminator_probabilities,
            num_thresholds=200,
            curve="PR",
            name="discriminator_auc_pr"
        )
    }
    print_obj("get_eval_metric_ops", "eval_metric_ops", eval_metric_ops)

    return loss, eval_metric_ops
