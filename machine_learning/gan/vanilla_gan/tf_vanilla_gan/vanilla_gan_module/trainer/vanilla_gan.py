import tensorflow as tf

from . import discriminator
from . import generator
from .print_object import print_obj


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

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=tf.trainable_variables(scope=scope),
        name="{}_gradients".format(scope)
    )

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, tf.trainable_variables(scope=scope))

    # Get optimizer and instantiate it.
    optimizer = optimizers[params["{}_optimizer".format(scope)]](
        learning_rate=params["{}_learning_rate".format(scope)]
    )

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )

    return loss, train_op


def vanilla_gan_model(features, labels, mode, params):
    """Vanilla GAN custom Estimator model function.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Instance of `tf.estimator.EstimatorSpec` class.
    """
    print_obj("\nvanilla_gan_model", "features", features)
    print_obj("vanilla_gan_model", "labels", labels)
    print_obj("vanilla_gan_model", "mode", mode)
    print_obj("vanilla_gan_model", "params", params)

    # Loss function, training/eval ops, etc.
    predictions_dict = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract given latent vectors from features dictionary.
        Z = tf.cast(x=features["Z"], dtype=tf.float32)

        # Establish generator network subgraph.
        generator_outputs = generator.generator_network(
            Z, params, reuse=False
        )

        generated_images = tf.reshape(
            tensor=generator_outputs,
            shape=[-1, params["height"], params["width"], params["depth"]]
        )

        # Create predictions dictionary.
        predictions_dict = {
            "generated_images": generated_images
        }

        # Create export outputs.
        export_outputs = {
            "predict_export_outputs": tf.estimator.export.PredictOutput(
                outputs=predictions_dict)
        }
    else:
        # Extract image from features dictionary.
        X = tf.reshape(
            tensor=features["image"],
            shape=[-1, params["height"] * params["width"] * params["depth"]]
        )

        # Get dynamic batch size in case of partial batch.
        cur_batch_size = tf.shape(
            input=X,
            out_type=tf.int32,
            name="vanilla_gan_model_cur_batch_size"
        )[0]

        # Create random noise latent vector for each batch example.
        Z = tf.random.normal(
            shape=[cur_batch_size, params["latent_size"]],
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32
        )

        # Establish generator network subgraph.
        generator_outputs = generator.generator_network(
            Z, params, reuse=False
        )

        # Establish discriminator network subgraph.
        real_logits = discriminator.discriminator_network(
            X, params, reuse=False
        )

        # Get generated logits too.
        generated_logits = discriminator.discriminator_network(
            generator_outputs, params, reuse=True
        )

        # Get generator total loss.
        generator_total_loss = generator.get_generator_loss(generated_logits)

        # Get discriminator total loss.
        discriminator_total_loss = discriminator.get_discriminator_loss(
            generated_logits, real_logits
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Get global step.
            global_step = tf.train.get_global_step()

            # Determine if it is time to train generator or discriminator.
            cycle_step = tf.mod(
                x=global_step,
                y=tf.cast(
                    x=tf.add(
                        x=params["generator_train_steps"],
                        y=params["discriminator_train_steps"]
                    ),
                    dtype=tf.int64
                )
            )

            # Create choose generator condition.
            condition = tf.less(
                x=cycle_step, y=params["generator_train_steps"]
            )

            # Conditionally choose to train generator or discriminator subgraph.
            loss, train_op = tf.cond(
                pred=condition,
                true_fn=lambda: train_network(
                    loss=generator_total_loss,
                    global_step=global_step,
                    params=params,
                    scope="generator"
                ),
                false_fn=lambda: train_network(
                    loss=discriminator_total_loss,
                    global_step=global_step,
                    params=params,
                    scope="discriminator"
                )
            )
        else:
            loss = discriminator_total_loss

            # Concatenate discriminator logits and labels.
            discriminator_logits = tf.concat(
                values=[real_logits, generated_logits],
                axis=0,
                name="discriminator_concat_logits"
            )

            discriminator_labels = tf.concat(
                values=[
                    tf.ones_like(tensor=real_logits),
                    tf.zeros_like(tensor=generated_logits)
                ],
                axis=0,
                name="discriminator_concat_labels"
            )

            # Calculate discriminator probabilities.
            discriminator_probabilities = tf.nn.sigmoid(
                x=discriminator_logits, name="discriminator_probabilities"
            )

            # Create eval metric ops dictionary.
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="vanilla_gan_model_accuracy"
                ),
                "precision": tf.metrics.precision(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="vanilla_gan_model_precision"
                ),
                "recall": tf.metrics.recall(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="vanilla_gan_model_recall"
                ),
                "auc_roc": tf.metrics.auc(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    num_thresholds=200,
                    curve="ROC",
                    name="vanilla_gan_model_auc_roc"
                ),
                "auc_pr": tf.metrics.auc(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    num_thresholds=200,
                    curve="PR",
                    name="vanilla_gan_model_auc_pr"
                )
            }

    # Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
    )
