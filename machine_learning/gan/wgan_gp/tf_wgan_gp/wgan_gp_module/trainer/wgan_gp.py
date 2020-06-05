import tensorflow as tf

from . import critic
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
    print_obj("\ntrain_network", "loss", loss)
    print_obj("train_network", "global_step", global_step)
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

    # Get trainable variables.
    variables = tf.trainable_variables(scope=scope)
    print_obj("train_network", "variables", variables)

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=variables,
        name="{}_gradients".format(scope)
    )
    print_obj("train_network", "gradients", gradients)

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )
        print_obj("train_network", "gradients", gradients)

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, variables)
    print_obj("train_network", "grads_and_vars", grads_and_vars)

    # Get optimizer and instantiate it.
    optimizer = optimizers[params["{}_optimizer".format(scope)]](
        learning_rate=params["{}_learning_rate".format(scope)]
    )
    print_obj("train_network", "optimizer", optimizer)

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )
    print_obj("train_network", "train_op", train_op)

    return loss, train_op


def wgan_gp_model(features, labels, mode, params):
    """Wasserstein GAN with gradient penalty custom Estimator model function.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Instance of `tf.estimator.EstimatorSpec` class.
    """
    print_obj("\nwgan_model", "features", features)
    print_obj("wgan_model", "labels", labels)
    print_obj("wgan_model", "mode", mode)
    print_obj("wgan_model", "params", params)

    # Loss function, training/eval ops, etc.
    predictions_dict = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract given latent vectors from features dictionary.
        Z = tf.cast(x=features["Z"], dtype=tf.float32)

        # Get predictions from generator.
        generated_images = generator.generator_network(
            Z, mode, params, reuse=False
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
        X = features["image"]

        # Get dynamic batch size in case of partial batch.
        cur_batch_size = tf.shape(
            input=X,
            out_type=tf.int32,
            name="wgan_model_cur_batch_size"
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
            Z, mode, params, reuse=False
        )

        # Establish critic network subgraph.
        real_logits = critic.critic_network(X, params, reuse=False)

        # Get generated logits too.
        fake_logits = critic.critic_network(
            generator_outputs, params, reuse=True
        )

        # Get generator total loss.
        generator_total_loss = generator.get_generator_loss(fake_logits)

        # Get critic total loss.
        critic_total_loss = critic.get_critic_loss(
            cur_batch_size=cur_batch_size,
            fake_images=generator_outputs,
            real_images=X,
            fake_logits=fake_logits,
            real_logits=real_logits,
            params=params
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Get global step.
            global_step = tf.train.get_global_step()

            # Determine if it is time to train generator or critic.
            cycle_step = tf.mod(
                x=global_step,
                y=tf.cast(
                    x=tf.add(
                        x=params["generator_train_steps"],
                        y=params["critic_train_steps"]
                    ),
                    dtype=tf.int64
                )
            )

            # Create choose generator condition.
            condition = tf.less(
                x=cycle_step, y=params["generator_train_steps"]
            )

            # Needed for batch normalization, but has no effect otherwise.
            update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(control_inputs=update_ops):
                # Conditionally choose to train generator or critic.
                loss, train_op = tf.cond(
                    pred=condition,
                    true_fn=lambda: train_network(
                        loss=generator_total_loss,
                        global_step=global_step,
                        params=params,
                        scope="generator"
                    ),
                    false_fn=lambda: train_network(
                        loss=critic_total_loss,
                        global_step=global_step,
                        params=params,
                        scope="critic"
                    )
                )
        else:
            loss = critic_total_loss

            # Concatenate critic logits and labels.
            critic_logits = tf.concat(
                values=[real_logits, fake_logits],
                axis=0,
                name="critic_concat_logits"
            )

            critic_labels = tf.concat(
                values=[
                    tf.ones_like(tensor=real_logits),
                    tf.zeros_like(tensor=fake_logits)
                ],
                axis=0,
                name="critic_concat_labels"
            )

            # Calculate critic probabilities.
            critic_probabilities = tf.nn.sigmoid(
                x=critic_logits, name="critic_probabilities"
            )

            # Create eval metric ops dictionary.
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=critic_labels,
                    predictions=critic_probabilities,
                    name="wgan_model_accuracy"
                ),
                "precision": tf.metrics.precision(
                    labels=critic_labels,
                    predictions=critic_probabilities,
                    name="wgan_model_precision"
                ),
                "recall": tf.metrics.recall(
                    labels=critic_labels,
                    predictions=critic_probabilities,
                    name="wgan_model_recall"
                ),
                "auc_roc": tf.metrics.auc(
                    labels=critic_labels,
                    predictions=critic_probabilities,
                    num_thresholds=200,
                    curve="ROC",
                    name="wgan_model_auc_roc"
                ),
                "auc_pr": tf.metrics.auc(
                    labels=critic_labels,
                    predictions=critic_probabilities,
                    num_thresholds=200,
                    curve="PR",
                    name="wgan_model_auc_pr"
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
