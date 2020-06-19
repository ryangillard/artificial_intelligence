import tensorflow as tf

from . import critic
from . import eval_metrics
from . import generator
from . import predict
from . import train
from . import train_and_eval
from .print_object import print_obj


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
    func_name = "wgan_gp_model"
    print_obj("\n" + func_name, "features", features)
    print_obj(func_name, "labels", labels)
    print_obj(func_name, "mode", mode)
    print_obj(func_name, "params", params)

    # Loss function, training/eval ops, etc.
    predictions_dict = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    # Instantiate generator.
    wgan_generator = generator.Generator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["generator_l1_regularization_scale"],
            scale_l2=params["generator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="generator"
    )

    # Instantiate critic.
    wgan_critic = critic.Critic(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["critic_l1_regularization_scale"],
            scale_l2=params["critic_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="critic"
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Get predictions and export outputs.
        (predictions_dict,
         export_outputs) = predict.get_predictions_and_export_outputs(
            features=features, generator=wgan_generator, params=params
        )
    else:
        # Get logits and losses from networks for train and eval modes.
        (real_logits,
         fake_logits,
         generator_total_loss,
         critic_total_loss) = train_and_eval.get_logits_and_losses(
            features=features,
            generator=wgan_generator,
            critic=wgan_critic,
            mode=mode,
            params=params
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create variable and gradient histogram summaries.
            train.create_variable_and_gradient_histogram_summaries(
                loss_dict={
                    "generator": generator_total_loss,
                    "critic": critic_total_loss
                },
                params=params
            )

            # Get loss and train op for EstimatorSpec.
            loss, train_op = train.get_loss_and_train_op(
                generator_total_loss=generator_total_loss,
                critic_total_loss=critic_total_loss,
                params=params
            )
        else:
            # Set eval loss.
            loss = critic_total_loss

            # Get eval metrics.
            eval_metric_ops = eval_metrics.get_eval_metric_ops(
                real_logits=real_logits,
                fake_logits=fake_logits,
                params=params
            )

    # Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
    )
