import tensorflow as tf

from . import discriminator
from . import eval_metrics
from . import generator
from . import predict
from . import train
from . import train_and_eval
from .print_object import print_obj


def cgan_model(features, labels, mode, params):
    """Conditional GAN custom Estimator model function.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Instance of `tf.estimator.EstimatorSpec` class.
    """
    func_name = "cgan_model"
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
    cgan_generator = generator.Generator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["generator_l1_regularization_scale"],
            scale_l2=params["generator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="generator"
    )

    # Instantiate discriminator.
    cgan_discriminator = discriminator.Discriminator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["discriminator_l1_regularization_scale"],
            scale_l2=params["discriminator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="discriminator"
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Get predictions and export outputs.
        (predictions_dict,
         export_outputs) = predict.get_predictions_and_export_outputs(
            features=features, generator=cgan_generator, params=params
        )
    else:
        # Expand labels from vector to matrix.
        labels = tf.expand_dims(input=labels, axis=-1)
        print_obj(func_name, "labels", labels)

        # Get logits and losses from networks for train and eval modes.
        (real_logits,
         fake_logits,
         generator_total_loss,
         discriminator_total_loss) = train_and_eval.get_logits_and_losses(
            features=features,
            labels=labels,
            generator=cgan_generator,
            discriminator=cgan_discriminator,
            params=params
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create variable and gradient histogram summaries.
            train.create_variable_and_gradient_histogram_summaries(
                loss_dict={
                    "generator": generator_total_loss,
                    "discriminator": discriminator_total_loss
                },
                params=params
            )

            # Get loss and train op for EstimatorSpec.
            loss, train_op = train.get_loss_and_train_op(
                generator_total_loss=generator_total_loss,
                discriminator_total_loss=discriminator_total_loss,
                params=params
            )
        else:
            # Set eval loss.
            loss = discriminator_total_loss

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
